"""Readable local viewer for six-camera and synthetic-scene detection audits."""
import json
import os
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .api import CLASSES, MODEL_PATHS, REPO


def interpret(camera):
    m = camera['metrics']['overall']
    gt = sum(camera['visible_ground_truth'].values())
    pred = sum(camera['predicted'].values())
    if gt == 0:
        return 'No labelled instruments visible in this camera; inspect false positives.'
    if m['FN'] > m['TP']:
        return 'More labelled instances missed than found in this view.'
    if m['FP'] > m['TP']:
        return 'More extra predictions than correct matches in this view.'
    if m['FN'] == 0 and m['FP'] == 0:
        return 'All labelled visible instances matched in this view.'
    return f'{m["TP"]} of {gt} visible labels matched; {m["FP"]} extra predictions. Predicted total: {pred}.'


class Studio:
    def __init__(self, root):
        self.root = root
        root.title('Instrument Detection Studio | A / B')
        root.geometry('1350x880')
        root.minsize(1000, 700)
        self.model = tk.StringVar(value='B')
        self.threshold = tk.StringVar(value='0.50')
        self.path = tk.StringVar()
        self.source = tk.StringVar(value='synthetic')
        self.frame = tk.StringVar(value='0')
        self.overlay_mode = tk.StringVar(value='Model prediction')
        self.save_root = tk.StringVar(value=str(REPO.parent / 'datasets' / 'Object datasets' / 'inference_studio'))
        self.status = tk.StringVar(value='Choose a synthetic scene or DP episode, then run detection.')
        self.report = None
        self.output = None
        self.preview_ref = None
        self.busy = False
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('.', font=('DejaVu Sans', 10))
        style.configure('Title.TLabel', font=('DejaVu Sans', 19, 'bold'))
        style.configure('Metric.TLabel', font=('DejaVu Sans', 12, 'bold'))
        shell = ttk.Frame(root, padding=16)
        shell.pack(fill='both', expand=True)
        ttk.Label(shell, text='Instrument Detection Studio', style='Title.TLabel').pack(anchor='w')
        ttk.Label(shell, text='Per-camera evidence · same A/B checkpoint choices · mask IoU ≥ 0.50').pack(anchor='w', pady=(2, 12))
        toolbar = ttk.Frame(shell)
        toolbar.pack(fill='x')
        for label, value in (('Model', self.model), ('Source', self.source)):
            ttk.Label(toolbar, text=label).pack(side='left', padx=(0, 5))
            options = ('A', 'B') if label == 'Model' else ('synthetic', 'dp')
            ttk.Combobox(toolbar, textvariable=value, values=options, state='readonly', width=11).pack(side='left', padx=(0, 15))
        ttk.Label(toolbar, text='Score threshold').pack(side='left', padx=(0, 5))
        ttk.Entry(toolbar, textvariable=self.threshold, width=6).pack(side='left', padx=(0, 15))
        ttk.Label(toolbar, text='DP frame').pack(side='left', padx=(0, 5))
        ttk.Entry(toolbar, textvariable=self.frame, width=6).pack(side='left', padx=(0, 15))
        ttk.Button(toolbar, text='Browse source', command=self.browse).pack(side='left', padx=4)
        ttk.Button(toolbar, text='Open scene recorder', command=self.open_recorder).pack(side='left', padx=4)
        ttk.Entry(shell, textvariable=self.path).pack(fill='x', pady=(10, 6))
        save_row = ttk.Frame(shell)
        save_row.pack(fill='x', pady=(0, 6))
        ttk.Label(save_row, text='Save reports to').pack(side='left', padx=(0, 8))
        ttk.Entry(save_row, textvariable=self.save_root).pack(side='left', fill='x', expand=True)
        ttk.Button(save_row, text='Choose', command=self.choose_output).pack(side='left', padx=(8, 0))
        action = ttk.Frame(shell)
        action.pack(fill='x')
        self.run_button = ttk.Button(action, text='Run detection on every camera', command=self.run)
        self.run_button.pack(side='left')
        ttk.Label(action, textvariable=self.status, wraplength=850).pack(side='left', padx=15)
        self.meta = tk.Text(shell, height=4, wrap='word', bg='white', relief='flat')
        self.meta.pack(fill='x', pady=(12, 7))
        self.meta.insert('1.0', 'Spawned on table and visible ground truth are different quantities. Each camera is evaluated on what its image can see.')
        self.meta.configure(state='disabled')
        body = ttk.Panedwindow(shell, orient='horizontal')
        body.pack(fill='both', expand=True)
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=2)
        body.add(right, weight=3)
        ttk.Label(left, text='Camera results', style='Metric.TLabel').pack(anchor='w', pady=(0, 6))
        self.camera_list = tk.Listbox(left, font=('DejaVu Sans', 11), height=14, exportselection=False)
        self.camera_list.pack(fill='x')
        self.camera_list.bind('<<ListboxSelect>>', self.show_camera)
        self.details = tk.Text(left, wrap='word', font=('DejaVu Sans Mono', 10), bg='white')
        self.details.pack(fill='both', expand=True, pady=(8, 0))
        preview_toolbar = ttk.Frame(right)
        preview_toolbar.pack(fill='x', pady=(0, 6))
        ttk.Label(preview_toolbar, text='Image overlay').pack(side='left', padx=(0, 8))
        ttk.Combobox(preview_toolbar, textvariable=self.overlay_mode,
                     values=('Model prediction', 'Ground truth'), state='readonly', width=20).pack(side='left')
        self.overlay_mode.trace_add('write', lambda *_: self.show_camera())
        self.preview = ttk.Label(right, text='Camera preview with predicted boxes and masks')
        self.preview.pack(fill='both', expand=True)
        ttk.Label(shell, text='One camera = one conclusion. No pooled success percentage. DP labels can be partial; inspect missing labels before interpreting false positives.', wraplength=1200).pack(anchor='w', pady=(8, 0))

    def browse(self):
        if self.source.get() == 'dp':
            value = filedialog.askopenfilename(title='Choose DP episode', filetypes=[('HDF5 episode', '*.h5 *.hdf5')])
        else:
            value = filedialog.askdirectory(title='Choose scene folder containing scene.json')
        if value:
            self.path.set(value)

    def choose_output(self):
        value = filedialog.askdirectory(title='Choose report directory')
        if value:
            self.save_root.set(value)

    def open_recorder(self):
        launcher = REPO / 'scripts/launchers/run_detection6.sh'
        try:
            subprocess.Popen([str(launcher), 'panel'], cwd=REPO, start_new_session=True)
            self.status.set('Recorder opened. Choose object count, camera positions and wide lighting; load a saved scene folder here after capture.')
        except OSError as exc:
            messagebox.showerror('Recorder could not start', str(exc))

    def run(self):
        if self.busy:
            return
        source = self.source.get()
        path = Path(self.path.get()).expanduser()
        if source == 'synthetic' and not (path / 'scene.json').is_file():
            messagebox.showerror('Choose a scene', 'The synthetic scene folder must contain scene.json.')
            return
        if source == 'dp' and not path.is_file():
            messagebox.showerror('Choose an episode', 'Choose an existing DP HDF5 episode.')
            return
        if not MODEL_PATHS[self.model.get()].is_file() or MODEL_PATHS[self.model.get()].stat().st_size < 1000000:
            messagebox.showerror('Model unavailable', 'Download Git LFS checkpoints first: git lfs pull')
            return
        try:
            threshold = float(self.threshold.get())
            frame = int(self.frame.get())
            if not 0 <= threshold <= 1 or frame < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror('Invalid value', 'Threshold must be 0–1 and DP frame must be a nonnegative integer.')
            return
        self.output = Path(self.save_root.get()).expanduser() / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.output.mkdir(parents=True, exist_ok=False)
        python = os.environ.get('NSTC_RFDETR_PYTHON', sys.executable)
        command = [python, '-m', 'instrument_bridge.studio_worker', '--source', source,
                   '--path', str(path), '--output', str(self.output), '--model', self.model.get(),
                   '--threshold', str(threshold), '--frame', str(frame)]
        self.busy = True
        self.run_button.configure(state='disabled')
        self.status.set('Running model; each camera may take several seconds…')
        threading.Thread(target=self._worker, args=(command,), daemon=True).start()

    def _worker(self, command):
        try:
            done = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=True)
            report = json.loads((self.output / 'report.json').read_text())
            self.root.after(0, lambda: self.display(report))
        except Exception as exc:
            stderr = getattr(exc, 'stderr', '') or ''
            error_text = f'{exc}\n{stderr[-2200:]}'
            self.root.after(0, lambda: messagebox.showerror('Detection failed', error_text))
        finally:
            self.root.after(0, self._idle)

    def _idle(self):
        self.busy = False
        self.run_button.configure(state='normal')

    def display(self, report):
        self.report = report
        self.camera_list.delete(0, 'end')
        for row in report['per_camera']:
            metric = row['metrics']['overall']
            self.camera_list.insert('end', f"{row['name']:<18}  F1 {metric['F1'] * 100:5.1f}%  ·  TP {metric['TP']}  FP {metric['FP']}  FN {metric['FN']}")
        self.meta.configure(state='normal')
        self.meta.delete('1.0', 'end')
        self.meta.insert('1.0', f"MODEL {report['model']} | {report['source'].upper()} | threshold {report['threshold']:.2f}\n"
                         f"Spawned on table (all, including hidden): {report['spawn_on_table']}\n"
                         f"Lighting metadata: {str(report.get('lighting'))[:220] if report.get('lighting') else 'See DP episode metadata / not present'}\n"
                         f"Report: {self.output / 'report.json'}")
        self.meta.configure(state='disabled')
        self.camera_list.selection_set(0)
        self.show_camera()
        self.status.set(f"Finished {len(report['per_camera'])} cameras. Select each row for its own conclusion.")

    def show_camera(self, event=None):
        if not self.report or not self.camera_list.curselection():
            return
        row = self.report['per_camera'][self.camera_list.curselection()[0]]
        m = row['metrics']['overall']
        lines = [row['name'].upper(), interpret(row), '',
                 f"Visible labels: {sum(row['visible_ground_truth'].values())}",
                 f"Predicted objects: {sum(row['predicted'].values())}",
                 f"Precision: {m['precision'] * 100:.1f}%  Recall: {m['recall'] * 100:.1f}%  F1: {m['F1'] * 100:.1f}%",
                 f"Correct matches: {m['TP']}  Extra: {m['FP']}  Missed: {m['FN']}", '',
                 'CLASS              SPAWN   VISIBLE   PRED   F1']
        for name in CLASSES:
            c = row['metrics']['per_class'][name]
            lines.append(f"{name:<19}{self.report['spawn_on_table'][name]:>3}{c['visible_gt']:>10}{c['predicted']:>7}{c['F1'] * 100:>7.1f}%")
        lines.extend(['', 'Predicted instances:'])
        for obj in row['objects']:
            lines.append(f"{obj['class_name']:<18} score {obj['score']:.2f}  center ({obj['center_xy'][0]:.0f}, {obj['center_xy'][1]:.0f})")
        self.details.delete('1.0', 'end')
        self.details.insert('1.0', '\n'.join(lines))
        filename = row['preview'] if self.overlay_mode.get() == 'Model prediction' else row['ground_truth_preview']
        image = Image.open(self.output / filename).convert('RGB')
        image.thumbnail((680, 680))
        self.preview_ref = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.preview_ref, text='')


def main():
    root = tk.Tk()
    Studio(root)
    root.mainloop()


if __name__ == '__main__':
    main()
