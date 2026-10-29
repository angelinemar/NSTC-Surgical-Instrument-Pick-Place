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
from .target_handoff import build_request, matching_indices, pick_at_pixel, requested_class, save_request


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
        root.geometry('1450x980')
        root.minsize(1150, 850)
        self.model = tk.StringVar(value='B')
        self.threshold = tk.StringVar(value='0.50')
        self.path = tk.StringVar()
        self.source = tk.StringVar(value='synthetic')
        self.frame = tk.StringVar(value='0')
        self.overlay_mode = tk.StringVar(value='Model prediction')
        self.save_root = tk.StringVar(value=str(REPO.parent / 'datasets' / 'Object datasets' / 'inference_studio'))
        self.instruction = tk.StringVar(value='pick up the scalpel')
        self.action_backend = tk.StringVar(value='OpenVLA adapter')
        self.adapter_python = tk.StringVar(value=sys.executable)
        self.adapter_script = tk.StringVar()
        self.status = tk.StringVar(value='Choose a synthetic scene or DP episode, then run detection.')
        self.report = None
        self.output = None
        self.preview_ref = None
        self.busy = False
        self.locked_index = None
        self.locked_camera = None
        self.request_path = None
        self.display_size = (0, 0)
        self.updating_object_list = False
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
        ttk.Button(toolbar, text='Use latest recorded scene', command=self.latest_scene).pack(side='left', padx=4)
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
        ttk.Label(left, text='Detected instances in selected camera').pack(anchor='w', pady=(8, 2))
        self.object_list = tk.Listbox(left, font=('DejaVu Sans', 10), height=6, exportselection=False)
        self.object_list.pack(fill='x')
        self.object_list.bind('<<ListboxSelect>>', self.choose_from_list)
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
        self.preview.bind('<Button-1>', self.choose_from_image)
        handoff = ttk.LabelFrame(shell, text='Target lock and action handoff', padding=8)
        handoff.pack(fill='x', pady=(8, 0))
        first = ttk.Frame(handoff)
        first.pack(fill='x')
        ttk.Label(first, text='Language command').pack(side='left', padx=(0, 6))
        ttk.Entry(first, textvariable=self.instruction, width=42).pack(side='left', padx=(0, 8))
        ttk.Button(first, text='Select by command', command=self.choose_from_language).pack(side='left', padx=4)
        ttk.Button(first, text='Lock selected target', command=self.lock_target).pack(side='left', padx=4)
        self.lock_status = tk.StringVar(value='No target locked. Click a predicted instrument, choose the list, or enter a command.')
        ttk.Label(handoff, textvariable=self.lock_status, wraplength=1200).pack(anchor='w', pady=4)
        second = ttk.Frame(handoff)
        second.pack(fill='x')
        ttk.Combobox(second, textvariable=self.action_backend, values=('OpenVLA adapter', 'DP adapter'), state='readonly', width=18).pack(side='left', padx=(0, 8))
        ttk.Label(second, text='Adapter Python').pack(side='left')
        ttk.Entry(second, textvariable=self.adapter_python, width=23).pack(side='left', padx=(4, 8))
        ttk.Label(second, text='Adapter script').pack(side='left')
        ttk.Entry(second, textvariable=self.adapter_script, width=38).pack(side='left', fill='x', expand=True, padx=4)
        ttk.Button(second, text='Browse', command=self.choose_adapter).pack(side='left', padx=4)
        ttk.Button(second, text='Run adapter', command=self.run_adapter).pack(side='left', padx=4)
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

    def choose_adapter(self):
        value = filedialog.askopenfilename(title='Choose your OpenVLA or DP adapter Python script', filetypes=[('Python script', '*.py')])
        if value:
            self.adapter_script.set(value)

    def latest_scene(self):
        state = REPO / 'debug' / 'detection_panel_state.json'
        if not state.is_file():
            messagebox.showinfo('No recorder session', 'Record a scene first, or browse to an existing scene folder.')
            return
        try:
            collection = Path(json.loads(state.read_text())['folder'])
            scenes = sorted((collection / 'train' / 'scenes').glob('scene_*/scene.json'))
            if not scenes:
                raise FileNotFoundError('No completed scene in the selected recorder collection')
            self.source.set('synthetic')
            self.path.set(str(scenes[-1].parent))
            self.status.set(f'Loaded latest recorded scene: {scenes[-1].parent.name}')
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror('Cannot load scene', str(exc))

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
        self.locked_index = None
        self.locked_camera = None
        self.request_path = None
        self.lock_status.set('No target locked. Waiting for fresh detections.')
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
        if self.locked_camera != row['name']:
            self.locked_index = None
            self.locked_camera = None
            self.request_path = None
            self.lock_status.set('No target locked in this camera. Choose one of its predicted instances.')
        self.updating_object_list = True
        self.object_list.delete(0, 'end')
        for index, obj in enumerate(row['objects']):
            self.object_list.insert('end', f"#{index + 1}  {obj['class_name']:<18} score {obj['score']:.2f}  ({obj['center_xy'][0]:.0f}, {obj['center_xy'][1]:.0f})")
        self.updating_object_list = False
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
        self.display_size = image.size
        self.preview_ref = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.preview_ref, text='')

    def current_camera(self):
        if not self.report or not self.camera_list.curselection():
            raise ValueError('Run detection and choose a camera first')
        return self.report['per_camera'][self.camera_list.curselection()[0]]

    def select_index(self, index, method):
        row = self.current_camera()
        if not 0 <= index < len(row['objects']):
            raise ValueError('Select a predicted instance')
        self.updating_object_list = True
        self.object_list.selection_clear(0, 'end')
        self.object_list.selection_set(index)
        self.object_list.see(index)
        self.updating_object_list = False
        self.locked_index = index
        self.locked_camera = row['name']
        self.request_path = None
        self.selected_by = method
        obj = row['objects'][index]
        self.lock_status.set(f"Selected {obj['class_name']} #{index + 1} in {row['name']} (score {obj['score']:.2f}). Click Lock selected target to export the handoff JSON.")

    def choose_from_list(self, event=None):
        if not self.updating_object_list and self.object_list.curselection():
            index = self.object_list.curselection()[0]
            if self.locked_index != index or self.locked_camera != self.current_camera()['name']:
                self.select_index(index, 'list')

    def choose_from_image(self, event):
        if self.overlay_mode.get() != 'Model prediction':
            messagebox.showinfo('Use model overlay', 'Switch Image overlay to Model prediction to select a model detection.')
            return
        try:
            row = self.current_camera()
            width, height = self.display_size
            x = event.x - (self.preview.winfo_width() - width) / 2
            y = event.y - (self.preview.winfo_height() - height) / 2
            source_width = row['objects'][0]['mask_rle']['size'][1] if row['objects'] else width
            source_height = row['objects'][0]['mask_rle']['size'][0] if row['objects'] else height
            index = pick_at_pixel(row['objects'], x * source_width / width, y * source_height / height)
            if index is None:
                self.lock_status.set('No predicted box at that pixel. Choose an instance from the list.')
            else:
                self.select_index(index, 'click')
        except ValueError as exc:
            messagebox.showerror('Target selection', str(exc))

    def choose_from_language(self):
        try:
            row = self.current_camera()
            class_name = requested_class(self.instruction.get())
            candidates = matching_indices(row['objects'], class_name)
            if len(candidates) == 1:
                self.select_index(candidates[0], 'language_unique')
            elif not candidates:
                self.lock_status.set(f'No {class_name} detected in {row["name"]}. Choose another camera or inspect detections.')
            else:
                self.lock_status.set(f'{len(candidates)} {class_name} instances detected in {row["name"]}. Click the intended one or choose it in the list.')
        except ValueError as exc:
            messagebox.showerror('Language selection', str(exc))

    def lock_target(self):
        try:
            row = self.current_camera()
            if self.locked_camera != row['name'] or self.locked_index is None:
                raise ValueError('Choose a predicted instance in this camera first')
            request = build_request(self.report, row['name'], self.locked_index,
                                    self.instruction.get(), self.selected_by,
                                    self.output / row['rgb'])
            request['action_backend_requested'] = self.action_backend.get()
            self.request_path = save_request(self.output / 'target_request.json', request)
            self.lock_status.set(f"LOCKED {request['target']['class_name']} #{self.locked_index + 1} in {row['name']}. Saved {self.request_path}")
        except (ValueError, FileNotFoundError) as exc:
            messagebox.showerror('Cannot lock target', str(exc))

    def run_adapter(self):
        if self.request_path is None or not self.request_path.is_file():
            messagebox.showerror('No target request', 'Lock one detected instance first.')
            return
        if json.loads(self.request_path.read_text()).get('action_backend_requested') != self.action_backend.get():
            messagebox.showerror('Backend changed', 'Lock the target again after changing the action backend.')
            return
        python = Path(self.adapter_python.get()).expanduser()
        script = Path(self.adapter_script.get()).expanduser()
        if not python.is_file() or not script.is_file():
            messagebox.showerror('Adapter missing', 'Choose an existing adapter Python and script. The GUI does not contain trained OpenVLA or a closed-loop DP runner.')
            return
        self.lock_status.set(f'Running {self.action_backend.get()} script with locked target…')
        threading.Thread(target=self._run_adapter_process, args=(str(python), str(script), str(self.request_path)), daemon=True).start()

    def _run_adapter_process(self, python, script, request):
        try:
            result = subprocess.run([python, script, request], cwd=REPO, capture_output=True, text=True)
            outcome = dict(adapter=script, exit_code=result.returncode, stdout=result.stdout[-10000:], stderr=result.stderr[-10000:])
            (self.output / 'action_result.json').write_text(json.dumps(outcome, indent=2))
            status = f'Adapter finished, exit {result.returncode}. See action_result.json in report folder.'
        except OSError as exc:
            status = f'Adapter failed to start: {exc}'
        self.root.after(0, lambda: self.lock_status.set(status))


def main():
    root = tk.Tk()
    Studio(root)
    root.mainloop()


if __name__ == '__main__':
    main()
