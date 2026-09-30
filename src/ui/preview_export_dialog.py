"""Episode selection for PNG inspection; no H5 dependencies in the Tk process."""
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog


class PreviewSelectionDialog(simpledialog.Dialog):
    def __init__(self, parent, source):
        self.source = Path(source)
        self.files = sorted(self.source.rglob('episode_*.h5'))
        self.visible = []
        super().__init__(parent, title='PNG previews — choose skill and episodes')

    def body(self, master):
        master.columnconfigure(0, weight=1)
        ttk.Label(master, text='1. Skill → 2. Select episode files → 3. Choose frames', padding=6).grid(sticky='w')
        self.skill = tk.StringVar(value='Both')
        skill = ttk.Combobox(master, textvariable=self.skill, values=('Both','Pick','Place'), state='readonly')
        skill.grid(sticky='ew', pady=4)
        skill.bind('<<ComboboxSelected>>', self.refresh)
        box = ttk.Frame(master); box.grid(sticky='nsew')
        self.episodes = tk.Listbox(box, selectmode=tk.EXTENDED, exportselection=False, width=95, height=12)
        scroll = ttk.Scrollbar(box, orient='vertical', command=self.episodes.yview)
        self.episodes.configure(yscrollcommand=scroll.set)
        self.episodes.pack(side='left', fill='both', expand=True); scroll.pack(side='right', fill='y')
        ttk.Label(master, text='Ctrl/Shift-click for multiple episodes. Paths distinguish objects and sessions.').grid(sticky='w', pady=4)
        ttk.Button(master, text='Select all listed episodes', command=lambda:self.episodes.selection_set(0,tk.END)).grid(sticky='w')
        self.mode = tk.StringVar(value='Middle frame (quick check)')
        ttk.Label(master, text='Frames per selected episode:').grid(sticky='w', pady=(12,0))
        ttk.Combobox(master, textvariable=self.mode, state='readonly', width=40,
                     values=('Middle frame (quick check)','First frame','Last frame','All frames / interval')).grid(sticky='w')
        interval = ttk.Frame(master); interval.grid(sticky='w', pady=6)
        ttk.Label(interval, text='Interval for all-frame mode (1 = every frame): ').pack(side='left')
        self.stride = tk.StringVar(value='25')
        ttk.Entry(interval, textvariable=self.stride, width=8).pack(side='left')
        ttk.Label(master, text='All available cameras: RGB + colored semantics + raw ID masks + depth previews.\nNative resolution; source H5 unchanged. PNGs only, no GIF/video.', wraplength=700).grid(sticky='w', pady=8)
        self.refresh()
        return self.episodes

    def refresh(self, event=None):
        wanted={'Pick':'pick_policy','Place':'place_policy'}.get(self.skill.get())
        self.visible=[p for p in self.files if wanted is None or wanted in p.parts]
        self.episodes.delete(0,tk.END)
        for p in self.visible: self.episodes.insert(tk.END,str(p.relative_to(self.source)))
        if self.visible: self.episodes.selection_set(0)

    def validate(self):
        if not self.episodes.curselection():
            messagebox.showerror('Select episodes','Select at least one saved H5 episode.',parent=self); return False
        try:
            if int(self.stride.get()) < 1: raise ValueError()
        except ValueError:
            messagebox.showerror('Invalid interval','Use a whole number of at least 1.',parent=self); return False
        return True

    def apply(self):
        mode={'Middle frame (quick check)':'middle','First frame':'first','Last frame':'last','All frames / interval':'all'}[self.mode.get()]
        self.result=dict(files=[str(self.visible[i].relative_to(self.source)) for i in self.episodes.curselection()],
                         mode=mode, stride=int(self.stride.get()))
