"""Standalone Tk panel for static object-detection collection."""
import datetime
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import ttk,filedialog,messagebox,font as tkfont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.entry.runtime_python import isaac_python


class DetectionPanel:
    def __init__(self, win):
        self.win=win; self.process=None; self.log=None; self.run=None; self.controls=[]
        win.title('Instrument Dataset Studio'); win.geometry('1080x780'); win.minsize(900,700)
        win.configure(background='#eef2f6')
        families={name.lower():name for name in tkfont.families(win)}
        sans=next((families[name] for name in ('dejavu sans','noto sans','helvetica','nimbus sans l') if name in families),'TkDefaultFont')
        style=ttk.Style(win); style.theme_use('clam')
        style.configure('.', font=(sans,11), background='#eef2f6', foreground='#192b40')
        style.configure('TFrame', background='#eef2f6')
        style.configure('Card.TFrame', background='#ffffff')
        style.configure('TLabel', background='#ffffff', foreground='#192b40')
        style.configure('Muted.TLabel', foreground='#526479', font=(sans,10))
        style.configure('Title.TLabel', font=(sans,23,'bold'), background='#eef2f6')
        style.configure('Subtitle.TLabel', foreground='#526479', background='#eef2f6')
        style.configure('Heading.TLabel', font=(sans,12,'bold'))
        style.configure('Metric.TLabel', font=(sans,23,'bold'), foreground='#17645c')
        style.configure('TEntry', padding=7, fieldbackground='#f5f7fa', foreground='#192b40')
        style.configure('TCombobox', padding=7, fieldbackground='#f5f7fa', foreground='#192b40')
        style.map('TCombobox',fieldbackground=[('readonly','#f5f7fa')],foreground=[('readonly','#192b40')])
        style.configure('TButton', padding=(14,9), background='#e8edf3', borderwidth=0)
        style.map('TButton',background=[('active','#d6e1eb')])
        style.configure('Primary.TButton',background='#17645c',foreground='white',font=(sans,10,'bold'))
        style.map('Primary.TButton',background=[('active','#0e5049'),('disabled','#a6b8b5')],foreground=[('disabled','#f5f7fa')])
        style.configure('Stop.TButton',foreground='#a33a30')
        style.configure('TCheckbutton',background='#ffffff',padding=4)
        style.configure('TNotebook',background='#eef2f6',borderwidth=0)
        style.configure('TNotebook.Tab',padding=(18,10))
        style.map('TNotebook.Tab',background=[('selected','#ffffff')],foreground=[('selected','#17645c')])
        style.configure('Horizontal.TProgressbar',background='#238577',troughcolor='#e7edf2',borderwidth=0)
        main=ttk.Frame(win,padding=24); main.pack(fill='both',expand=True)
        header=ttk.Frame(main); header.pack(fill='x',pady=(0,18))
        ttk.Label(header,text='Instrument Dataset Studio',style='Title.TLabel').pack(anchor='w')
        ttk.Label(header,text='Object detection recorder   ·   Isaac Sim 6   ·   DLAA 448 × 448',style='Subtitle.TLabel').pack(anchor='w',pady=(6,0))
        body=ttk.Frame(main); body.pack(fill='both',expand=True)
        body.columnconfigure(0,weight=3); body.columnconfigure(1,weight=2); body.rowconfigure(0,weight=1)
        notebook=ttk.Notebook(body); notebook.grid(row=0,column=0,sticky='nsew',padx=(0,18))
        setup=ttk.Frame(notebook,style='Card.TFrame',padding=22)
        advanced=ttk.Frame(notebook,style='Card.TFrame',padding=22)
        notebook.add(setup,text='Collection'); notebook.add(advanced,text='Advanced')
        defaults=dict(scenes=100,min_objects=8,max_objects=18,ring_cameras=8,randomization='original',
                      profile='',seed=42,robot=25,tray='random',output=str(ROOT/'datasets'/'detection_runs'))
        self.values={k:tk.StringVar(value=str(v)) for k,v in defaults.items()}
        self.headless=tk.BooleanVar(value=True)
        def field(parent,key,label,row,choices=None):
            ttk.Label(parent,text=label).grid(row=row,column=0,sticky='w',pady=5,padx=(0,12))
            widget=(ttk.Combobox(parent,textvariable=self.values[key],values=choices,state='readonly',width=14)
                    if choices else ttk.Entry(parent,textvariable=self.values[key],width=16))
            widget.grid(row=row,column=1,sticky='ew',pady=5); self.controls.append(widget)
            return widget
        for page in (setup,advanced): page.columnconfigure(1,weight=1)
        ttk.Label(setup,text='Scene & camera',style='Heading.TLabel').grid(row=0,column=0,columnspan=2,sticky='w',pady=(0,10))
        field(setup,'scenes','Scenes to collect',1)
        field(setup,'randomization','Randomization',2,('original','wide'))
        field(setup,'min_objects','Minimum tools',3)
        field(setup,'max_objects','Maximum tools',4)
        field(setup,'ring_cameras','Views around table',5)
        self.summary=tk.StringVar()
        ttk.Label(setup,textvariable=self.summary,style='Muted.TLabel',wraplength=400).grid(row=6,column=0,columnspan=2,sticky='w',pady=(6,12))
        ttk.Separator(setup).grid(row=7,column=0,columnspan=2,sticky='ew')
        ttk.Label(setup,text='Save location',style='Heading.TLabel').grid(row=8,column=0,columnspan=2,sticky='w',pady=(12,6))
        output=ttk.Entry(setup,textvariable=self.values['output']); output.grid(row=9,column=0,columnspan=2,sticky='ew'); self.controls.append(output)
        ttk.Button(setup,text='Choose folder',command=self.choose).grid(row=10,column=0,sticky='w',pady=(10,0))
        ttk.Label(advanced,text='Scene variation',style='Heading.TLabel').grid(row=0,column=0,columnspan=2,sticky='w',pady=(0,10))
        field(advanced,'tray','Tray occupancy',1,('random','empty','full'))
        field(advanced,'robot','Robot visibility (%)',2)
        field(advanced,'seed','Random seed',3)
        ttk.Label(advanced,text='Custom lighting & cameras',style='Heading.TLabel').grid(row=4,column=0,columnspan=2,sticky='w',pady=(20,8))
        ttk.Label(advanced,text='Optional JSON settings for wide mode.',style='Muted.TLabel').grid(row=5,column=0,columnspan=2,sticky='w')
        field(advanced,'profile','Settings file',6)
        ttk.Button(advanced,text='Browse settings',command=self.choose_profile).grid(row=7,column=1,sticky='e')
        ttk.Checkbutton(advanced,text='Render without an Isaac viewport',variable=self.headless).grid(row=8,column=0,columnspan=2,sticky='w',pady=(24,0))
        ttk.Label(advanced,text='Changes apply to new collections. Resume uses the saved scene settings.',style='Muted.TLabel',wraplength=400).grid(row=9,column=0,columnspan=2,sticky='w',pady=12)
        right=ttk.Frame(body,style='Card.TFrame',padding=22); right.grid(row=0,column=1,sticky='nsew')
        ttk.Label(right,text='Collection progress',style='Heading.TLabel').pack(anchor='w')
        self.state=tk.StringVar(value='Ready to record')
        ttk.Label(right,textvariable=self.state,font=(sans,15,'bold')).pack(anchor='w',pady=(16,8))
        self.status=tk.StringVar(value='Choose your settings, then start a new collection.')
        ttk.Label(right,textvariable=self.status,style='Muted.TLabel',wraplength=300).pack(anchor='w')
        self.progress=ttk.Progressbar(right,mode='determinate'); self.progress.pack(fill='x',pady=(14,14))
        self.metrics={}
        for key,label in [('scenes','Scenes saved'),('images','Images'),('annotations','Bounding boxes')]:
            row=ttk.Frame(right,style='Card.TFrame'); row.pack(fill='x',pady=5)
            ttk.Label(row,text=label).pack(side='left')
            self.metrics[key]=tk.StringVar(value='0')
            ttk.Label(row,textvariable=self.metrics[key],style='Metric.TLabel').pack(side='right')
        ttk.Separator(right).pack(fill='x',pady=12)
        ttk.Label(right,text='PNG images & COCO labels',style='Muted.TLabel').pack(anchor='w')
        ttk.Button(right,text='Open collection folder',command=self.open_collection).pack(anchor='w',pady=(12,0))
        ttk.Button(right,text='Technical log',command=self.show_log).pack(anchor='w',pady=(8,0))
        self.log_window=tk.Toplevel(win); self.log_window.title('Recorder technical log'); self.log_window.geometry('900x500'); self.log_window.withdraw()
        self.log_window.protocol('WM_DELETE_WINDOW',self.log_window.withdraw)
        self.logtext=tk.Text(self.log_window,wrap='word',state='disabled',background='#f5f7fa',foreground='#192b40',font=('Courier',10),padx=16,pady=16,relief='flat')
        scroll=ttk.Scrollbar(self.log_window,command=self.logtext.yview); scroll.pack(side='right',fill='y')
        self.logtext.configure(yscrollcommand=scroll.set); self.logtext.pack(fill='both',expand=True)
        actions=ttk.Frame(main); actions.pack(side='bottom',fill='x',pady=(14,0),before=body)
        self.start_button=ttk.Button(actions,text='Start new collection',style='Primary.TButton',command=self.start); self.start_button.pack(side='left')
        self.resume_button=ttk.Button(actions,text='Resume collection',command=self.resume); self.resume_button.pack(side='left',padx=10)
        self.stop_button=ttk.Button(actions,text='Stop after current scene',style='Stop.TButton',command=self.stop,state='disabled'); self.stop_button.pack(side='right')
        for key in ('scenes','ring_cameras','randomization'): self.values[key].trace_add('write',self.update_summary)
        self.update_summary(); win.protocol('WM_DELETE_WINDOW',self.close); self.poll()

    def update_summary(self,*_):
        try:
            scenes=int(self.values['scenes'].get()); views=int(self.values['ring_cameras'].get())+1
            mode=self.values['randomization'].get()
            self.summary.set(f'{views} views per scene, including one top view.\n{scenes*views:,} planned images. '+('Wide variation; tool count is limited by table capacity.' if mode=='wide' else 'Original lighting and camera ranges.'))
        except ValueError: self.summary.set('Enter scene and camera counts to see the image estimate.')

    def choose_profile(self):
        path=filedialog.askopenfilename(title='Wide randomization settings',filetypes=[('JSON settings','*.json')])
        if path: self.values['profile'].set(path); self.values['randomization'].set('wide')

    def open_collection(self):
        path=self.run or Path(self.values['output'].get()).expanduser()
        if not path.is_dir(): messagebox.showinfo('Collection folder','The folder will be created when recording starts.'); return
        try:
            if os.name=='nt': os.startfile(str(path))
            else: subprocess.Popen(['open' if sys.platform=='darwin' else 'xdg-open',str(path)])
        except OSError as error: messagebox.showerror('Cannot open folder',str(error))

    def show_log(self):
        self.log_window.deiconify(); self.log_window.lift()

    def set_running(self,running):
        self.start_button.configure(state='disabled' if running else 'normal')
        self.resume_button.configure(state='disabled' if running else 'normal')
        self.stop_button.configure(state='normal' if running else 'disabled')

    def choose(self):
        folder=filedialog.askdirectory()
        if folder:self.values['output'].set(folder)

    def start(self):
        if self.process and self.process.poll() is None:return
        try:
            v={k:x.get() for k,x in self.values.items()}
            n,lo,hi,views,seed=[int(v[k]) for k in ('scenes','min_objects','max_objects','ring_cameras','seed')]
            probability=float(v['robot'])/100
            minimum,cap,viewcap=(1,64,32) if v['randomization']=='wide' else (5,30,16)
            if not(n>=1 and minimum<=lo<=hi<=cap and 3<=views<=viewcap and 0<=probability<=1):raise ValueError(f'Check scene count, {minimum}–{cap} objects, 3–{viewcap} views and robot 0–100%.')
            folder=Path(v['output'])/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.launch(folder,n,lo,hi,views,seed,probability,v['tray'],False,v['randomization'],v['profile'])
        except Exception as e: messagebox.showerror('Cannot start',str(e))

    def resume(self):
        if self.process and self.process.poll() is None:return
        folder=filedialog.askdirectory(title='Select collection with config.json')
        if not folder:return
        try:
            c=json.loads((Path(folder)/'config.json').read_text())
            self.launch(Path(folder),int(self.values['scenes'].get()),c['min_objects'],c['max_objects'],c['ring_cameras'],c['seed'],c['robot_visible_probability'],c['tray'],True,c.get('randomization','original'),wide=c.get('wide_profile'))
        except Exception as e:messagebox.showerror('Cannot resume',str(e))

    def launch(self,folder,n,lo,hi,views,seed,robot,tray,resume,randomization='original',profile='',wide=None):
        from detection.randomization import load_profile
        if profile and randomization!='wide': raise ValueError('Select wide mode to use a settings JSON.')
        if randomization=='wide':
            wide=wide if wide is not None else load_profile(Path(profile) if profile else None)
        python=Path(isaac_python())
        if not python.is_file():raise FileNotFoundError('Isaac Python not found: '+str(python))
        logdir=ROOT/'debug'/'logs'/'detection'; logdir.mkdir(parents=True,exist_ok=True)
        stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.stopfile=logdir/(stamp+'.stop'); self.logpath=logdir/(stamp+'.log')
        argv=[str(python),str(ROOT/'detection'/'record.py'),'--output',str(folder),'--scenes',str(n),
              '--min-objects',str(lo),'--max-objects',str(hi),'--ring-cameras',str(views),'--seed',str(seed),
              '--robot-visible-probability',str(robot),'--tray',tray,'--stop-file',str(self.stopfile)]
        argv.extend(['--randomization',randomization])
        if wide is not None:
            profilepath=logdir/(stamp+'.wide.json')
            profilepath.write_text(json.dumps(wide),encoding='utf-8')
            argv.extend(['--randomization-config',str(profilepath)])
        if resume:argv.append('--resume')
        if self.headless.get():argv.append('--headless')
        self.log=self.logpath.open('w',encoding='utf-8'); self.run=folder; self.offset=0
        self.process=subprocess.Popen(argv,cwd=ROOT,stdout=self.log,stderr=subprocess.STDOUT,
                                      creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        self.set_running(True)
        self.state.set('Starting simulator')
        self.status.set('Loading the scene. The first capture may take a moment.')
        self.progress.configure(value=0,maximum=n)
        for value in self.metrics.values(): value.set('0')

    def stop(self):
        if self.process and self.process.poll() is None:
            self.stopfile.write_text('stop',encoding='utf-8'); self.state.set('Finishing current scene')
            self.status.set('The recorder will stop after saving this scene.'); self.stop_button.configure(state='disabled')

    def poll(self):
        if self.process:
            if self.logpath.exists():
                with self.logpath.open(encoding='utf-8',errors='replace') as f:
                    f.seek(self.offset); lines=f.read(); self.offset=f.tell()
                lines=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', lines)
                relevant='\n'.join(s for s in lines.splitlines() if '[DETECTION' in s or 'Error' in s or 'Traceback' in s)
                if relevant:
                    self.logtext.configure(state='normal'); self.logtext.insert('end',relevant+'\n'); self.logtext.see('end'); self.logtext.configure(state='disabled')
            status=self.run/'status.json'
            if status.is_file():
                try:
                    s=json.loads(status.read_text())
                    for key,value in self.metrics.items(): value.set(f"{s[key]:,}")
                    self.progress.configure(maximum=max(1,s['goal']),value=s['scenes'])
                    if not self.stopfile.exists():
                        self.state.set({'complete':'Collection complete','stopped':'Collection paused'}.get(s['state'],'Recording'))
                        self.status.set(f"{s['scenes']} of {s['goal']} scenes saved.")
                except (OSError,ValueError):pass
            code=self.process.poll()
            if code is not None:
                self.log.close(); self.process=None
                self.set_running(False)
                if code:
                    self.state.set('Recording needs attention')
                    self.status.set('Capture stopped with an error. Open the technical log for details.')
                else:
                    self.state.set('Collection paused' if self.stopfile.exists() else 'Collection complete')
                    self.status.set('Saved successfully. Open the collection folder to view your dataset.')
        self.win.after(800,self.poll)

    def close(self):
        if self.process and self.process.poll() is None:
            self.stop(); return
        self.win.destroy()


if __name__=='__main__':
    app=tk.Tk(); DetectionPanel(app); app.mainloop()
