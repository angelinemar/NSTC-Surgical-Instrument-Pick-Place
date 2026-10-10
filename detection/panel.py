"""Standalone Tk panel for static object-detection collection."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import ttk,filedialog,messagebox

ROOT=Path(__file__).resolve().parents[1]


class DetectionPanel:
    def __init__(self,win):
        self.win=win; self.process=None; self.log=None; self.run=None; self.controls=[]
        win.title('P4 Object Detection Recorder'); win.geometry('920x720'); win.minsize(800,620)
        main=ttk.Frame(win,padding=20); main.pack(fill='both',expand=True)
        ttk.Label(main,text='Static Object Detection Recorder',font=('Segoe UI',20,'bold')).pack(anchor='w')
        ttk.Label(main,text='Scattered instruments • DLAA 448 × 448 • PNG + COCO • Training scenes only').pack(anchor='w',pady=(5,16))
        form=ttk.Frame(main); form.pack(fill='x'); form.columnconfigure(1,weight=1)
        self.values={}
        fields=[('scenes','Scenes to save',100,'One scene = a new arrangement. Each scene produces ring views + one top view.'),
                ('min_objects','Minimum objects on table',8,'Duplicates of all five classes are allowed.'),
                ('max_objects','Maximum objects on table',18,'Tray occupants are additional. High counts can fail the geometry fit check.'),
                ('ring_cameras','Views around the table',8,'3–16 surrounding views, plus one top view. One sensor captures sequentially.'),
                ('seed','Collection seed',42,'Use a different seed for each independent training collection.'),
                ('robot','Scenes with visible static robot (%)',25,'The robot stays at its home pose. It never performs pick/place.'),
                ('tray','Tray occupancy','random','Random keeps instruments in their fixed tray slots.'),
                ('output','Save collections under',str(ROOT/'datasets'/'detection_runs'),'Each Start creates a new collection folder.')]
        for row,(key,label,default,helptext) in enumerate(fields):
            var=tk.StringVar(value=str(default)); self.values[key]=var
            ttk.Label(form,text=label).grid(row=row*2,column=0,sticky='w',padx=(0,12))
            widget=ttk.Combobox(form,textvariable=var,values=('random','empty','full'),state='readonly') if key=='tray' else ttk.Entry(form,textvariable=var)
            widget.grid(row=row*2,column=1,sticky='ew'); self.controls.append(widget)
            ttk.Label(form,text=helptext,foreground='#53616d',wraplength=780).grid(row=row*2+1,column=0,columnspan=2,sticky='w',pady=(2,8))
        self.headless=tk.BooleanVar(value=True)
        ttk.Checkbutton(main,text='Headless rendering (no Isaac viewport)',variable=self.headless).pack(anchor='w')
        actions=ttk.Frame(main); actions.pack(fill='x',pady=12)
        for label,fn in [('Choose folder',self.choose),('Start new',self.start),('Resume collection',self.resume),('Stop after scene',self.stop)]:
            ttk.Button(actions,text=label,command=fn).pack(side='left',padx=(0,8))
        self.status=tk.StringVar(value='Ready. Existing DP recording data remains separate.')
        ttk.Label(main,textvariable=self.status,wraplength=840).pack(anchor='w')
        self.logtext=tk.Text(main,height=8,wrap='word',state='disabled'); self.logtext.pack(fill='both',expand=True,pady=(8,0))
        win.protocol('WM_DELETE_WINDOW',self.close); self.poll()

    def choose(self):
        folder=filedialog.askdirectory()
        if folder:self.values['output'].set(folder)

    def start(self):
        if self.process and self.process.poll() is None:return
        try:
            v={k:x.get() for k,x in self.values.items()}
            n,lo,hi,views,seed=[int(v[k]) for k in ('scenes','min_objects','max_objects','ring_cameras','seed')]
            probability=float(v['robot'])/100
            if not(n>=1 and 5<=lo<=hi<=30 and 3<=views<=16 and 0<=probability<=1):raise ValueError('Check scene count, 5–30 objects, 3–16 views and robot 0–100%.')
            folder=Path(v['output'])/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.launch(folder,n,lo,hi,views,seed,probability,v['tray'],False)
        except Exception as e: messagebox.showerror('Cannot start',str(e))

    def resume(self):
        if self.process and self.process.poll() is None:return
        folder=filedialog.askdirectory(title='Select collection with config.json')
        if not folder:return
        try:
            c=json.loads((Path(folder)/'config.json').read_text())
            self.launch(Path(folder),int(self.values['scenes'].get()),c['min_objects'],c['max_objects'],c['ring_cameras'],c['seed'],c['robot_visible_probability'],c['tray'],True)
        except Exception as e:messagebox.showerror('Cannot resume',str(e))

    def launch(self,folder,n,lo,hi,views,seed,robot,tray,resume):
        python=ROOT.parents[3]/'_isaac_sim'/'python.bat'
        if not python.is_file():raise FileNotFoundError('Isaac Python not found: '+str(python))
        logdir=ROOT/'debug'/'logs'/'detection'; logdir.mkdir(parents=True,exist_ok=True)
        stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.stopfile=logdir/(stamp+'.stop'); self.logpath=logdir/(stamp+'.log')
        argv=[str(python),str(ROOT/'detection'/'record.py'),'--output',str(folder),'--scenes',str(n),
              '--min-objects',str(lo),'--max-objects',str(hi),'--ring-cameras',str(views),'--seed',str(seed),
              '--robot-visible-probability',str(robot),'--tray',tray,'--stop-file',str(self.stopfile)]
        if resume:argv.append('--resume')
        if self.headless.get():argv.append('--headless')
        self.log=self.logpath.open('w',encoding='utf-8'); self.run=folder; self.offset=0
        self.process=subprocess.Popen(argv,cwd=ROOT,stdout=self.log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        self.status.set('Starting Isaac. Collection: '+str(folder))

    def stop(self):
        if self.process and self.process.poll() is None:
            self.stopfile.write_text('stop',encoding='utf-8'); self.status.set('Stop requested. Saving the current scene before stopping.')

    def poll(self):
        if self.process:
            if self.logpath.exists():
                with self.logpath.open(encoding='utf-8',errors='replace') as f:
                    f.seek(self.offset); lines=f.read(); self.offset=f.tell()
                relevant='\n'.join(s for s in lines.splitlines() if '[DETECTION' in s or 'Error' in s or 'Traceback' in s)
                if relevant:
                    self.logtext.configure(state='normal'); self.logtext.insert('end',relevant+'\n'); self.logtext.see('end'); self.logtext.configure(state='disabled')
            status=self.run/'status.json'
            if status.is_file():
                try:
                    s=json.loads(status.read_text()); self.status.set(f"{s['state']}: {s['scenes']}/{s['goal']} scenes, {s['images']} RGB images, {s['annotations']} boxes")
                except (OSError,ValueError):pass
            code=self.process.poll()
            if code is not None:
                self.log.close(); self.process=None
                self.status.set(self.status.get()+f' | Process exited {code}. Log: {self.logpath}')
        self.win.after(800,self.poll)

    def close(self):
        if self.process and self.process.poll() is None:
            self.stop(); return
        self.win.destroy()


if __name__=='__main__':
    app=tk.Tk(); DetectionPanel(app); app.mainloop()
