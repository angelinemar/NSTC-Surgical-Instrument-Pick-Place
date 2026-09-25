"""Run with ordinary Python (Tkinter), then launch Isaac using its python.bat."""
import json
import math
import os
from pathlib import Path
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
from phase4_panel_log import LogFilter,LogTail
from phase4_panel_state import Dashboard,rotate_outline
from phase4_panel_widgets import Tooltip,cell_style
from src.recorder.capture_contract import LABELS, capture_contract, storage_budget

ROOT = Path(__file__).resolve().parent
NAMES = ('scalpel','scissor','love_retractor','kelly','scalpel_type2')
# Display/early validation only; backend validates exact asset envelopes again.
RADII = dict(zip(NAMES,(.081124,.082338,.075123,.066302,.075110)))
COLORS = ('#ffd166','#ef476f','#06d6a0','#45aaf2','#b794f4')


def default_positions(layout):
    x0,x1 = layout['grid_x']; y0,y1 = layout['grid_y']
    return {name:dict(x=x1-.10,y=y0+.10+i*.20,yaw_deg=0.) for i,name in enumerate(NAMES)}


class Panel:
    def __init__(self, root):
        self.root = root
        self.layout = json.loads((ROOT/'env'/'scene_layout.json').read_text())
        self.positions = default_positions(self.layout)
        self.process = None
        self.log_file = None
        self.session = None
        self.session_path = None
        self.output_dir = None
        self.export_process = None
        self.export_log = None
        self.output_text = tk.StringVar(value='No run selected - Launch a run or browse an existing run folder.')
        self._poll_id=None
        self.dashboard=Dashboard(); self.live_geometry=None
        preview_path=ROOT/'env'/'instrument_preview_geometry.json'
        self.preview_geometry=json.loads(preview_path.read_text(encoding='utf-8'))['instruments'] if preview_path.exists() else {}
        root.title('P4 Recorder - prepare, record, retry')
        root.geometry('1300x900'); root.minsize(1100,780)
        root.configure(bg='#151b24')
        style=ttk.Style(root); style.theme_use('clam')
        style.configure('.',background='#151b24',foreground='#e5edf5',fieldbackground='#253140',bordercolor='#39485b')
        style.configure('TButton',padding=7,background='#263b50')
        style.map('TButton',background=[('active','#365779')])
        style.map('TCombobox',fieldbackground=[('readonly','#253140'),('disabled','#1b222e')],foreground=[('readonly','#e5edf5'),('disabled','#8695a7')])
        root.option_add('*TCombobox*Listbox.background','#253140')
        root.option_add('*TCombobox*Listbox.foreground','#e5edf5')
        self.target = tk.StringVar(value='scalpel')
        self.collection = tk.StringVar(value='single')
        self.count_caption=tk.StringVar(value='Successful episodes')
        self.coverage=None
        self.live_positions=None; self.live_tray=None
        self.log_filter=LogFilter(); self.log_tail=LogTail()
        self.mode = tk.StringVar(value='manual')
        self.skill = tk.StringVar(value='both')
        self.dataset_purpose = tk.StringVar(value='Both: DP + detector (recommended)')
        self.camera_size = tk.StringVar(value='224')
        self.dataset_split = tk.StringVar(value='auto')
        self.session_seed = tk.IntVar(value=int(datetime.now().timestamp()))
        self.destination = tk.StringVar(value=str(ROOT/'datasets'/'panel_runs'))
        self.episodes = tk.IntVar(value=1)
        self.tray_mode = tk.StringVar(value='random')
        self.distractor_min = tk.IntVar(value=12)
        self.distractor_max = tk.IntVar(value=18)
        self.tray_counts = {n:tk.IntVar(value=0) for n in NAMES}
        self.status = tk.StringVar(value='Choose target, drag instrument footprints, then Launch / Prepare.')
        self.selected = tk.StringVar(value='scalpel')
        self.x = tk.DoubleVar(); self.y = tk.DoubleVar(); self.yaw = tk.DoubleVar()
        self.counter_text=tk.StringVar(); self.active_stage=tk.StringVar()
        header=ttk.Frame(root,padding=(12,8)); header.pack(fill='x')
        ttk.Label(header,text='P4 RECORDER',font=('Segoe UI',14,'bold')).pack(side='left')
        counters=ttk.Frame(header); counters.pack(side='right')
        ttk.Label(counters,textvariable=self.counter_text,font=('Consolas',11,'bold'),justify='right').pack(anchor='e')
        ttk.Label(counters,textvariable=self.active_stage,wraplength=600,justify='right').pack(anchor='e')
        footer=ttk.Frame(root,padding=(12,6)); footer.pack(side='bottom',fill='x')
        self.notebook=ttk.Notebook(root); self.notebook.pack(fill='both',expand=True,padx=12,pady=5)
        style.configure('TNotebook.Tab',padding=(16,8))
        style.map('TNotebook.Tab',background=[('selected','#28506b')],foreground=[('selected','#ffffff')])
        self.pages={}
        for key,title in (('workspace','1. Record & live log'),('tray','2. Dataset & tray'),('results','3. Files & export')):
            page=ttk.Frame(self.notebook,padding=8); self.pages[key]=page; self.notebook.add(page,text=title)
        main=self.pages['workspace']; traypage=self.pages['tray']; results=self.pages['results']
        self.tooltips=[]; self.action_buttons={}
        settings = ttk.LabelFrame(main,text='Recording setup',padding=8); settings.pack(fill='x')
        self.controls = []
        hints=('Single: requested successful episodes. Grid cycles: one success in every cell per round.',
               'Instrument to pick. Its assigned tray slot must initially be empty.',
               'Manual: positions from this map. Auto: randomized spawn. Grid cycles always uses auto.',
               'Choose which H5 segments to save. This does not change the physical success checks.')
        for index,(label,var,values) in enumerate((('Collection',self.collection,('single','grid_cycles')),('Target',self.target,NAMES),('Spawn',self.mode,('manual','auto')),('Save skill',self.skill,('both','pick','place')))):
            row,col=divmod(index,2); col*=2
            ttk.Label(settings,text=label).grid(row=row,column=col,sticky='w',padx=5,pady=3)
            widget = ttk.Combobox(settings,textvariable=var,values=values,state='readonly',width=16)
            widget.grid(row=row,column=col+1,sticky='ew',padx=5,pady=3); self.controls.append(widget)
            self.tooltips.append(Tooltip(widget,hints[index]))
        settings.columnconfigure(1,weight=1); settings.columnconfigure(3,weight=1)
        count = ttk.Frame(settings); count.grid(row=2,column=0,columnspan=4,sticky='ew',pady=4)
        for label,var in (('Successful episodes',self.episodes),):
            ttk.Label(count,textvariable=self.count_caption).pack(side='left',padx=5)
            widget = ttk.Spinbox(count,from_=1,to=100000,textvariable=var,width=8)
            widget.pack(side='left'); self.controls.append(widget)
        self.coverage_text=tk.StringVar(value='Single mode: saved successes, no per-cell quota.')
        ttk.Label(main,textvariable=self.coverage_text,padding=(3,6),wraplength=1050).pack(fill='x')
        dataset_summary=ttk.Frame(main); dataset_summary.pack(fill='x')
        ttk.Label(dataset_summary,textvariable=self.dataset_purpose).pack(side='left',padx=4)
        ttk.Button(dataset_summary,text='Dataset / camera / save folder...',command=lambda:self.notebook.select(traypage)).pack(side='right')
        dataset = ttk.LabelFrame(traypage,text='Dataset output - independent of Pick / Place / Both',padding=8)
        dataset.pack(fill='x',pady=4)
        for row,(label,var,values) in enumerate((
            ('Training outputs to make later',self.dataset_purpose,tuple(LABELS)),
            ('Recorded image size',self.camera_size,('448','224')))):
            ttk.Label(dataset,text=label).grid(row=row,column=0,sticky='w',padx=4)
            widget=ttk.Combobox(dataset,textvariable=var,values=values,state='readonly',width=36)
            widget.grid(row=row,column=1,sticky='ew',pady=2); self.controls.append(widget)
        self.split_preview=tk.StringVar(value='Split: auto 70/20/10   Seed: auto')
        ttk.Label(dataset,textvariable=self.split_preview).grid(row=2,column=0,columnspan=3,sticky='w',padx=4,pady=2)
        ttk.Label(dataset,text='Save raw to').grid(row=3,column=0,sticky='w',padx=4)
        widget=ttk.Entry(dataset,textvariable=self.destination,width=55)
        widget.grid(row=3,column=1,sticky='ew'); self.controls.append(widget)
        self.help_button(dataset,'Choose folder',self.choose_destination,'Choose storage before launch. Existing recordings are never deleted.').grid(row=3,column=2,padx=4)
        self.dataset_hint=tk.StringVar()
        ttk.Label(dataset,textvariable=self.dataset_hint,wraplength=760,justify='left').grid(row=4,column=0,columnspan=2,sticky='w',pady=5)
        dataset.columnconfigure(1,weight=1)
        self.dataset_purpose.trace_add('write',self.dataset_changed)
        self.dataset_changed()
        tray_controls=ttk.LabelFrame(traypage,text='Base tray layout + extra distractors',padding=12)
        tray_controls.pack(fill='x',pady=6)
        ttk.Label(tray_controls,text='Mode').grid(row=0,column=0,sticky='w',padx=8)
        traymode=ttk.Combobox(tray_controls,textvariable=self.tray_mode,values=('random','empty','full','manual'),state='readonly',width=16)
        traymode.grid(row=0,column=1,sticky='w',pady=6); self.controls.append(traymode)
        self.tooltips.append(Tooltip(traymode,'Random: non-target classes split between table and tray, with at least one table distractor. Full: all four non-target classes in tray. Manual: use counts below. Target is never preloaded.'))
        for i,n in enumerate(NAMES,1):
            ttk.Label(tray_controls,text=n).grid(row=i,column=0,sticky='w',padx=8,pady=4)
            widget=ttk.Spinbox(tray_controls,from_=0,to=1,textvariable=self.tray_counts[n],width=5)
            widget.grid(row=i,column=1,sticky='w'); self.controls.append(widget)
        ttk.Label(tray_controls,text='Base objects: 0 = table, 1 = tray (manual mode).\nExtra distractors repeat non-target types on table/tray.\nTarget type appears exactly once; its tray lane stays free.',justify='left').grid(row=1,column=2,rowspan=3,sticky='nw',padx=20)
        clutter = ttk.Frame(tray_controls)
        clutter.grid(row=4,column=2,rowspan=2,sticky='w',padx=20)
        ttk.Label(clutter,text='Total distractors, random range:').pack(side='left')
        for var in (self.distractor_min, self.distractor_max):
            spin = ttk.Spinbox(clutter,from_=4,to=30,textvariable=var,width=4)
            spin.pack(side='left',padx=5); self.controls.append(spin)
            self.tooltips.append(Tooltip(spin,'Total distractors excludes the single target. Default 12-18. Includes the four base non-targets plus randomly placed duplicate instances. Restart a running simulator to change this range. Dense scenes that cannot fit safely are rejected.'))
        body=ttk.Panedwindow(main,orient='horizontal'); body.pack(fill='both',expand=True)
        workspace=ttk.Frame(body); logbox=ttk.LabelFrame(body,text='Run log - saves, failures and stage timing',padding=5)
        body.add(workspace,weight=3); body.add(logbox,weight=1)
        self.canvas = tk.Canvas(workspace,width=850,height=415,bg='#122128',highlightthickness=0)
        self.canvas.pack(fill='both',expand=True)
        self.log_view=tk.Text(logbox,width=38,height=15,bg='#10161e',fg='#dae5f2',wrap='word',state='disabled',font=('Consolas',10),borderwidth=0)
        scroll=ttk.Scrollbar(logbox,command=self.log_view.yview)
        self.log_view.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right',fill='y'); self.log_view.pack(fill='both',expand=True)
        for tag,color in (('info','#adc3d9'),('success','#72dfa6'),('error','#ff8787')): self.log_view.tag_configure(tag,foreground=color)
        self.canvas.bind('<Configure>',lambda e:self.redraw())
        self.canvas.bind('<Button-1>',self.click)
        self.canvas.bind('<B1-Motion>',self.drag)
        self.tray_preview=tk.Canvas(traypage,width=1000,height=110,bg='#23333e',highlightthickness=0)
        self.tray_preview.pack(fill='x',pady=8)
        self.tray_preview.bind('<Configure>',lambda e:self.redraw())
        for var in (self.target,self.tray_mode,*self.tray_counts.values()):
            var.trace_add('write',lambda *a:self.redraw())
        self.collection.trace_add('write',self.collection_changed)
        edit = ttk.LabelFrame(main,text='Manual placement editor - X/Y in meters, yaw in degrees',padding=8); edit.pack(fill='x',pady=(6,0))
        choose = ttk.Combobox(edit,textvariable=self.selected,values=NAMES,state='readonly',width=18)
        choose.pack(side='left'); choose.bind('<<ComboboxSelected>>',lambda e:self.select(self.selected.get()))
        self.help_button(edit,'Use as target',self.use_selected_target,'Select this instrument as the target to pick. Does not move it.').pack(side='left',padx=4)
        for label,var in (('X',self.x),('Y',self.y),('Yaw °',self.yaw)):
            ttk.Label(edit,text=label).pack(side='left',padx=4)
            ttk.Entry(edit,textvariable=var,width=10).pack(side='left')
        self.help_button(edit,'Apply coordinates',self.apply_edit,'Apply X/Y/yaw to the selected manual instrument. Grid cycles ignores manual target positions.').pack(side='left',padx=8)
        buttons = ttk.LabelFrame(footer,text='Run controls - hover over a button for details',padding=6); buttons.pack(fill='x')
        for label,command,hint in (
            ('Launch + record',self.prepare,'Launch Isaac, settle the scene, then automatically record. Failed attempts retry until the saved goal is reached or you press Stop.'),
            ('Prepare preview',lambda:self.prepare(False),'Launch and settle only. Inspect the scene, then click Start prepared. No recording starts until that click.'),
            ('Start prepared',lambda:self.command('start_id'),'Start recording a READY prepared scene. Subsequent retries start automatically.'),
            ('Discard attempt',lambda:self.command('discard_id'),'Discard the active attempt at the next control check. It does not count as saved. Retry the same coverage cell.'),
            ('Stop run',self.stop,'Request a safe stop at the next recorder check. Previously saved episodes are retained.')):
            button=self.help_button(buttons,label,command,hint); button.pack(side='left',padx=4); self.action_buttons[label]=button
        ttk.Label(traypage,text='GRID CYCLES\n1 round = one saved success in each cell. 2 rounds = two per cell.\nCells run in order 0-9; offset, yaw and distractors randomize. A failure never advances the cell.\n\nMAP LEGEND\nBlue CURRENT = active cell. Green DONE = full quota. PARTIAL = some rounds completed.\nMarkers exist only in this panel, not in recorded RGB/semantic images.\n\nREACHABILITY\nCoverage is a requested schedule, not proof every pose is reachable. Near-base cells can fail.\nThe yellow circle is hidden; existing safety validation is unchanged.',justify='left',padding=8).pack(anchor='w')
        ttk.Label(results,text='Files from the active or selected run',font=('Segoe UI',13,'bold'),padding=8).pack(anchor='w')
        ttk.Label(results,text='Saved skill selects pick_policy / place_policy. Dataset purpose selects downstream consumers.\nBoth consumers share raw RGB once; exporters create separate training artifacts. No model is trained by Record.\nFailure previews are diagnostics, not training samples. capture_contract.json records your choices.',padding=8).pack(anchor='w')
        output_bar=ttk.LabelFrame(results,text='Open in File Explorer',padding=12); output_bar.pack(fill='x',pady=8)
        for label,kind in (('Run folder','run'),('Pick H5','pick'),('Place H5','place'),('GIF / previews','gif'),('Failure previews','failure')):
            self.help_button(output_bar,label,lambda k=kind:self.open_output(k),'Open '+label+' for the active/selected run. Missing outputs are explained; files are never fabricated.').pack(side='left',padx=3)
        self.help_button(results,'Browse saved run...',self.choose_output,'Select an older run/object folder to inspect its H5, GIF and failure previews. Available when no run is active.').pack(anchor='w',pady=8)
        self.help_button(results,'Export collection for training...',self.export_collection,'Select the parent folder containing completed train / valid / test sessions. Exports selected dataset purpose without recording again.').pack(anchor='w',pady=4)
        ttk.Label(results,text='Selected output path (copyable):',padding=6).pack(anchor='w')
        ttk.Entry(results,textvariable=self.output_text,state='readonly').pack(fill='x',pady=4)
        ttk.Separator(footer,orient='horizontal').pack(fill='x',pady=5)
        ttk.Label(footer,textvariable=self.status,wraplength=1000,padding=4).pack(fill='x')
        root.protocol('WM_DELETE_WINDOW',self.close)
        self.lamps=tk.Canvas(footer,width=430,height=52,bg='#151b24',highlightthickness=0)
        self.lamps.pack(side='bottom',anchor='e',padx=12)
        self.episodes.trace_add('write',self.count_changed)
        self.update_dashboard()
        self.select('scalpel'); self.redraw(); self.poll()

    def help_button(self,parent,label,command,hint):
        button=ttk.Button(parent,text=label,command=command)
        self.tooltips.append(Tooltip(button,hint))
        return button

    def count_changed(self,*args):
        if not self.process:
            try:
                goal=max(1,self.episodes.get())*(self.layout['grid_rows']*self.layout['grid_cols'] if self.collection.get()=='grid_cycles' else 1)
            except tk.TclError:
                return
            self.dashboard=Dashboard(goal)
            self.coverage=None
        self.update_dashboard()

    def update_dashboard(self):
        d=self.dashboard
        if not self.process and self.session_path is None:
            try: d.goal=max(1,self.episodes.get())*(self.layout['grid_rows']*self.layout['grid_cols'] if self.collection.get()=='grid_cycles' else 1)
            except tk.TclError: pass
        self.counter_text.set(f'Episode target {d.episode}/{d.goal}  |  Saved {d.saved}/{d.goal}\nAttempt {d.attempt}  |  Fail {d.failures}')
        self.active_stage.set(d.stage)
        if hasattr(self,'canvas'): self.redraw()
        if not hasattr(self,'lamps'): return
        self.lamps.delete('all')
        for i,(phase,label) in enumerate((('reset','RESET / WAIT'),('recording','RECORDING'),('end','END / SAVING'))):
            x=65+i*145; active=d.phase==phase
            self.lamps.create_oval(x-8,5,x+8,21,fill='#ff424f' if active else '#49242c',outline='#ff8991' if active else '#66323c')
            self.lamps.create_text(x,36,text=label,fill='#f1e4e6' if active else '#81909f',font=('Segoe UI',9,'bold'))

    def draw_instrument(self,name,color,pose,actual=None):
        data=actual
        if data is None and name in self.preview_geometry: data=rotate_outline(self.preview_geometry[name],pose)
        if data is None: return
        flat=[]
        for x,y in data['hull_xy']: flat.extend(self.xy_to_canvas(x,y))
        self.canvas.create_polygon(*flat,fill='',outline=color,width=2)
        x,y=data['center_xy']; ax,ay=data['axis_xy']; norm=math.hypot(ax,ay) or 1
        half=RADII[name]*.65
        self.canvas.create_line(*self.xy_to_canvas(x-ax/norm*half,y-ay/norm*half),*self.xy_to_canvas(x+ax/norm*half,y+ay/norm*half),fill=color,width=2,arrow='last')

    def collection_changed(self,*args):
        grid=self.collection.get()=='grid_cycles'
        self.count_caption.set(f"Grid rounds ({self.layout['grid_rows']*self.layout['grid_cols']} successes / round)" if grid else 'Successful episodes')
        if grid: self.mode.set('auto')
        self.coverage=None; self.live_positions=None; self.live_tray=None; self.live_geometry=None; self.redraw(); self.count_changed()

    def use_selected_target(self):
        if self.process: self.add_log('error','Stop the current run before changing target.'); return
        self.target.set(self.selected.get())

    def add_log(self,tag,text):
        self.log_view.configure(state='normal')
        self.log_view.insert('end',datetime.now().strftime('%H:%M:%S')+' '+text+'\n',tag)
        if int(self.log_view.index('end-1c').split('.')[0])>400: self.log_view.delete('1.0','80.0')
        self.log_view.see('end'); self.log_view.configure(state='disabled')

    def pump_log(self):
        if not self.session_path: return
        path=Path((self.session or {}).get('console_log',self.session_path.parent/'console.log'))
        for line in self.log_tail.read(path):
            self.dashboard.feed(line)
            event=self.log_filter.feed(line)
            if event: self.add_log(*event)
        self.update_dashboard()

    def edits_allowed(self):
        if self.process and self.process.poll() is None:
            if self.read_status().get('state') not in ('waiting_prepare','ready_to_record'):
                self.add_log('error','Recording is active. Stop first to change positions; pending edits cannot affect the current attempt.')
                return False
        return True

    def xy_to_canvas(self,x,y):
        scale,ox,oy=self.view_transform()
        return ox+(y+1.12)*scale,oy+(x+.18)*scale

    def canvas_to_xy(self,u,v):
        scale,ox,oy=self.view_transform()
        return (v-oy)/scale-.18,(u-ox)/scale-1.12

    def view_transform(self):
        w=max(100,self.canvas.winfo_width()); h=max(100,self.canvas.winfo_height())
        scale=min((w-50)/1.6,(h-45)/.9)
        return scale,(w-1.6*scale)/2,(h-.9*scale)/2

    def redraw(self):
        c=self.canvas; c.delete('all')
        grid_mode=self.collection.get()=='grid_cycles'
        coverage=self.coverage
        if grid_mode:
            from phase4_coverage import progress
            try: cycles=max(1,int(self.session['cycles'] if self.process and self.session else self.episodes.get()))
            except (tk.TclError,ValueError): cycles=1
            if self.session and self.session.get('collection')=='grid_cycles' and self.session_path and self.dashboard.stage!='Idle':
                cycles=int(self.session['cycles'])
                saved=max(self.dashboard.saved,int((coverage or {}).get('saved',0)))
                coverage=progress(saved,cycles,self.layout['grid_rows'],self.layout['grid_cols'])
            elif coverage is None:
                coverage=progress(0,cycles,self.layout['grid_rows'],self.layout['grid_cols'])
            current_label='Complete' if coverage['complete'] else f"Cell {coverage['cell_id']}"
            self.coverage_text.set(f"{coverage['cycles']} rounds x 10 cells = {coverage['goal']} saved goal | {current_label} | Blue: current, green: done, teal: partial. Failure retries the SAME cell.")
        else:
            self.coverage_text.set('Single mode: requested saved successes. Drag instruments or edit X/Y/yaw below. Hover buttons for help.')
        x0,x1=self.layout['grid_x']; y0,y1=self.layout['grid_y']
        a=self.xy_to_canvas(x0,y0); b=self.xy_to_canvas(x1,y1)
        c.create_rectangle(*a,*b,fill='#165b47',outline='#8ce0bc',width=2)
        for i in range(self.layout['grid_rows']+1):
            y=y0+i*(y1-y0)/self.layout['grid_rows']
            c.create_line(*self.xy_to_canvas(x0,y),*self.xy_to_canvas(x1,y),fill='#347663')
        for i in range(self.layout['grid_cols']+1):
            x=x0+i*(x1-x0)/self.layout['grid_cols']
            c.create_line(*self.xy_to_canvas(x,y0),*self.xy_to_canvas(x,y1),fill='#347663')
        nx,ny=self.layout['grid_cols'],self.layout['grid_rows']
        for cell in range(nx*ny):
            row,col=divmod(cell,nx); x=x0+(col+.5)*(x1-x0)/nx; y=y0+(row+.5)*(y1-y0)/ny
            count=coverage['counts'][cell] if coverage else 0
            current=bool(grid_mode and coverage and coverage['cell_id']==cell)
            if grid_mode:
                fill,edge,state=cell_style(count,coverage['cycles'],current)
                a=self.xy_to_canvas(x0+col*(x1-x0)/nx,y0+row*(y1-y0)/ny)
                b=self.xy_to_canvas(x0+(col+1)*(x1-x0)/nx,y0+(row+1)*(y1-y0)/ny)
                c.create_rectangle(*a,*b,fill=fill,outline=edge,width=3 if current else 1,tags=('coverage_cell',f'cell_{cell}',state))
                c.create_text((a[0]+b[0])/2,min(a[1],b[1])+5,anchor='n',width=max(30,abs(b[0]-a[0])-6),text=f"{cell} {state}\n{count}/{coverage['cycles']} saved",fill=edge,font=('Segoe UI',9,'bold'),tags=('coverage_label',))
            else:
                c.create_text(*self.xy_to_canvas(x,y),text=str(cell),fill='#8baaa2',font=('Segoe UI',11,'bold'))
        rx,ry,_=self.layout['robot_pos']; u,v=self.xy_to_canvas(rx,ry)
        scale,_,_=self.view_transform()
        c.create_oval(u-26,v-22,u+26,v+22,fill='#dde4e8'); c.create_text(u,v,text='Robot')
        tx,ty=self.layout['tray_xy']
        theta=math.radians(self.layout['tray_yaw_deg']); ct,st=math.cos(theta),math.sin(theta)
        length,width=self.layout['tray_dimensions_local_xy']
        polygon=[]
        for a,b in ((-length/2,-width/2),(length/2,-width/2),(length/2,width/2),(-length/2,width/2)):
            polygon.extend(self.xy_to_canvas(tx+ct*a-st*b,ty+st*a+ct*b))
        c.create_polygon(*polygon,fill='#6b7880',outline='#c4d0d4')
        selected_tray=set()
        if self.tray_mode.get()=='full': selected_tray=set(NAMES)-{self.target.get()}
        elif self.tray_mode.get()=='manual':
            try: selected_tray={n for n in NAMES if self.tray_counts[n].get()==1}
            except tk.TclError: pass
        if self.process and self.live_tray is not None: selected_tray=set(self.live_tray)
        for i,(name,color) in enumerate(zip(NAMES,COLORS)):
            b=(i-2)*(width-.030)/5
            xy=(tx-st*b,ty+ct*b)
            u,v=self.xy_to_canvas(*xy)
            c.create_text(u,v,text=f'{i}',fill=color)
            if name in selected_tray:
                offset=self.preview_geometry.get(name,{}).get('heading_offset_deg',0.)
                self.draw_instrument(name,color,dict(x=xy[0],y=xy[1],yaw_deg=self.layout['tray_yaw_deg']-offset),
                                     self.live_geometry.get(name) if self.process and self.live_geometry else None)
        for name,color in zip(NAMES,COLORS):
            if name in selected_tray: continue
            positions=self.live_positions if self.process and self.live_positions else self.positions
            p=positions[name]; u,v=self.xy_to_canvas(p['x'],p['y']); radius=RADII[name]
            ru=rv=radius*scale
            c.create_oval(u-ru,v-rv,u+ru,v+rv,outline=color,width=3 if name==self.target.get() else 1,dash=(3,3))
            self.draw_instrument(name,color,p,self.live_geometry.get(name) if self.process and self.live_geometry else None)
            c.create_text(u,v+rv+11,text=name,fill=color)
        if self.process and self.live_geometry:
            for key, geometry in self.live_geometry.items():
                if not key.startswith('clutter_'):
                    continue
                color=COLORS[NAMES.index(geometry['class_name'])]
                polygon=[v for xy in geometry['hull_xy'] for v in self.xy_to_canvas(*xy)]
                c.create_polygon(*polygon,outline=color,fill='',width=2,tags=('extra_distractor',key))
            caption=f"After settle: 1 target + {len(self.live_geometry)-1} distractors (snapshot)"
        else:
            caption='Base-object preview; randomized duplicates appear after Prepare / settle'
        c.create_text(self.canvas.winfo_width()/2,self.canvas.winfo_height()-10,text=caption,fill='#cbd5e0')
        p=self.tray_preview; p.delete('all')
        p.create_text(12,12,anchor='w',fill='white',text='Canonical tray top view: right = tray +Y; down = tray +X. Fixed slots, NOT pick order.')
        for i,(name,color) in enumerate(zip(NAMES,COLORS)):
            cell=(max(100,p.winfo_width())-24)/5
            x=12+i*cell; middle=x+(cell-8)/2
            p.create_rectangle(x,26,x+cell-8,104,outline=color,width=2)
            target=name==self.target.get()
            mode=self.tray_mode.get()
            state='TARGET / EMPTY' if target else ('random table/tray' if mode=='random' else '1 in tray' if name in selected_tray else '0 in tray')
            p.create_text(middle,39,fill=color,text=f'{i}: {name}')
            p.create_line(middle,51,middle,80,fill=color,width=3,arrow='last')
            p.create_text(middle,94,fill='white',text=state)

    def select(self,name):
        self.selected.set(name); p=self.positions[name]
        self.x.set(round(p['x'],4)); self.y.set(round(p['y'],4)); self.yaw.set(round(p['yaw_deg'],1))

    def click(self,event):
        x,y=self.canvas_to_xy(event.x,event.y)
        names=list(NAMES)
        if self.tray_mode.get()=='full': names=[self.target.get()]
        elif self.tray_mode.get()=='manual':
            names=[n for n in NAMES if self.tray_counts[n].get()==0]
        if not names: return
        name=min(names,key=lambda n:(self.positions[n]['x']-x)**2+(self.positions[n]['y']-y)**2)
        self.select(name)

    def drag(self,event):
        if not self.edits_allowed(): return
        x,y=self.canvas_to_xy(event.x,event.y)
        self.positions[self.selected.get()].update(x=x,y=y)
        self.select(self.selected.get()); self.redraw()

    def apply_edit(self):
        if not self.edits_allowed(): return
        try:
            self.positions[self.selected.get()]=dict(x=self.x.get(),y=self.y.get(),yaw_deg=self.yaw.get())
            self.redraw()
        except tk.TclError:
            messagebox.showerror('Coordinates','Enter finite numeric coordinates and yaw.')

    def write(self):
        tmp=self.session_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.session,indent=2),encoding='utf-8'); os.replace(tmp,self.session_path)

    def choose_destination(self):
        if self.process and self.process.poll() is None: return
        folder=filedialog.askdirectory(title='Destination for new recordings (existing data stays untouched)')
        if folder: self.destination.set(folder)

    def export_collection(self):
        if self.process or self.export_process:
            messagebox.showinfo('Busy','Finish the active recording or export first.'); return
        source=filedialog.askdirectory(title='Collection root containing completed train / valid / test sessions')
        if not source: return
        parent=filedialog.askdirectory(title='Export destination (outside the raw collection)')
        if not parent: return
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output=Path(parent)/('training_export_'+stamp)
        purpose=LABELS[self.dataset_purpose.get()]
        log=ROOT/'debug'/'logs'/('export_'+stamp+'.log'); log.parent.mkdir(parents=True,exist_ok=True)
        self.export_log=log.open('w',encoding='utf-8')
        launcher=ROOT.parents[3]/'_isaac_sim'/'python.bat'
        try:
            self.export_process=subprocess.Popen([str(launcher),str(ROOT/'training/export_recordings.py'),
                '--source',source,'--output',str(output),'--purpose',purpose],cwd=ROOT,
                stdout=self.export_log,stderr=subprocess.STDOUT)
        except Exception:
            self.export_log.close(); self.export_log=None
            raise
        self.status.set(f'Exporting {purpose}. Output: {output}. Log: {log}')
        self.root.after(500,lambda:self.poll_export(output,log))

    def poll_export(self,output,log):
        if self.export_process.poll() is None:
            self.root.after(500,lambda:self.poll_export(output,log)); return
        code=self.export_process.returncode
        self.export_log.close(); self.export_log=None; self.export_process=None
        valid=code==0 and (output/'export_complete.json').is_file()
        self.status.set(f"Export {'complete (not accuracy-certified)' if valid else 'FAILED'}: {output}. Log: {log}")
        self.add_log('success' if valid else 'error',self.status.get())

    def dataset_changed(self,*args):
        purpose=LABELS[self.dataset_purpose.get()]
        if purpose == 'both':
            text='Complete mode: records DP data and standalone detector data from one raw run.'
        elif purpose == 'dp':
            text='DP joint export: RGB, robot state and actions, with semantic training labels.'
        else:
            text='Detector export: RGB, semantic and separate instance labels for repeated instruments.'
        self.dataset_hint.set(text+' Distractor range below applies to every mode. Saved RGB: 224 default, 448 for finer detail.')

    def choose_auto_split(self,destination):
        counts={s:0 for s in ('train','valid','test')}
        for path in Path(destination).resolve().rglob('capture_contract.json'):
            try:
                split=json.loads(path.read_text(encoding='utf-8')).get('split')
            except (OSError,ValueError):
                continue
            if split in counts:
                counts[split]+=1
        total=sum(counts.values())+1
        desired=dict(train=max(1,round(total*.7)),valid=round(total*.2),test=0)
        desired['test']=max(0,total-desired['train']-desired['valid'])
        if total>=3 and desired['test']==0:
            desired['test']=1; desired['train']=max(1,desired['train']-1)
        if total>=5 and desired['valid']==0:
            desired['valid']=1; desired['train']=max(1,desired['train']-1)
        for split in ('train','valid','test'):
            if counts[split]<desired[split]:
                return split
        return min(counts,key=lambda s:counts[s]/(.7 if s=='train' else .2 if s=='valid' else .1))

    def prepare(self,start=True):
        try:
            if self.export_process:
                raise ValueError('Finish the active export before launching another recorder.')
            purpose=LABELS[self.dataset_purpose.get()]
            tray_objects=[n for n in NAMES if self.tray_counts[n].get()==1]
            if any(self.tray_counts[n].get() not in (0,1) for n in NAMES):
                raise ValueError('Base-object tray choices must be 0 or 1. Extra duplicates use the distractor range.')
            if not 4 <= self.distractor_min.get() <= self.distractor_max.get() <= 30:
                raise ValueError('Distractor range must satisfy 4 <= minimum <= maximum <= 30.')
            if self.tray_mode.get()=='manual' and self.target.get() in tray_objects:
                raise ValueError('Target must not already exist in the tray. Set its tray count to 0.')
            from phase4_session import validate_positions,validate_target_workspace
            if self.mode.get()=='manual' and self.collection.get()!='grid_cycles':
                validate_positions(self.positions,self.layout,RADII)
                validate_target_workspace(self.target.get(),self.positions,self.layout)
            if self.process and self.process.poll() is None:
                if (self.distractor_min.get(),self.distractor_max.get()) != (self.session.get('distractor_min',12),self.session.get('distractor_max',18)):
                    raise ValueError('Stop and relaunch to change the number of spawned distractor bodies.')
                state=self.read_status().get('state')
                if state not in ('waiting_prepare','ready_to_record'):
                    raise ValueError('Wait for READY or waiting for the next attempt before preparing positions.')
                self.session.update(positions=json.loads(json.dumps(self.positions)),tray_mode=self.tray_mode.get(),tray_objects=tray_objects,auto_start=start)
                self.session['prepare_id']+=1; self.write(); return
            episodes=self.episodes.get()
            if episodes<1: raise ValueError('Successful episodes must be at least 1.')
            cycles=episodes
            if self.collection.get()=='grid_cycles':
                episodes*=self.layout['grid_rows']*self.layout['grid_cols']
                self.mode.set('auto')
            stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            auto_split=self.dataset_split.get()=='auto'
            seed=int(datetime.now().timestamp()*1000)%2_147_483_647 if auto_split else self.session_seed.get()
            split=self.choose_auto_split(self.destination.get()) if auto_split else self.dataset_split.get()
            self.session_seed.set(seed)
            self.split_preview.set(f'Split: {split} (auto 70/20/10)   Seed: {seed} (auto)')
            contract=capture_contract(purpose,self.skill.get(),int(self.camera_size.get()),seed,split)
            output=Path(self.destination.get()).resolve()/split/stamp/self.target.get()
            budget=storage_budget(output,episodes,int(self.camera_size.get()))
            if not budget['capacity_pass']:
                raise ValueError(f"Not enough storage: estimate {budget['estimated_bytes']/1e9:.1f} GB + 30 GB reserve, free {budget['free_bytes']/1e9:.1f} GB. Choose another folder, fewer episodes or native 224.")
            output.mkdir(parents=True)
            (output/'capture_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
            self.session_path=output/'session.json'
            self.output_dir=output; self.output_text.set(str(output))
            self.session=dict(target=self.target.get(),mode=self.mode.get(),positions=json.loads(json.dumps(self.positions)),prepare_id=1,start_id=0,discard_id=0,stop=False,auto_start=start)
            self.session.update(tray_mode=self.tray_mode.get(),tray_objects=tray_objects)
            self.session.update(distractor_min=self.distractor_min.get(),distractor_max=self.distractor_max.get())
            self.session.update(collection=self.collection.get(),cycles=cycles)
            self.session.update(dataset_purpose=purpose,dataset_split=split,session_seed=seed)
            self.write()
            self.log_filter=LogFilter(); self.log_tail=LogTail()
            self.live_positions=None; self.live_tray=None
            self.live_geometry=None; self.dashboard=Dashboard(episodes)
            self.dashboard.phase='reset'; self.dashboard.stage='Launching / waiting for environment'; self.update_dashboard()
            self.add_log('info',f'Launching {self.target.get()}: goal {episodes} saved successes; no attempt limit.')
            launcher=ROOT.parents[3]/'_isaac_sim'/'python.bat'
            args=[str(launcher),str(ROOT/'record.py'),'--object',self.target.get(),'--episodes',str(episodes),
                  '--max-attempts','0','--record_mode',self.skill.get(),'--out_dir',str(output),'--session-config',str(self.session_path),
                  '--camera-size',self.camera_size.get(),'--randomization-seed',str(seed),
                  '--dataset-purpose',purpose,'--dataset-split',split,'--tray-occupancy',self.tray_mode.get()]
            log=ROOT/'debug'/'logs'/'panel_runs'/stamp/(self.target.get()+'.log')
            log.parent.mkdir(parents=True,exist_ok=True)
            self.log_file=log.open('w',encoding='utf-8')
            self.session['console_log']=str(log); self.write()
            env=os.environ.copy(); env.pop('P4_SYMMETRIC_GRASP',None)
            self.process=subprocess.Popen(args,cwd=ROOT,stdout=self.log_file,stderr=subprocess.STDOUT,env=env)
            for control in self.controls: control.configure(state='disabled')
            self.status.set('Launching Isaac GUI; preparing the shared scene…')
        except Exception as error:
            self.add_log('error',str(error))
            messagebox.showerror('Cannot prepare',str(error))

    def command(self,key):
        if not self.process or self.process.poll() is not None:
            return
        if key=='start_id' and self.read_status().get('state')!='ready_to_record':
            messagebox.showinfo('Scene not ready','Wait for the settled scene and READY status.'); return
        if key=='start_id': self.session['auto_start']=True
        self.session[key]+=1; self.write()

    def stop(self):
        if self.process and self.process.poll() is None:
            self.session['stop']=True; self.write(); self.status.set('Stopping at the next recorder check…')

    def read_status(self):
        if not self.session_path: return {}
        try: return json.loads(self.session_path.with_suffix('.status.json').read_text())
        except (OSError,ValueError): return {}

    def poll(self):
        self.pump_log()
        if self.process:
            code=self.process.poll()
            if code is None:
                state=self.read_status()
                self.live_geometry=state.get('instrument_geometry')
                live=state.get('table_positions')
                if live and set(live)==set(NAMES): self.live_positions=live
                self.live_tray=state.get('tray_objects')
                self.coverage=state.get('coverage'); self.redraw()
                if state: self.status.set(f"{state['state'].upper()} | {state['object']} | attempt {state.get('attempt',0)} | saved {state.get('saved',0)}/{state.get('requested','?')} | Tray: {', '.join(state.get('tray_objects',[])) or 'empty'} | {state.get('message','')}")
            else:
                self.log_file.close(); self.log_file=None; self.process=None
                metrics=self.session_path.parent/'run_metrics.json'
                try: summary=json.loads(metrics.read_text(encoding='utf-8')) if metrics.exists() else {}
                except (OSError,ValueError): summary={}
                self.dashboard.finish(summary,code); self.update_dashboard()
                self.status.set(f"Finished: exit={code}, saved successes={summary.get('success_count','?')}, failures={summary.get('failure_count','?')}. {self.session_path.parent}")
                self.add_log('success' if code==0 else 'error',self.status.get())
                if code and self.log_filter.last_reason: self.add_log('error','Last reason: '+self.log_filter.last_reason)
                if code and self.read_status().get('message'): self.add_log('error',self.read_status()['message'])
                if self.collection.get()=='grid_cycles':
                    from phase4_coverage import progress
                    self.coverage=progress(int(summary.get('success_count',0)),self.session['cycles'],self.layout['grid_rows'],self.layout['grid_cols'])
                    report_path=self.session_path.parent/'coverage_report.json'
                    try:
                        report=json.loads(report_path.read_text(encoding='utf-8'))
                        self.coverage.update(report)
                        self.add_log('success' if report['complete'] else 'error',f"H5 coverage: {report['counts']} | complete={report['complete']}")
                    except (OSError,ValueError,KeyError):
                        self.add_log('error','No valid on-disk coverage report. Counts are provisional, not proof of complete coverage.')
                    self.redraw()
                for control in self.controls: control.configure(state='readonly' if isinstance(control,ttk.Combobox) else 'normal')
        self._poll_id=self.root.after(300,self.poll)

    def choose_output(self):
        if self.process and self.process.poll() is None:
            messagebox.showinfo('Run active','The folder buttons follow the active run. Browse another run after stopping this one.')
            return
        folder=filedialog.askdirectory(title='Choose a saved run / object folder',initialdir=str(ROOT/'datasets'/'panel_runs'))
        if folder:
            self.output_dir=Path(folder); self.output_text.set(str(self.output_dir))

    def open_output(self,kind='run'):
        output=self.output_dir or (self.session_path.parent if self.session_path else None)
        if output is None:
            messagebox.showinfo('No run selected','Launch a recording or use Browse saved run first.'); return
        output=Path(output)
        if not output.is_dir():
            messagebox.showerror('Folder unavailable',str(output)); return
        candidates={'run':[output], 'pick':[output/'pick_policy'], 'place':[output/'place_policy'],
                    'gif':[output/'gifs',output/'gifs_semantic',output/'previews'],
                    'failure':[output/'failure_previews']}[kind]
        target=next((p for p in candidates if p.is_dir()),None)
        if kind=='gif' and target is None:
            # Recorder previews may be saved beside an episode rather than in a gifs directory.
            target=next((p.parent for p in output.rglob('*.gif')),None)
        if target is None:
            explanation={'pick':'No pick folder yet; wait for a saved episode and check record mode.',
                         'place':'No place folder yet; pick-only recording does not produce place data.',
                         'gif':'No GIF has been generated in this run. H5 recording and GIF export are separate.',
                         'failure':'No diagnostic failure previews have been written.'}.get(kind,'Folder unavailable.')
            messagebox.showinfo('Not available yet',explanation+'\n\nRun: '+str(output)); return
        try:
            os.startfile(str(target))
        except OSError as exc:
            self.add_log('error',f'Cannot open folder: {exc}'); messagebox.showerror('Cannot open folder',str(exc))

    def close(self):
        if self.export_process and self.export_process.poll() is None:
            messagebox.showinfo('Export active','Wait for export to finish before closing the panel.'); return
        self.stop()
        if self._poll_id: self.root.after_cancel(self._poll_id)
        self.root.destroy()


if __name__=='__main__':
    app=tk.Tk(); Panel(app); app.mainloop()
