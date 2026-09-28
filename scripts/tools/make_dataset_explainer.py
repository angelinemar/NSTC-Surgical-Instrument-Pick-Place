"""PPT-ready visual explanations of the recorded H5 schema using real frames."""
import json
from pathlib import Path
import zipfile

import h5py
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'debug/output/dataset_visual_explainer'
SOURCE = ROOT/'datasets/panel_runs/20260918_231630_309506/scalpel/pick_policy/scalpel/episode_000009.h5'
W, H = 1600, 900
BG, INK, TEAL, MUTED = '#ffffff', '#173c43', '#087f8c', '#577078'
BLUE, ORANGE, GREEN = '#3879b6', '#ba6b27', '#498162'
COLORS = np.array([(20,20,20),(60,150,255),(255,150,40),(80,220,100),
                   (245,220,60),(225,80,210),(80,225,220),(255,90,90)],np.uint8)
CAMERAS = ('front','wrist','cam_top','cam_left','cam_right','cam_tray')
INSTRUMENTS = ('scalpel','scissor','love_retractor','kelly','scalpel_type2')
TITLES = ('One moment. Six camera views.', 'One camera. Three different answers.',
          'Robot state and robot commands.', 'Class ID and tray slot are different.',
          'From an image pixel to a 3D location.', 'What is saved, and what training uses.')
FILENAMES = ('01_six_cameras','02_rgb_depth_semantic','03_state_and_actions',
             '04_semantic_classes_and_slots','05_calibration','06_training_and_metadata')


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/bahnschrift.ttf',size)


def text(d, xy, value, size=28, fill=INK, anchor=None):
    d.text(xy,str(value),font=font(size),fill=fill,anchor=anchor)


def card(d, box, fill='white', outline=None, radius=20):
    d.rounded_rectangle(box,radius=radius,fill=fill,outline=outline,width=3)


def wrapped(d, xy, value, width, size=28, fill=INK, spacing=9):
    words=value.split(); lines=[]; line=''
    for word in words:
        candidate=(line+' '+word).strip()
        if d.textlength(candidate,font=font(size))>width and line:
            lines.append(line); line=word
        else: line=candidate
    if line: lines.append(line)
    for i,line in enumerate(lines): text(d,(xy[0],xy[1]+i*(size+spacing)),line,size,fill)


def arrow(d, a, b, color=TEAL, width=5, dashed=False):
    a,b=np.asarray(a,float),np.asarray(b,float); delta=b-a
    length=np.linalg.norm(delta); unit=delta/length; perp=np.array([-unit[1],unit[0]])
    if dashed:
        for t in np.arange(0,length-15,20):
            d.line((*tuple(a+unit*t),*tuple(a+unit*min(t+10,length-15))),fill=color,width=width)
    else: d.line((*a,*b),fill=color,width=width)
    d.polygon([tuple(b),tuple(b-unit*17+perp*8),tuple(b-unit*17-perp*8)],fill=color)


def base(chapter, progress):
    image=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(image)
    text(d,(55,25),f'INSIDE THE RECORDED DATASET   /   {chapter+1:02d} OF 06',21,TEAL)
    text(d,(55,65),TITLES[chapter],49)
    d.line((55,132,1545,132),fill='#c6d3cd',width=2)
    for i in range(6):
        x=55+i*249
        d.rounded_rectangle((x,870,x+230,875),radius=2,fill=TEAL if i==chapter else '#d0d9d1')
        if i==chapter: d.rounded_rectangle((x,870,x+int(230*progress),875),radius=2,fill=ORANGE)
    return image,d


def paste(image, array, xy, size, nearest=False):
    im=Image.fromarray(array) if isinstance(array,np.ndarray) else array
    im=im.resize(size,Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC)
    image.paste(im,xy,im if im.mode=='RGBA' else None)


def scene0(data, i, progress):
    image,d=base(0,progress)
    text(d,(60,154),'Every time step captures the same moment from six viewpoints.',29)
    for c,camera in enumerate(CAMERAS):
        x=60+(c%3)*302; y=211+(c//3)*301
        card(d,(x,y,x+280,y+287))
        text(d,(x+15,y+10),camera.replace('cam_','').upper(),22,TEAL)
        paste(image,data[camera+'_rgb'][i],(x+27,y+48),(225,225))
    card(d,(1000,212,1540,799),fill='#173c43')
    text(d,(1030,240),'ONE EPISODE = A SEQUENCE',25,'#c4e2d7')
    text(d,(1030,291),'T = number of frames',37,'white')
    text(d,(1030,350),f'T = {data["T"]} in this pick segment',28,'white')
    for n in range(5):
        x=1050+n*77; y=442-(n%2)*10
        card(d,(x,y,x+67,y+75),fill='#477277',radius=7)
        text(d,(x+33,y+36),str(1+n),24,'white',anchor='mm')
    text(d,(1460,472),'...',30,'white')
    arrow(d,(1050,559),(1484,559),color='#70d5c1')
    d.ellipse((1050+int(420*progress)-8,551,1050+int(420*progress)+8,567),fill='#f3cb74')
    text(d,(1030,602),f'Current row: {i+1} / {data["T"]}',30,'white')
    text(d,(1030,652),'6 views x RGB, depth, semantic',25,'#c4e2d7')
    text(d,(1030,701),'20 ms per recorded step (50 Hz)',24,'#c4e2d7')
    text(d,(1030,748),'Sampled frames / slowed for explanation',22,'#c4e2d7')
    text(d,(60,825),'T can differ between episodes. Camera arrays align with the robot data at each row.',25,MUTED)
    return image


def scene1(data,i,progress):
    image,d=base(1,progress)
    sem=data['front_semantic'][i]; depth=data['front_depth'][i].astype(float)
    amount=np.clip((depth-1.0)/1.0,0,1)
    near=np.array([27,82,161]); far=np.array([249,210,85])
    depth_rgb=(near[None,None,:]*(1-amount[...,None])+far[None,None,:]*amount[...,None]).astype(np.uint8)
    ys,xs=np.where(sem==3); pos=len(xs)//2; u,v=int(xs[pos]),int(ys[pos])
    images=(data['front_rgb'][i],depth_rgb,COLORS[sem])
    headings=('RGB / APPEARANCE','DEPTH / DISTANCE','SEMANTIC / CLASS')
    answers=('What does it look like?','How far is this pixel?','Which object is this pixel?')
    for n in range(3):
        x=60+n*510
        card(d,(x,176,x+480,784))
        text(d,(x+24,199),headings[n],28,TEAL)
        text(d,(x+24,246),answers[n],25)
        paste(image,images[n],(x+50,294),(380,380),nearest=n>0)
        px,py=x+50+(u+0.5)*380/224,294+(v+0.5)*380/224
        d.ellipse((px-12,py-12,px+12,py+12),outline='white',width=7)
        d.ellipse((px-12,py-12,px+12,py+12),outline=INK,width=3)
        detail=(f'R/G/B: {list(map(int,data["front_rgb"][i,v,u]))}',
                f'Distance: {depth[v,u]:.3f} m','Class 3 = scalpel')[n]
        text(d,(x+240,708),detail,26,anchor='mm')
        text(d,(x+240,752),('3 channels / uint8','1 channel / float16','1 channel / uint16')[n],21,MUTED,anchor='mm')
    text(d,(62,814),'Same marked pixel in all three views. Each image is 224 x 224 pixels.',27)
    text(d,(62,848),'Depth display: blue = 1.0 m, yellow = 2.0 m or farther. Semantic colors visualize stored integer IDs.',20,MUTED)
    return image


def boxes(d, x,y, groups, cell=36, height=50, focus=-1):
    cursor=x
    for n,(count,color,title) in enumerate(groups):
        start=cursor
        for j in range(count):
            card(d,(cursor,y,cursor+cell-5,y+height),fill=color,radius=5)
            text(d,(cursor+(cell-5)/2,y+height/2),str(j+1),18,'white',anchor='mm')
            cursor+=cell
        text(d,((start+cursor-5)/2,y+height+24),title,21,color,anchor='mm')
        if n==focus: d.rounded_rectangle((start-5,y-6,cursor,y+height+5),radius=8,outline=INK,width=3)
        cursor+=18


def scene2(data,i,progress):
    image,d=base(2,progress)
    card(d,(55,169,950,770));card(d,(995,169,1545,770))
    text(d,(85,192),'MEASURED ROBOT STATE',31,TEAL)
    text(d,(85,240),'robot_proprio: 16 numbers per time step',27)
    # Diagram is conceptual, not a forward-kinematics reconstruction.
    joints=np.array([[126,469],[178,440],[232,386],[285,403],[340,335],[397,333],[442,370]],float)
    joints[2:5,1]+=np.sin(progress*np.pi*2)*9
    d.line([tuple(p) for p in joints],fill='#9cafb1',width=20)
    for j,(x,y) in enumerate(joints):
        d.ellipse((x-14,y-14,x+14,y+14),fill=BLUE)
        text(d,(x,y),j+1,17,'white',anchor='mm')
    d.line((434,389,434,415),fill=TEAL,width=7);d.line((453,389,453,415),fill=TEAL,width=7)
    text(d,(105,518),'7 joints + 2 fingers',25)
    text(d,(105,554),'Robot diagram is schematic.',18,MUTED)
    text(d,(535,326),'Tool position',27,ORANGE)
    text(d,(535,370),'x, y, z',38,ORANGE)
    text(d,(535,428),'Tool orientation',27,GREEN)
    text(d,(535,472),'qw, qx, qy, qz',33,GREEN)
    boxes(d,87,626,[(7,BLUE,'7 joints'),(2,TEAL,'2 fingers'),(3,ORANGE,'3 position'),(4,GREEN,'4 orientation')],cell=41,focus=int(progress*4)%4)
    text(d,(1025,192),'COMMANDED ACTION',31,ORANGE)
    text(d,(1025,240),'actions: 8 numbers',28)
    text(d,(1025,302),'Where should the tool go?',25)
    boxes(d,1030,359,[(3,ORANGE,'3 position'),(4,GREEN,'4 orientation'),(1,TEAL,'1 grip')],cell=46)
    grip=float(data['actions'][i,-1])
    gap=30 if grip>0 else 7
    cx=1250
    d.line((cx-70,565,cx+70,565),fill=INK,width=12)
    for sign in (-1,1):d.line((cx+sign*gap,565,cx+sign*gap,630),fill=TEAL,width=12)
    text(d,(1250,672),'OPEN (+1)' if grip>0 else 'CLOSE (-1)',29,TEAL,anchor='mm')
    text(d,(1025,720),'float32 / robot-base coordinates',22,MUTED)
    text(d,(63,809),'Stored state (18) = robot_proprio (16) + object_type_id (1) + skill_id (1)',28)
    text(d,(63,846),'Measured pose and commanded pose are different signals. EE means the robot end effector / tool.',20,MUTED)
    return image


def scene3(data,i,progress):
    image,d=base(3,progress)
    active=min(4,int(progress*5))
    text(d,(65,164),'Semantic ID answers: "What is it?"',29)
    for n,name in enumerate(('background','robot','surgical tray')):
        x=65+n*490
        card(d,(x,215,x+455,292))
        color=tuple(map(int,COLORS[n]))
        d.ellipse((x+15,231,x+57,273),fill=color)
        text(d,(x+78,232),f'{n}  {name}',27)
        text(d,(x+435,253),'NO SLOT',18,MUTED,anchor='rm')
    text(d,(65,324),'INSTRUMENT CLASS',23,MUTED)
    text(d,(700,324),'SEMANTIC ID',23,MUTED)
    text(d,(1210,324),'TRAY SLOT',23,MUTED)
    for n,name in enumerate(INSTRUMENTS):
        y=370+n*79; color=tuple(map(int,COLORS[n+3]))
        card(d,(60,y,850,y+66),outline=TEAL if n==active else None)
        d.ellipse((79,y+15,113,y+49),fill=color)
        text(d,(134,y+16),name.replace('_',' ').title(),27)
        model=ROOT/'output/instrument_dimension_sheets'/f'{name}_top_model.png'
        if model.exists():
            with Image.open(model) as im:
                im.thumbnail((205,50),Image.Resampling.LANCZOS)
                image.paste(im,(432+(205-im.width)//2,y+8+(50-im.height)//2),im)
        text(d,(756,y+34),n+3,35,anchor='mm')
        arrow(d,(880,y+33),(1100,y+33),color=TEAL if n==active else '#bac8c8',dashed=n!=active)
        card(d,(1132,y,1524,y+66),fill=color,outline=INK if n==active else None)
        text(d,(1160,y+16),f'SLOT {n}',28,'#173c43')
        text(d,(1495,y+34),f'object type {n}',22,'#173c43',anchor='rm')
    text(d,(65,803),'Example: scalpel is semantic class 3, but belongs to tray slot 0.',29)
    text(d,(65,845),'Slots are fixed by instrument class, not placement order. Tray layout shown schematically.',21,MUTED)
    return image


def scene4(data,i,progress):
    image,d=base(4,progress)
    text(d,(65,166),'Calibration connects image measurements to the robot coordinate frame.',29)
    sem=data['front_semantic'][i]; ys,xs=np.where(sem==3); k=len(xs)//2; u,v=int(xs[k]),int(ys[k])
    card(d,(60,226,540,770));paste(image,data['front_rgb'][i],(95,281),(410,410))
    px,py=95+(u+.5)*410/224,281+(v+.5)*410/224
    d.line((px-17,py,px+17,py),fill='white',width=4);d.line((px,py-17,px,py+17),fill='white',width=4)
    text(d,(90,242),'PIXEL + DEPTH',25,TEAL)
    text(d,(90,720),f'(u, v) = ({u}, {v})',25)
    cards=[(620,260,1000,418),(620,491,1000,666),(1100,359,1535,639)]
    for idx,box in enumerate(cards):card(d,box,outline=TEAL if int(progress*3)%3==idx else None)
    text(d,(645,282),'CAMERA INTRINSICS K',26,TEAL)
    text(d,(645,331),'Focal length + image center',23)
    text(d,(645,375),'Pixel + depth -> camera point',23,MUTED)
    text(d,(645,514),'CAMERA POSE',26,TEAL)
    text(d,(645,562),'Position + orientation',25)
    text(d,(645,609),'Camera frame -> robot base',23,MUTED)
    arrow(d,(545,382),(609,339));arrow(d,(809,427),(809,479));arrow(d,(1008,571),(1086,571))
    text(d,(1130,386),'3D POINT IN ROBOT BASE',24,TEAL)
    center=(1240,526)
    arrow(d,center,(1400,526),color=ORANGE);text(d,(1417,522),'X',24,ORANGE)
    arrow(d,center,(1180,593),color=GREEN);text(d,(1160,601),'Y',24,GREEN)
    arrow(d,center,(1240,454),color=BLUE);text(d,(1232,429),'Z',24,BLUE)
    d.ellipse((1310,469,1328,487),fill=TEAL)
    text(d,(66,807),'Saved for every camera: raw K, cropped-image K, position, and quaternion.',27)
    text(d,(66,845),'Use the intrinsics that match the 224 x 224 crop. Depth is stored in metres.',22,MUTED)
    return image


def scene5(data,i,progress):
    image,d=base(5,progress)
    text(d,(65,166),'Current sensor-only training pipeline',30,TEAL)
    inputs=[('RGB from 6 cameras + robot_proprio',TEAL,218),
            ('Recorded actions',ORANGE,348),('Semantic class masks',BLUE,478)]
    targets=[('POLICY INPUT','What the policy observes'),('ACTION TARGET','What the policy learns to do'),
             ('SEGMENTATION TARGET','What each pixel should represent')]
    for n,((title,color,y),(target,sub)) in enumerate(zip(inputs,targets)):
        card(d,(65,y,805,y+102),outline=color)
        text(d,(90,y+35),title,30,color)
        arrow(d,(823,y+51),(971,y+51),color=color,dashed=int(progress*3)%3!=n)
        card(d,(995,y,1535,y+102),fill=color)
        text(d,(1020,y+15),target,29,'white');text(d,(1020,y+59),sub,23,'white')
    card(d,(65,622,1535,780),fill='#e3e9e2')
    text(d,(90,641),'ALSO RECORDED: DEPTH, CALIBRATION, TASK IDs, AND DIAGNOSTIC / TEACHER DATA',24)
    text(d,(90,691),'stage_id = which phase   |   step_ids = index within the stage   |   dones = segment end',23)
    text(d,(90,733),'Example: LIFT_CLEAR has internal stage_id 8. step_ids restart when the stage changes.',22,MUTED)
    text(d,(65,816),'Training reads HDF5 arrays. PNG previews and GIFs are only for people.',29,TEAL)
    return image


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with h5py.File(SOURCE,'r') as h:
        data={key:h['observations'][key][:] for key in h['observations'] if key.endswith(('_rgb','_depth','_semantic'))}
        data.update(T=len(h['actions']),actions=h['actions'][:])
        assert h['observations/robot_proprio'].shape==(data['T'],16)
        assert h['observations/state'].shape==(data['T'],18)
        assert data['actions'].shape==(data['T'],8)
    draw_scenes=(scene0,scene1,scene2,scene3,scene4,scene5)
    all_gif=[]; report=[]; duration=240; total=30
    video=imageio_ffmpeg.write_frames(str(OUT/'recorded_dataset_explained.mp4'),(W,H),fps=25,
        codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',quality=8,
        output_params=['-movflags','+faststart'],macro_block_size=2)
    video.send(None)
    overview=Image.new('RGB',(1600,3*450),BG)
    for chapter,render in enumerate(draw_scenes):
        frames=[]
        for step in range(total):
            fraction=step/(total-1); index=min(data['T']-1,int(fraction*(data['T']-1)))
            image=render(data,index,fraction)
            for _ in range(6):video.send(np.asarray(image))
            frames.append(image)
        frames[16].save(OUT/(FILENAMES[chapter]+'.png'))
        overview.paste(frames[16].resize((800,450),Image.Resampling.LANCZOS),((chapter%2)*800,(chapter//2)*450))
        palette_sheet=Image.new('RGB',(W,H*5))
        for row,k in enumerate([0,7,14,21,29]):palette_sheet.paste(frames[k],(0,row*H))
        palette=palette_sheet.quantize(colors=256,method=Image.Quantize.MEDIANCUT)
        # Keep the white slide background exact after GIF palette reduction.
        colors=palette.getpalette()
        white_index=palette.getpixel((0,0))
        colors[white_index*3:white_index*3+3]=[255,255,255]
        palette.putpalette(colors)
        quantized=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in frames]
        for source,encoded in zip(frames,quantized):
            white_mask=Image.fromarray(np.all(np.asarray(source)==255,axis=2))
            encoded.paste(white_index,(0,0),white_mask)
        name=FILENAMES[chapter]+'.gif'
        quantized[0].save(OUT/name,save_all=True,append_images=quantized[1:],duration=duration,loop=0,optimize=False,disposal=2)
        all_gif.extend(quantized)
        report.append(dict(title=TITLES[chapter],gif=name,seconds=total*duration/1000))
        print('Saved:',name,flush=True)
    video.close()
    all_gif[0].save(OUT/'recorded_dataset_explained.gif',save_all=True,append_images=all_gif[1:],
                     duration=duration,loop=0,optimize=False,disposal=2)
    overview.save(OUT/'all_sections_overview.png')
    (OUT/'manifest.json').write_text(json.dumps(dict(source_h5=str(SOURCE.resolve()),chapters=report,
        language='English',size=[W,H],seconds=len(all_gif)*duration/1000,
        notes=['RGB/depth/semantic panels use matching rows from real H5.',
               'Depth display is clipped to 1.0-2.0 metres; raw data is unchanged.',
               'Robot and tray drawings are schematic.',
               'Current training allowlist is RGB plus robot_proprio; labels are supervision.']),indent=2))
    (OUT/'README.txt').write_text('RECORDED DATASET EXPLAINER\n\n'
        'Start with recorded_dataset_explained.mp4: one 43.2-second presentation video.\n'
        'A looping combined GIF and six individual GIFs are also included.\n'
        'Each chapter has a static PNG suitable for a separate PowerPoint slide.\n'
        'Everything is 1600 x 900 (16:9), with English labels.\n\n'
        '1. Six synchronized camera views and the meaning of T.\n'
        '2. RGB, depth, semantic: three arrays of the same scene.\n'
        '3. Robot proprioception (16), stored state (18), action (8).\n'
        '4. Semantic classes 0-7 versus instrument tray slots 0-4.\n'
        '5. Camera calibration: image pixel and depth to robot coordinates.\n'
        '6. Policy inputs, training targets, and metadata.\n\n'
        'Recordings provide the real RGB/depth/semantic frames. Other diagrams are explanatory.\n'
        'step_ids are local to a stage, not global sequence indices.\n'
        'Stage display order is not the same as internal stage_id.\n'
        'Default input usage shown follows training/export_sensor_only.py and training/sensor_policy.py.\n',encoding='utf-8')
    with zipfile.ZipFile(OUT/'dataset_visual_explainer.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.iterdir()):
            if path.suffix in ('.png','.gif','.mp4','.json','.txt'):archive.write(path,path.name)
    print('Done:',OUT.resolve(),flush=True)


if __name__=='__main__':main()
