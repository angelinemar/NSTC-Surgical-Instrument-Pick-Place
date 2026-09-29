"""Rebuild the public, illustration-first pipeline PNG (Pillow required)."""
from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
INK, GREEN, BLUE, PURPLE = '#18303E', '#76B900', '#237FA8', '#8358B5'
im = Image.new('RGB', (3200, 1500), '#F7F9FB')
d = ImageDraw.Draw(im)


def text(x, y, s, size=30, color=INK, bold=False):
    face = 'arialbd.ttf' if bold else 'arial.ttf'
    d.text((x,y), s, font=ImageFont.truetype('C:/Windows/Fonts/'+face,size), fill=color)


def arrow(points, color=GREEN):
    d.line(points, fill=color, width=7, joint='curve')
    a,b=points[-2:];t=math.atan2(b[1]-a[1],b[0]-a[0])
    d.polygon([b,(b[0]-24*math.cos(t-.45),b[1]-24*math.sin(t-.45)),
               (b[0]-24*math.cos(t+.45),b[1]-24*math.sin(t+.45))],fill=color)


def tool(x,y,angle=0,color=INK,length=90):
    dx,dy=math.cos(angle)*length/2,math.sin(angle)*length/2
    d.line((x-dx,y-dy,x+dx,y+dy),fill=color,width=6)
    d.ellipse((x-dx-9,y-dy-9,x-dx+9,y-dy+9),outline=color,width=4)


def scene(x,y,w,h,boxes=False):
    d.rounded_rectangle((x,y,x+w,y+h),radius=16,fill='#409886')
    d.rounded_rectangle((x+15,y+15,x+w*.25,y+h-15),radius=9,fill='#CDD7DC')
    for j in range(4):tool(x+w*.13,y+40+j*(h-70)/4,math.pi/2,length=30)
    for i,(u,v,a) in enumerate(((.4,.3,.7),(.7,.2,1.1),(.52,.7,2.2),(.8,.65,.2))):
        xx,yy=x+w*u,y+h*v
        tool(xx,yy,a, length=min(w,h)*.22,color='#E6F1F2')
        if boxes:d.rectangle((xx-35,yy-40,xx+35,yy+40),outline='#CEFF62',width=4)


def network(x,y,color):
    layers=((0,3),(95,5),(190,4),(285,3))
    for (xa,na),(xb,nb) in zip(layers,layers[1:]):
        for a in range(na):
            for b in range(nb):
                d.line((x+xa,y+(a-(na-1)/2)*48,x+xb,y+(b-(nb-1)/2)*48),fill='#C9D6DF',width=2)
    for xx,n in layers:
        for j in range(n):
            yy=y+(j-(n-1)/2)*48
            d.ellipse((x+xx-14,yy-14,x+xx+14,yy+14),fill=color)


def stack(x,y,size=180):
    for offset in (32,16,0):
        d.rounded_rectangle((x+offset,y-offset,x+size+offset,y+size-offset),radius=12,fill='white',outline='#B4C9D1',width=3)
    scene(x+10,y+10,size-20,size-20)


def build():
    d.rectangle((0,0,3200,15),fill=GREEN)
    text(75,52,'P4  /  SURGICAL INSTRUMENT PICK & PLACE',27,GREEN,True)
    text(75,110,'Record once. Learn perception and action.',65,bold=True)
    text(75,210,'COLLECTION',28,GREEN,True)
    text(1350,210,'TRAINING',28,BLUE,True)
    text(2590,210,'INFERENCE',28,INK,True)
    # A scene illustration, camera glyphs and paired motion trajectories.
    scene(90,525,490,380)
    for x,y in ((130,405),(345,380),(555,430)):
        d.rounded_rectangle((x-40,y-25,x+35,y+25),radius=8,fill=INK)
        d.ellipse((x-16,y-16,x+16,y+16),outline='#B7D793',width=5)
        d.line((x,y+30,x,510),fill='#B8CCD3',width=3)
    text(90,980,'Randomized scene',37,bold=True)
    text(90,1035,'Lighting  /  pose  /  table clutter',27)
    text(130,340,'6 cameras',30,bold=True)
    arrow([(610,710),(725,710)])
    stack(755,565,245)
    text(760,870,'RGB 448',35,bold=True)
    # A paired action trace, not a measured trajectory.
    arrow([(750,1030),(800,1030),(830,1110),(895,1110),(930,1000)],BLUE)
    arrow([(945,1000),(985,1110),(1040,1110),(1080,1030)],PURPLE)
    text(758,1155,'Pick',28,BLUE,True);text(966,1155,'Place',28,PURPLE,True)
    text(736,1240,'Robot state + target + actions',27)
    arrow([(1080,710),(1190,710),(1190,480),(1320,480)],BLUE)
    arrow([(1190,710),(1190,1100),(1320,1100)],PURPLE)
    # DP branch.
    text(1360,295,'ACTION LEARNING',30,BLUE,True)
    stack(1370,405,165)
    text(1350,630,'6 views / 224',31,bold=True)
    text(1350,679,'+ state + target',27)
    arrow([(1585,480),(1700,480)],BLUE)
    network(1740,480,BLUE)
    text(1740,630,'Diffusion Policy',34,bold=True)
    text(1740,679,'Pick + Place checkpoints',27)
    arrow([(2075,480),(2200,480)],BLUE)
    for i in range(8):
        x=2230+i*26
        d.rectangle((x,425,x+17,550),fill=BLUE if i%2 else '#8ABFD4')
    text(2210,630,'Action sequence',30,bold=True)
    arrow([(2460,480),(2580,480)],BLUE)
    # End-effector and lift path, not a whole robot background.
    d.rounded_rectangle((2780,335,2870,420),radius=18,fill='#CAD6DF')
    d.line((2795,420,2795,475,2820,475),fill=INK,width=12)
    d.line((2855,420,2855,475,2830,475),fill=INK,width=12)
    tool(2825,525,.1,length=160)
    arrow([(2940,545),(2940,350)],BLUE)
    text(2620,630,'Pick / Place control',34,bold=True)
    text(2615,684,'Controller integration required',25)
    # Independent perception branch.
    text(1360,885,'OBJECT DETECTION',30,PURPLE,True)
    scene(1355,1010,230,180,True)
    text(1350,1240,'RGB 448 + boxes',31,bold=True)
    arrow([(1610,1100),(1700,1100)],PURPLE)
    network(1740,1100,PURPLE)
    text(1770,1240,'RF-DETR',34,bold=True)
    arrow([(2075,1100),(2200,1100)],PURPLE)
    # Checkpoint represented by a chip, not another text card.
    d.rounded_rectangle((2240,1035,2390,1185),radius=15,fill='#E9E1F2',outline=PURPLE,width=5)
    for i in range(4):
        yy=1060+i*32
        d.line((2215,yy,2240,yy),fill=PURPLE,width=5)
        d.line((2390,yy,2415,yy),fill=PURPLE,width=5)
    text(2264,1085,'DETR',28,PURPLE,True)
    text(2230,1240,'Detector',31,bold=True)
    arrow([(2460,1100),(2580,1100)],PURPLE)
    scene(2635,995,365,215,True)
    text(2635,1240,'Class + box + score',32,bold=True)
    d.line((75,1380,3125,1380),fill='#D5DFE6',width=2)
    text(75,1420,'Independent models share the same recording. RF-DETR is not an input to the current DP.',27)
    target=ROOT/'docs/media/pipeline.png';target.parent.mkdir(parents=True,exist_ok=True)
    im.save(target,dpi=(200,200))
    local=ROOT/'output/visual_guide_20260929/01_full_pipeline.png'
    if local.parent.exists():im.save(local,dpi=(200,200))
    print(target)


if __name__=='__main__':build()
