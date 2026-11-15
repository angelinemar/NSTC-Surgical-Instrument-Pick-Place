"""Native-resolution camera sequences, never shuffled slides."""
from pathlib import Path
import html
from PIL import Image,ImageDraw,ImageFont

def save_gif(frames,output,skill,target,success,steps,model_label='DP'):
 if not frames:return None
 output=Path(output);outcome='SUCCESS' if success else 'FAIL'
 try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
 except OSError:font=ImageFont.load_default()
 rendered=[]
 for step,front,top in frames:
  a=Image.fromarray(front);b=Image.fromarray(top)
  image=Image.new('RGB',(a.width+b.width,max(a.height,b.height)+64),'white')
  image.paste(a,(0,64));image.paste(b,(a.width,64))
  draw=ImageDraw.Draw(image)
  draw.text((12,5),f'{model_label} | {skill.upper()} | {target} | {outcome} | t={step*.02:.2f}s',font=font,fill='#137547' if success else '#b22222')
  draw.text((12,34),'Front camera',font=font,fill='black')
  draw.text((a.width+12,34),'Top camera',font=font,fill='black')
  rendered.append(image)
 durations=[max(20,int((frames[i+1][0]-f[0])*.02*1000)) if i+1<len(frames) else 1000 for i,f in enumerate(frames)]
 path=output/f'rollout_{outcome}.gif'
 rendered[0].save(path,save_all=True,append_images=rendered[1:],duration=durations,loop=0,optimize=False)
 root=output.parent
 links=['<html><meta charset="utf-8"><title>DP rollout recordings</title><body style="background:white;font-family:sans-serif"><h1>DP rollout recordings</h1><p>Front + top cameras at native resolution. Recorded motion; final outcome labeled. Earlier trials without captured frames are not included.</p>']
 for p in sorted(root.glob('*/rollout_*.gif')):
  rel=p.relative_to(root).as_posix();label=p.parent.name+' — '+p.stem
  links.append(f'<p><a href="{html.escape(rel)}">{html.escape(label)}</a></p>')
 links.append('</body></html>')
 (root/'GIF_INDEX.html').write_text('\n'.join(links))
 return str(path)
