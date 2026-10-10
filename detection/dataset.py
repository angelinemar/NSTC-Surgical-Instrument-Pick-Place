"""Scene-grouped PNG/COCO storage and deterministic camera sampling."""
import json
import math
import os
from pathlib import Path

CLASSES = ('scalpel', 'scissor', 'love_retractor', 'kelly', 'scalpel_type2')
CATEGORIES = [dict(id=i+1, name=n, supercategory='instrument') for i,n in enumerate(CLASSES)]


def camera_views(rng, layout, count):
    # Bounding sphere around table workspace AND tray, including raised tools.
    xmin, xmax = layout['grid_x']; ymin, ymax = layout['grid_y']
    tx, ty = layout['tray_xy']
    xmin, xmax = min(xmin, tx-.28), max(xmax, tx+.28)
    ymin, ymax = min(ymin, ty-.12), max(ymax, ty+.12)
    center = [(xmin+xmax)/2, (ymin+ymax)/2, .035]
    radius = math.sqrt(((xmax-xmin)/2)**2 + ((ymax-ymin)/2)**2 + .06**2)
    # 16 mm focal length, 20.955 mm square aperture.
    distance = radius / math.sin(math.atan(20.955/32)) * 1.12
    phase = rng.uniform(0, 2*math.pi)
    result=[]
    for i in range(count):
        az=phase+2*math.pi*i/count+rng.uniform(-.06,.06)
        el=math.radians(rng.uniform(40,65)); d=distance*rng.uniform(1.,1.05)
        eye=[center[0]+d*math.cos(el)*math.cos(az), center[1]+d*math.cos(el)*math.sin(az), center[2]+d*math.sin(el)]
        result.append(dict(name=f'ring_{i:02d}', eye=eye, target=center))
    result.append(dict(name='top', eye=[center[0]+.001,center[1],center[2]+distance], target=center))
    return result


def write_json(path, data):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2),encoding='utf-8'); os.replace(tmp,path)


def export_index(output):
    """Index only atomically committed scenes. Pick/place data never enters here."""
    output=Path(output); images=[]; annotations=[]; scenes=[]
    for folder in sorted((output/'train'/'scenes').glob('scene_*')):
        metadata=json.loads((folder/'scene.json').read_text(encoding='utf-8'))
        if metadata['scene_id'] != len(scenes):
            raise ValueError('Missing or out-of-order scene: '+str(folder))
        scenes.append(metadata['scene_id'])
        for view in metadata['views']:
            for kind in ('rgb','semantic','instance'):
                if not (folder/f'{view["name"]}_{kind}.png').is_file():
                    raise ValueError('Incomplete committed view: '+str(folder)+'/'+view['name'])
            iid=len(images)+1
            images.append(dict(id=iid,file_name=f'scenes/{folder.name}/{view["name"]}_rgb.png',
                               width=448,height=448,scene_id=metadata['scene_id'],camera=view['name']))
            for box in view['boxes']:
                annotations.append(dict(box,id=len(annotations)+1,image_id=iid))
    write_json(output/'train'/'_annotations.coco.json',dict(images=images,annotations=annotations,categories=CATEGORIES))
    return dict(scenes=len(scenes),images=len(images),annotations=len(annotations))
