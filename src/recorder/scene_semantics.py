"""Shared semantic extension; IDs 0..7 retain their historical meaning."""
CLASSES = {'table': 8, 'floor': 9, 'room': 10}


def complete_environment_labels(semantic, instance_ids, labels):
    """Resolve unlabeled robot mesh parts using renderer prim IDs, not RGB guesses."""
    result = semantic.copy()
    for raw_id, value in labels.items():
        path = str(value.get('primPath',value.get('path',''))) if isinstance(value,dict) else str(value)
        path = path.lower()
        label = None
        if '/robot/' in path or path.endswith('/robot'):
            label = 1
        elif '/hospitalroom/' in path:
            label = 8 if '/furniture/table_' in path else (9 if 'floor' in path else 10)
        elif 'groundplane' in path:
            label = 9
        if label is not None:
            result[instance_ids == int(raw_id)] = label
    return result


def audit_preview(env, ns):
    import json
    from pathlib import Path
    import numpy as np
    from PIL import Image, ImageDraw
    names = ('camera','grip_cam_b','cam_top','cam_left','cam_right','cam_tray')
    size = env.scene['camera'].cfg.width
    canvas = Image.new('RGB',(size*3,size*4))
    counts = {}
    for i,name in enumerate(names):
        camera = env.scene[name]
        rgb = camera.data.output['rgb'][0,...,:3].detach().cpu().numpy().astype(np.uint8)
        semantic = ns['_phase3_extract_semantic_u16'](camera,size,size,cam_name=name)
        ids, pixels = np.unique(semantic, return_counts=True)
        counts[name] = {str(int(k)):int(v) for k,v in zip(ids,pixels)}
        x,y = (i%3)*size,(i//3)*size*2
        canvas.paste(Image.fromarray(rgb),(x,y))
        canvas.paste(Image.fromarray(ns['colorize_semantic'](semantic)),(x,y+size))
        ImageDraw.Draw(canvas).text((x+4,y+4),name,fill='white',stroke_width=1,stroke_fill='black')
    out = Path(ns['args_cli'].out_dir)
    canvas.save(out/'scene_camera_semantic_preview.png')
    report = dict(pixel_counts=counts, classes=ns['PHASE3_SEMANTIC_CLASS_IDS'],clutter=ns.get('_p4_clutter'))
    (out/'scene_label_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for label in ('robot','surgical_tray','table',ns['PHASE3_TARGET_OBJECT']):
        key = str(ns['PHASE3_SEMANTIC_CLASS_IDS'][label])
        if not any(row.get(key,0)>0 for row in counts.values()):
            raise RuntimeError('Required semantic class missing from all cameras: '+label)
    ns['_p4_scene_label_audit'] = counts


def install(ns):
    if ns.get('_p4_scene_semantics_installed'):
        return
    ns['_p4_scene_semantics_installed'] = True
    ns['PHASE3_SEMANTIC_CLASS_IDS'].update(CLASSES)
    ns['PHASE3_SEMANTIC_ID_TO_CLASS'].update({v:k for k,v in CLASSES.items()})
    normalize = ns['_phase3_normalize_semantic_label']
    def label(value):
        text = str(value).strip().lower()
        for name in CLASSES:
            if name in text:
                return name
        return normalize(value)
    ns['_phase3_normalize_semantic_label'] = label
    extract = ns['_phase3_extract_semantic_u16']
    def extract_complete(camera, height, width, cam_name='camera'):
        semantic = extract(camera,height,width,cam_name=cam_name)
        raw = camera.data.output['instance_id_segmentation_fast'][0].detach().cpu().numpy().squeeze(-1)
        info = camera.data.info
        info = info[0] if isinstance(info,(list,tuple)) else info
        return complete_environment_labels(semantic,raw,info['instance_id_segmentation_fast']['idToLabels'])
    ns['_phase3_extract_semantic_u16'] = extract_complete
    colorize = ns['colorize_semantic']
    def colors(seg):
        result = colorize(seg)
        for value, color in ((8,(45,150,75)),(9,(110,110,120)),(10,(180,170,155))):
            result[seg == value] = color
        return result
    ns['colorize_semantic'] = colors
    from src.recorder.instance_labels import install as install_instances
    install_instances(ns)
