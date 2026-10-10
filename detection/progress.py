"""Read committed collection progress without starting Isaac or rewriting indexes."""
import json
from pathlib import Path


def read_collection(folder):
    folder=Path(folder)
    config=json.loads((folder/'config.json').read_text())
    if config.get('contract')!='p4_static_detection_v1':
        raise ValueError('This folder is not a static detection collection')
    status_path=folder/'status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    scenes=images=annotations=0
    for path in sorted((folder/'train/scenes').glob('scene_*/scene.json')):
        scene=json.loads(path.read_text())
        if scene['scene_id']!=scenes: raise ValueError('Missing or out-of-order committed scene')
        scenes+=1
        images+=len(scene['views'])
        annotations+=sum(len(view['boxes']) for view in scene['views'])
    return config,dict(scenes=scenes,images=images,annotations=annotations,
                       goal=max(scenes,int(status.get('goal',100))),state=status.get('state','stopped'))
