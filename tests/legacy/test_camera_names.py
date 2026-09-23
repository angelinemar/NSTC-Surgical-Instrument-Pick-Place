"""CPU-only P4 name, pose, resolution and legacy dataset contract checks."""
import ast
import json
from pathlib import Path
import phase3_camera_tuning as tuning
from phase4_camera_names import sensor_names, public_name, FRONT_PRIM

root=next(p for p in Path(__file__).resolve().parents if (p/'record.py').exists())
active=sensor_names(json.loads((root/'camera_layout.json').read_text())['cameras'])
assert len(tuning.PHASE3_CAMERAS)==6
assert tuning.PHASE3_CAMERAS['camera']['prim_path']=='{ENV_REGEX_NS}/cam_front'
assert tuning.CAMERA_WIDTH==448 and tuning.CAMERA_HEIGHT==336 and tuning.CAMERA_OUTPUT_CROP_SIZE==224
for name,pose in active.items():
    assert tuple(pose['pos'])==tuning.PHASE3_CAMERAS[name]['pos']
    assert tuple(pose['rot'])==tuning.PHASE3_CAMERAS[name]['rot']
public={public_name(k):v for k,v in active.items()}
assert 'cam_front' in public and 'camera' not in public
assert sensor_names(public)==active
try:
    sensor_names({'camera':{'pos':[0]},'cam_front':{'pos':[1]}})
    raise AssertionError('Conflicting aliases must fail')
except ValueError:
    pass
for path in (root/'backends').glob('*recorder.py'):
    source=path.read_text(encoding='utf-8-sig')
    ast.parse(source)
    assert '/World/envs/env_0/Camera' not in source
    assert '{ENV_REGEX_NS}/cam_front' in source
    if path.name == 'phase3_grid_split_scalpel_recorder.py':
        assert FRONT_PRIM in source  # Only this backend defines the legacy GUI tuner.
    assert 'env.scene["camera"]' in source  # Deliberate legacy sensor alias.
    assert '"front_rgb"' in source and '"front_semantic"' in source
    print('PASS',path.name,'cam_front prim; legacy single sensor/H5 contract retained')
from phase4_camera_preview import PATHS
assert PATHS['camera']==FRONT_PRIM
print('PASS 6 cameras; current poses preserved; canonical/legacy JSON roundtrip; crop224; preview path aligned')
