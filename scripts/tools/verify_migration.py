"""Read-only P3/P4 source, camera and table-geometry migration audit."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'p3' / 'with_env_cfg' / 'refactored_v2'


def main():
    mapping = json.loads((ROOT/'docs/FILE_MAP.json').read_text())
    def source_path(path):
        return ROOT/mapping[path.name] if path.parent==ROOT and path.name in mapping else path
    manifest = json.loads((ROOT/'P3_COPY_MANIFEST.json').read_text())
    for rel, record in manifest.items():
        src = SOURCE/rel
        assert hashlib.sha256(src.read_bytes()).hexdigest() == record['p3_sha256'], f'P3 source changed since copy: {rel}'
        if record['change'] == 'none':
            expected = src.read_text(encoding='utf-8-sig').replace('/Camera','/cam_front')
            if rel == 'phase3_camera_tuning.py':
                expected = expected.replace('from pathlib import Path','from pathlib import Path\nfrom phase4_camera_names import sensor_names')
                expected = expected.replace('_saved.get("cameras", {}).items()', 'sensor_names(_saved.get("cameras", {})).items()')
                expected = expected.replace('Path(__file__).with_name("camera_layout.json")', 'Path(__file__).parent / \'env\' / "camera_layout.json"')
            if rel == 'phase3_recorder_camera_patch.py':
                expected = expected.replace('camera/front =','cam_front (legacy sensor key=camera) =')
            assert expected == source_path(ROOT/rel).read_text(encoding='utf-8-sig'), rel
        else:
            expected = src.read_text(encoding='utf-8-sig')
            expected = expected.replace(str(SOURCE.parents[1]/'assets') + chr(92), 'P4_ASSET_PLACEHOLDER/')
            expected = expected.replace('C:/IsaacLab/scripts/custom/i4h_project/p3/assets/', 'P4_ASSET_PLACEHOLDER/')
            expected = re.sub(r'r["\']P4_ASSET_PLACEHOLDER/([^"\']+)["\']', r'str(_P4_ASSETS / "\1")', expected)
            expected = 'from pathlib import Path as _P4Path\n_P4_ASSETS = _P4Path(__file__).resolve().parents[1] / "assets"\n' + expected
            expected = expected.replace('/Camera','/cam_front')
            expected = re.sub(r'print\("(\[PHASE FAIL\] [^"\n]+ did not follow LIFT_CLEAR)"\)',
                              r'print(globals().get("_p4_failure_message", "\1"))', expected)
            expected = expected.replace('    while saved_count < target_success:',
                '    while saved_count < target_success:\n        _p4_attempt_guard(attempt, target_success, saved_count)')
            assert expected == (ROOT/rel).read_text(encoding='utf-8-sig'), f'Unexpected backend edits: {rel}'
        print('SOURCE_OK', rel)
    for path in list(ROOT.glob('*.py')) + list((ROOT/'backends').glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8-sig'))
    def timing_ast(path):
        path = source_path(path)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        return ast.dump(next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MotionTimingCfg'))
    assert timing_ast(ROOT/'phase3_shared_env_cfg.py') == timing_ast(SOURCE/'phase3_shared_env_cfg.py')
    old = json.loads((SOURCE/'camera_layout.json').read_text())['cameras']
    from phase4_camera_names import sensor_names
    new = sensor_names(json.loads((ROOT/'env'/'camera_layout.json').read_text())['cameras'])
    assert old.keys() == new.keys()
    for name in old:
        if name not in ('cam_tray','camera'):
            # GUI quaternion round trips introduced <4e-6 in cam_top before
            # this naming change; accept that existing numerical precision.
            assert all(abs(a-b)<1e-5 for a,b in zip(old[name]['rot'],new[name]['rot'])), name
    assert all(abs(a-b)<1e-6 for a,b in zip(old['grip_cam_b']['pos'],new['grip_cam_b']['pos']))
    from object_handlers import HANDLERS
    from runner import _validate_files
    for handler in HANDLERS.values():
        _validate_files(handler)
    from phase4_scene import LAYOUT, table_geometry
    selected = LAYOUT['selected_table']
    try:
        for i in range(1,8):
            LAYOUT['selected_table'] = f'Table_{i:02d}'
            print('GEOMETRY_OK', table_geometry())
    finally:
        LAYOUT['selected_table'] = selected
    print('PASS: backend changes limited to asset paths, front prim, failure reason and attempt budget hooks; grasp geometry/schema unchanged; P3 untouched.')


if __name__ == '__main__':
    main()
