"""Validate canonical sources, compatibility links, and all recorder dependencies."""
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
mapping = json.loads((ROOT/'docs/FILE_MAP.json').read_text())
for legacy, canonical in mapping.items():
    path = ROOT / canonical
    if not path.is_file() or not (ROOT/legacy).is_file():
        raise RuntimeError('Missing entry: ' + legacy)
    if path.suffix == '.py':
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        if canonical not in (ROOT/legacy).read_text():
            raise RuntimeError('Shim points to wrong source: ' + legacy)
for name in ('scene_layout.json', 'camera_layout.json', 'shared_layout.json',
             'camera_layout_front_tray_candidate.json', 'instrument_preview_geometry.json'):
    if not (ROOT/name).samefile(ROOT/'env'/name):
        raise RuntimeError('Environment compatibility link diverged: ' + name)
for name in ('validation', 'test_runs', 'archive', 'tmp', 'output'):
    if not (ROOT/name).samefile(ROOT/'debug'/name):
        raise RuntimeError('Debug compatibility junction diverged: ' + name)
from object_handlers import HANDLERS
from runner import _validate_files
for handler in HANDLERS.values():
    _validate_files(handler)
print('PASS: canonical sources, imports, env links, debug junctions, and five recorder backends')
