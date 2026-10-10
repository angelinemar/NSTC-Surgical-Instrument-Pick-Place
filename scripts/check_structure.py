"""Validate canonical sources, organized compatibility shims, and dependencies."""
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'compat'))
mapping = json.loads((ROOT/'docs/FILE_MAP.json').read_text())
for legacy, canonical in mapping.items():
    path = ROOT / canonical
    shim = ROOT / ('record.py' if legacy == 'record.py' else 'compat/' + legacy)
    if Path(legacy).suffix != '.py':
        continue
    if not path.is_file() or not shim.is_file():
        raise RuntimeError('Missing entry: ' + legacy)
    if path.suffix == '.py':
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        if canonical not in shim.read_text():
            raise RuntimeError('Shim points to wrong source: ' + legacy)
for name in ('scene_layout.json', 'camera_layout.json', 'shared_layout.json',
             'camera_layout_front_tray_candidate.json', 'instrument_preview_geometry.json'):
    if not (ROOT/'env'/name).is_file() or (ROOT/name).exists():
        raise RuntimeError('Environment configuration must exist only under env/: ' + name)
from object_handlers import HANDLERS
from runner import _validate_files
for handler in HANDLERS.values():
    _validate_files(handler)
allowed = {'.git', '.gitignore', 'README.md', 'RUNME.ps1', 'RUN_DETECTION.ps1', 'record.py'}
unexpected = sorted(p.name for p in ROOT.iterdir() if p.is_file() and p.name not in allowed)
if unexpected:
    raise RuntimeError('Unexpected root files: ' + ', '.join(unexpected))
for required in ('training/rfdetr_pipeline.py', 'training/rfdetr/requirements.txt',
                 'scripts/launchers/setup_rfdetr.ps1'):
    if not (ROOT/required).is_file():
        raise RuntimeError('Missing training pipeline file: ' + required)
print('PASS: clean root, organized compatibility shims, canonical env files, and five recorder backends')
