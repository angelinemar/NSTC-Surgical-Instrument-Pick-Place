"""Stable CLI: python scripts/p4.py record ... | panel | tool <name> ..."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if len(sys.argv) < 2:
    raise SystemExit(__doc__)
command = sys.argv.pop(1)
if command == 'tool':
    if len(sys.argv) < 2:
        raise SystemExit('tool requires a name, e.g. check_h5_consistency')
    name = sys.argv.pop(1).removesuffix('.py')
    if '/' in name or '\\' in name or not (ROOT/'scripts/tools'/f'{name}.py').is_file():
        raise SystemExit('Unknown tool')
    entry = ROOT / (name + '.py')
elif command in ('record', 'panel'):
    entry = ROOT / ('record.py' if command == 'record' else 'control_panel.py')
else:
    raise SystemExit('Unknown command: ' + command)
sys.argv[0] = str(entry)
runpy.run_path(str(entry), run_name='__main__')
