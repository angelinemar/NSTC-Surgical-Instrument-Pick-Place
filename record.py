"""Compatibility entry. Edit src/entry/record.py, not this shim."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'compat'))
from _p4_compat import execute
execute(globals(), 'src/entry/record.py')
