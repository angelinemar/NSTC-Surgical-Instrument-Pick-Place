"""Compatibility entry. Edit tests/legacy/test_phase4_fsm.py, not this shim."""
from _p4_compat import execute
execute(globals(), 'tests/legacy/test_phase4_fsm.py')
