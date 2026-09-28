"""Compatibility entry. Edit tests/legacy/test_panel_retry.py, not this shim."""
from _p4_compat import execute
execute(globals(), 'tests/legacy/test_panel_retry.py')
