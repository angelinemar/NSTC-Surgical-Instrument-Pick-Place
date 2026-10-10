"""Find the simulator Python without assuming a Windows checkout location."""
import os
from pathlib import Path
import sys


def isaac_python():
    override = os.environ.get('P4_ISAAC_PYTHON')
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'P4_ISAAC_PYTHON does not exist: {path}')
        return str(path)
    if os.name == 'nt':
        legacy = Path('C:/IsaacLab/_isaac_sim/python.bat')
        if legacy.is_file():
            return str(legacy)
    return sys.executable
