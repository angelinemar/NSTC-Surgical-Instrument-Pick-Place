"""Load organized sources behind the legacy flat module names."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def execute(namespace, relative):
    implementation = (ROOT / relative).resolve()
    if not implementation.is_relative_to(ROOT):
        raise ValueError('Implementation outside P4')
    namespace['__implementation__'] = str(implementation)
    # Preserve the former root-level resource base while keeping the physical
    # wrappers together in compat/. Canonical sources remain the traceback path.
    namespace['__file__'] = str(ROOT / Path(namespace['__file__']).name)
    sys.path.insert(0, str(ROOT / 'compat'))
    exec(compile(implementation.read_text(encoding='utf-8-sig'), str(implementation), 'exec'), namespace)
