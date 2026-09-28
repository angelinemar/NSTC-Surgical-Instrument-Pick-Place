"""Keep legacy module identities and resource roots while sources are organized."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def execute(namespace, relative):
    implementation = (ROOT / relative).resolve()
    if not implementation.is_relative_to(ROOT):
        raise ValueError('Implementation outside P4')
    namespace['__implementation__'] = str(implementation)
    # __file__ intentionally remains the legacy root path: legacy resource lookups
    # and monkeypatches keep their module identity. Tracebacks show the real source.
    exec(compile(implementation.read_text(encoding='utf-8-sig'), str(implementation), 'exec'), namespace)
