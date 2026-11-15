"""Locate the NSTC workspace without depending on one checkout path."""

import os
from pathlib import Path


def workspace_root() -> Path:
    override = os.environ.get('NSTC_WORKSPACE_ROOT')
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / 'datasets').is_dir():
            raise RuntimeError(f'NSTC_WORKSPACE_ROOT has no datasets directory: {root}')
        return root
    for parent in Path(__file__).resolve().parents:
        if (parent / 'datasets').is_dir() and (parent / 'versions/6.0').is_dir():
            return parent
    raise RuntimeError('Set NSTC_WORKSPACE_ROOT to the directory containing datasets and versions/6.0')
