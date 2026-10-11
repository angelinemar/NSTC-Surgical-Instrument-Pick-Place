"""Shared output location for object-detection recordings on this workstation."""
import os
from pathlib import Path


def object_datasets_root():
    override = os.environ.get('P4_OBJECT_DATASETS')
    if override:
        return Path(override).expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    workspace = repo.parent.parent if repo.parent.name == 'versions' else repo
    return workspace / 'datasets' / 'Object datasets'
