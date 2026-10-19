"""Restore training resolution explicitly: RF-DETR export may omit it from args."""
import json
from pathlib import Path


def checkpoint_resolution(checkpoint,override=None):
    path=Path(checkpoint).parent/'trial_config.json'
    recorded=json.loads(path.read_text()).get('resolution') if path.exists() else None
    value=override if override is not None else recorded
    if value is None:raise ValueError('Missing training resolution; supply --resolution explicitly')
    value=int(value)
    if value<24 or value%24:raise ValueError('Segmentation resolution must be a positive multiple of 24')
    if recorded is not None and value!=int(recorded):raise ValueError('Evaluation resolution differs from trial_config.json')
    return value
