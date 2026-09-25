"""Pure capture planning: one raw source, independent downstream consumers."""
from pathlib import Path
import shutil

PURPOSES = {'detection': ['detection'], 'dp': ['dp_joint'],
            'both': ['detection', 'dp_joint']}
LABELS = {
    'Both: DP joint + standalone detector (recommended)': 'both',
    'Detection only: standalone object detector': 'detection',
    'DP + detection: joint perception inside DP': 'dp',
}


def capture_contract(purpose, skill, size, seed, split='unassigned'):
    if purpose not in PURPOSES or skill not in ('pick', 'place', 'both'):
        raise ValueError('Unknown dataset purpose or saved skill')
    if size not in (224, 448) or split not in ('train', 'valid', 'test', 'unassigned'):
        raise ValueError('Invalid camera size or session split')
    return dict(contract='shared_raw_consumers_v1', purpose=purpose,
                consumers=PURPOSES[purpose], saved_skill=skill, camera_size=size,
                session_seed=int(seed), split=split, duplicate_raw=False,
                production_ready=False,
                note='Consumers are export targets, not trained models. Keep this session in one split.')


def storage_budget(destination, episodes, size, reserve_gb=30):
    if episodes < 1 or reserve_gb < 10 or size not in (224, 448):
        raise ValueError('Positive episode goal and >=10 GB reserve required')
    existing = Path(destination).resolve()
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    # Conservatively budget a full pair even when saving only one skill.
    estimated = int(episodes * (.9 if size == 448 else .25) * 1e9)
    reserve = int(reserve_gb * 1e9)
    return dict(estimated_bytes=estimated, reserve_bytes=reserve, free_bytes=free,
                capacity_pass=estimated + reserve <= free)
