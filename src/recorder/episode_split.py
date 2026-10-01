"""Online split assignment for committed episodes, grouped by target and grid."""
import json
from pathlib import Path

SPLITS = ('train', 'valid', 'test')
WEIGHTS = (0.7, 0.2, 0.1)
CONTRACT = 'episode_grid_balanced_v1'


def roots(root):
    root = Path(root)
    return [root] + [root / split for split in SPLITS]


def episode_files(root, skill, target='*'):
    files = [p for base in roots(root)
             for p in base.glob(f'{skill}_policy/{target}/episode_*.h5')]
    return sorted(files, key=lambda p: (p.name, str(p)))


def committed_assignments(root, target):
    result = []
    for split in SPLITS:
        for path in sorted((Path(root) / split / '.commits').glob(target + '_episode_*.json')):
            row = json.loads(path.read_text(encoding='utf-8'))
            if row.get('split_contract') != CONTRACT or row.get('dataset_split') != split:
                raise ValueError('Invalid split commit: ' + str(path))
            result.append(row)
    return result


def choose_split(history, cell, yaw=None):
    """No RNG or attempt counter: retries cannot consume split quotas.

    Each ten successes in a grid have capacity 7/2/1. Global deficits select
    among remaining capacities. Yaw-bin deficits break otherwise equal scores.
    Tiny collections prioritize train, validation, then test.
    """
    counts = {s: sum(r['dataset_split'] == s for r in history) for s in SPLITS}
    local = [r for r in history if r.get('split_cell_id', -1) == cell]
    blocks = len(local) // 10 + 1
    eligible = [s for s, quota in zip(SPLITS, (7, 2, 1))
                if sum(r['dataset_split'] == s for r in local) < blocks * quota]
    for s in SPLITS:
        if len(history) < 3 and counts[s] == 0 and s in eligible:
            return s
    yaw_bin = -1 if yaw is None else int(float(yaw) % 360 // 90)
    similar = [r for r in history if r.get('split_yaw_bin', -1) == yaw_bin]
    def score(s):
        weight = WEIGHTS[SPLITS.index(s)]
        return (round((len(history) + 1) * weight - counts[s], 8),
                (len(similar) + 1) * weight - sum(r['dataset_split'] == s for r in similar),
                -SPLITS.index(s))
    return max(eligible, key=score)


def assignment(root, target, index, meta):
    history = committed_assignments(root, target)
    if len(history) != index:
        raise ValueError('Auto split requires contiguous committed episode indices')
    cell = int(meta.get('coverage_cell_id', meta.get('grid_cell_id', -1)))
    poses = json.loads(meta.get('spawn_requested_all', '{}'))
    yaw = poses.get(target, {}).get('yaw_deg')
    split = choose_split(history, cell, yaw)
    randomization = json.loads(meta.get('domain_randomization', '{}'))
    return dict(dataset_split=split, split_contract=CONTRACT, split_cell_id=cell,
                split_randomization_attempt=int(randomization.get('attempt',meta.get('attempt',0))),
                split_yaw_bin=-1 if yaw is None else int(float(yaw) % 360 // 90))


def summary(root, target):
    rows = committed_assignments(root, target)
    return dict(contract=CONTRACT,
                counts={s:sum(r['dataset_split']==s for r in rows) for s in SPLITS},
                per_grid={str(cell):{s:sum(r['dataset_split']==s and r['split_cell_id']==cell
                                         for r in rows) for s in SPLITS}
                          for cell in sorted({r['split_cell_id'] for r in rows})})
