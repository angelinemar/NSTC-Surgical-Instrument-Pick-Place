"""Link published target-relative DP checkpoints into the NSTC workspace.

Existing files are never replaced. The raw training episodes are not included.
"""

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path(os.environ.get('NSTC_WORKSPACE_ROOT', Path.home() / 'NSTC')))
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    repository = Path(__file__).resolve().parents[2]
    source = repository / 'models/dp/target_relative'
    if not (workspace / 'datasets').is_dir():
        parser.error(f'Workspace has no datasets directory: {workspace}')
    for run in sorted(source.iterdir()):
        if not run.is_dir():
            continue
        target = workspace / 'datasets/DP_training/runs' / run.name
        target.mkdir(parents=True, exist_ok=True)
        for original in sorted(run.iterdir()):
            link = target / original.name
            if link.exists() or link.is_symlink():
                if link.resolve() != original.resolve():
                    print(f'KEEP existing {link}')
                continue
            link.symlink_to(original.resolve())
            print(f'LINK {link} -> {original}')


if __name__ == '__main__':
    main()
