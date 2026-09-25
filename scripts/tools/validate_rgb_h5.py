"""Read-only geometry/data integrity check before using P4 output for training."""
import argparse
from pathlib import Path
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    files = sorted(args.folder.rglob('episode_*.h5'))
    if not files:
        raise SystemExit('FAIL: no episode H5 found')
    failed = []
    for path in files:
        issues = []
        with h5py.File(path, 'r') as f:
            count = f['actions'].shape[0]
            image_hw = tuple(f['observations/front_rgb'].shape[1:3])
            if image_hw not in ((224,224),(448,448)):
                issues.append('unsupported image dimensions')
            if f['actions'].shape != (count,8):
                issues.append('action shape')
            if not np.isfinite(f['actions'][:]).all():
                issues.append('nonfinite actions')
            for key in ('front_rgb','wrist_rgb','cam_top_rgb','cam_left_rgb','cam_right_rgb','cam_tray_rgb'):
                d = f['observations'][key]
                if d.shape != (count,*image_hw,3) or d.dtype != np.uint8:
                    issues.append(f'{key} shape/dtype={d.shape}/{d.dtype}')
                # Read in chunks: never silently accept an all-black episode.
                black = sum(int(np.count_nonzero(np.max(d[start:start+32], axis=(1,2,3)) == 0)) for start in range(0,count,32))
                if black:
                    issues.append(f'{key}: {black}/{count} all-black frames')
            for key in ('front_semantic','grip_b_semantic','cam_top_semantic','cam_left_semantic','cam_right_semantic','cam_tray_semantic'):
                d = f['observations'][key]
                if d.shape != (count,*image_hw) or d.dtype != np.uint16:
                    issues.append(f'{key} shape/dtype')
                if np.max(d[:]) > 7:
                    issues.append(f'{key} unknown semantic ID')
            for name, group in f['camera_calibration'].items():
                for key, data in group.items():
                    if isinstance(data, h5py.Dataset) and not np.isfinite(data[:]).all():
                        issues.append(f'nonfinite calibration: {name}/{key}')
        print(('FAIL' if issues else 'PASS'), path, f'T={count}', '; '.join(issues))
        failed.extend(issues)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
