"""One entry point for every P4 object recorder, with output-safety preflight."""

from __future__ import annotations

import argparse
import os

from object_handlers import HANDLERS, get_handler
from runner import run_handler


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--object", choices=tuple(HANDLERS))
    parser.add_argument("--list-objects", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument('--camera-size', type=int, choices=(224, 448), default=448,
                        help='Recorded square RGB size. Native sensor, wider table view, no crop. 448 retains more detail.')
    parser.add_argument('--randomization', choices=('off', 'train'), default='train')
    parser.add_argument('--randomization-seed', type=int, default=17)
    parser.add_argument('--distractor-min', type=int, default=12)
    parser.add_argument('--distractor-max', type=int, default=18)
    parser.add_argument('--scene-audit-only', action='store_true',help='Settle and save six-camera label preview, then exit without recording demonstrations.')
    parser.add_argument('--dataset-purpose', choices=('detection','dp','both'), default='detection')
    parser.add_argument('--dataset-split', choices=('train','valid','test','unassigned'), default='unassigned')
    parser.add_argument('--shutdown-mode', choices=('native','verified-exit'),
                        default='verified-exit' if os.name=='nt' else 'native')
    parser.add_argument('--session-config',help='Control-panel session JSON')
    parser.add_argument('--tray-occupancy',choices=['random','empty','full','manual'],default='random',
                        help='Initial tray occupancy. Random holds 0-3 other classes so at least one non-target remains on the table; full holds all four. Target slot is always empty.')
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="Stop with nonzero exit after this many attempts; 0 = unlimited")
    known, forwarded = parser.parse_known_args()
    os.environ['P4_CAMERA_SIZE'] = str(known.camera_size)
    if not 4 <= known.distractor_min <= known.distractor_max <= 30:
        parser.error('Distractor range must satisfy 4 <= min <= max <= 30')
    os.environ['P4_DISTRACTOR_MIN'] = str(known.distractor_min)
    os.environ['P4_DISTRACTOR_MAX'] = str(known.distractor_max)
    os.environ['P4_SCENE_AUDIT_ONLY'] = '1' if known.scene_audit_only else '0'
    os.environ['P4_RANDOMIZATION'] = known.randomization
    os.environ['P4_RANDOMIZATION_SEED'] = str(known.randomization_seed)
    os.environ['P4_DATASET_PURPOSE'] = known.dataset_purpose
    os.environ['P4_DATASET_SPLIT'] = known.dataset_split
    os.environ['P4_SHUTDOWN_MODE'] = known.shutdown_mode
    if any(a.startswith('--center_crop_size') for a in forwarded):
        parser.error('P4 uses native square images; --center_crop_size is no longer supported')
    forwarded.extend(['--center_crop_size', '0'])
    os.environ['P4_TRAY_OCCUPANCY']=known.tray_occupancy
    if known.session_config:
        from pathlib import Path
        os.environ['P4_SESSION_CONFIG'] = str(Path(known.session_config).resolve())
    if known.max_attempts is not None:
        if known.max_attempts < 0:
            parser.error('--max-attempts must be nonnegative')
        os.environ['P4_MAX_ATTEMPTS'] = str(known.max_attempts)
    if known.list_objects:
        for name, handler in HANDLERS.items():
            print(f"{name:16s} id={handler.object_type_id} backend={handler.backend}")
        return
    if not known.object:
        parser.error("--object is required unless --list-objects is used")
    # A recorder always needs actual sensor rendering, including headless.
    if '--enable_cameras' not in forwarded:
        forwarded.append('--enable_cameras')
    if known.dry_run:
        from runner import _validate_files
        handler = get_handler(known.object)
        backend = _validate_files(handler)
        print("DRY_RUN_OK object=", handler.name)
        print("backend=", backend)
        return
    # Validate before run_handler writes a new scene manifest into this folder.
    from pathlib import Path
    from phase4_feedback import completed_episode_count
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument('--out_dir',default='datasets/phase3_grid_split')
    output_parser.add_argument('--resume',action='store_true')
    output_parser.add_argument('--record_mode',default='both',choices=['both','pick','place'])
    output, _ = output_parser.parse_known_args(forwarded)
    existing = list(Path(output.out_dir).glob('*_policy/*/episode_*.h5'))
    if existing and not output.resume:
        parser.error('Output already contains episodes; use --resume or a fresh --out_dir')
    if output.resume:
        completed_episode_count(output.out_dir,known.object,output.record_mode)
    run_handler(get_handler(known.object), forwarded)


if __name__ == "__main__":
    main()
