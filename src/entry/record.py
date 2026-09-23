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
    parser.add_argument('--session-config',help='Control-panel session JSON')
    parser.add_argument('--tray-occupancy',choices=['random','empty','full'],default='random',
                        help='Initial tray holds 0-4 other classes; target slot is always empty')
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="Stop with nonzero exit after this many attempts; 0 = unlimited")
    known, forwarded = parser.parse_known_args()
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
