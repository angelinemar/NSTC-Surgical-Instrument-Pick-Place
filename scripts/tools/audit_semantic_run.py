"""Read-only full-frame H5 audit, with regenerable six-camera previews."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CLASSES = dict(background=0, robot=1, surgical_tray=2, scalpel=3,
               scissor=4, love_retractor=5, kelly=6, scalpel_type2=7)
COLORS = np.array([(0, 0, 0), (60, 150, 255), (255, 150, 40),
                   (80, 220, 100), (245, 220, 60), (225, 80, 210),
                   (80, 225, 220), (255, 90, 90)], dtype=np.uint8)
CAMERAS = ('front', 'wrist', 'cam_top', 'cam_left', 'cam_right', 'cam_tray')


def audit_file(path_string, output_string, previous=None):
    from validate_feedback_dataset import audit
    from phase4_storage import require_commit

    path, output = Path(path_string), Path(output_string)
    before = path.stat()
    reusable = (previous is not None and previous.get('file') == str(path)
                and previous.get('size_bytes') == before.st_size
                and previous.get('mtime_ns') == before.st_mtime_ns)
    if reusable and previous.get('result') == 'PASS' and Path(previous['regenerated_preview']).is_file():
        return previous
    report = dict(file=str(path), size_bytes=before.st_size,
                  mtime_ns=before.st_mtime_ns, errors=[])
    try:
        # Includes every RGB frame, actions, alignment and recorded grasp evidence.
        baseline = previous.get('baseline_audit') if reusable else None
        if not baseline or baseline.get('result') != 'PASS':
            baseline = audit(path)
        report['baseline_audit'] = baseline
        with h5py.File(path, 'r') as h:
            if not (reusable and previous.get('commit_sha256_verified')):
                require_commit(path, h)
            report['storage_contract'] = str(h.attrs.get('storage_contract', 'legacy'))
            assert report['storage_contract'] == 'journaled_episode_v2', 'Missing SHA256 commit contract'
            report['commit_sha256_verified'] = True
            assert json.loads(h.attrs['semantic_class_ids']) == CLASSES, 'Semantic class mapping'
            target = str(h.attrs['target_object'])
            target_id = CLASSES[target]
            skill = str(h.attrs['policy_skill'])
            conditioning_class = target if skill == 'pick' else 'surgical_tray'
            conditioning_id = CLASSES[conditioning_class]
            assert h['supervision_gt'].attrs['conditioning_semantic_id'] == conditioning_id, 'Conditioning ID differs from phase contract'
            assert h['supervision_gt'].attrs['conditioning_class'] == conditioning_class, 'Conditioning class differs from phase contract'
            n = len(h['actions'])
            image_hw = tuple(h['observations/front_rgb'].shape[1:3])
            assert image_hw in ((224,224),(448,448))
            report.update(frames=n, target=target, skill=str(h.attrs['policy_skill']),
                          cell=int(h.attrs['coverage_cell_id']),
                          transaction_id=str(h.attrs['pair_transaction_id']), cameras={})
            for camera in CAMERAS:
                semantic_name = 'grip_b' if camera == 'wrist' else camera
                mask = h[f'observations/{semantic_name}_semantic']
                depth = h[f'observations/{camera}_depth']
                supervision = h[f'supervision_gt/conditioning_mask_{semantic_name}']
                assert mask.shape == depth.shape == supervision.shape == (n, *image_hw)
                assert mask.dtype == np.uint16 and supervision.dtype == np.uint8
                counts = np.zeros(8, dtype=np.int64)
                visible_frames = np.zeros(8, dtype=np.int64)
                for start in range(0, n, 32):
                    values = mask[start:start+32]
                    assert values.max() <= 7, f'{camera}: unknown class ID'
                    counts += np.bincount(values.ravel(), minlength=8)
                    for class_id in range(8):
                        visible_frames[class_id] += np.any(values == class_id, axis=(1, 2)).sum()
                    assert np.array_equal(supervision[start:start+32], values == conditioning_id), f'{camera}: supervision mask differs'
                    distances = depth[start:start+32]
                    assert np.isfinite(distances).all() and (distances >= 0).all(), f'{camera}: invalid depth'
                calibration = h[f'camera_calibration/{semantic_name}']
                for key, values in calibration.items():
                    assert len(values) == n and np.isfinite(values[:]).all(), f'{camera}: invalid {key}'
                report['cameras'][camera] = dict(ids=np.flatnonzero(counts).tolist(),
                    pixel_counts=counts.tolist(), frames_with_class=visible_frames.tolist())
            union = sorted(set().union(*(v['ids'] for v in report['cameras'].values())))
            assert {1, 2, target_id} <= set(union), 'Robot, tray or target absent from entire segment'
            report['semantic_ids'] = union
            report['absent_classes'] = [label for label, cid in CLASSES.items() if cid not in union]
            report['initial_tray_occupancy'] = json.loads(h.attrs['initial_tray_occupancy'])
            # Regenerate directly from integer labels; previews are disposable outputs.
            canvas = Image.new('RGB', (672, 1060), (24, 24, 24))
            draw = ImageDraw.Draw(canvas)
            index = n // 2
            draw.text((8, 8), f'{path.stem} | {report["skill"]} | frame {index} | RGB + semantic', fill='white')
            for i, camera in enumerate(CAMERAS):
                semantic_name = 'grip_b' if camera == 'wrist' else camera
                x, y = (i % 3) * 224, 30 + (i // 3) * 490
                draw.text((x+5, y), camera, fill='white')
                canvas.paste(Image.fromarray(h[f'observations/{camera}_rgb'][index]).resize((224,224),Image.Resampling.LANCZOS), (x, y+20))
                canvas.paste(Image.fromarray(COLORS[h[f'observations/{semantic_name}_semantic'][index]]).resize((224,224),Image.Resampling.NEAREST), (x, y+248))
            for label, cid in CLASSES.items():
                x, y = (cid % 4) * 168, 1014 + (cid // 4) * 22
                draw.rectangle((x+4, y, x+16, y+12), fill=tuple(map(int, COLORS[cid])))
                draw.text((x+20, y), f'{cid}: {label}', fill='white')
            preview = output / 'regenerated_previews' / (report['skill'] + '_' + path.stem + '.png')
            preview.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(preview)
            report['regenerated_preview'] = str(preview)
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'H5 changed during audit'
    except Exception as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    report['result'] = 'FAIL' if report['errors'] else 'PASS'
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(run.glob('*_policy/scalpel/episode_*.h5'))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit('No H5 files')
    started = time.monotonic()
    results = []
    previous = {}
    results_path = output / 'files.jsonl'
    if args.resume and results_path.exists():
        for line in results_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                record = json.loads(line)
                previous[record['file']] = record
        results_path.replace(output / 'files_before_resume.jsonl')
        print(f'Resuming {len(previous)} existing records; only unchanged H5 checks are reused.', flush=True)
    with (output / 'files.jsonl').open('w', encoding='utf-8') as stream:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(audit_file, str(path), str(output), previous.get(str(path))): path for path in files}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                stream.write(json.dumps(result) + '\n')
                stream.flush()
                if len(results) % 10 == 0 or result['errors'] or len(results) == len(files):
                    print(f'AUDIT {len(results)}/{len(files)} failures={sum(bool(r["errors"]) for r in results)} elapsed={time.monotonic()-started:.1f}s', flush=True)
                    if result['errors']:
                        print(result['file'], result['errors'], flush=True)
    results.sort(key=lambda r: r['file'])
    pairs = {}
    for result in results:
        pairs.setdefault(Path(result['file']).stem, []).append(result)
    pair_errors = []
    counts = [0] * 10
    for episode, segments in pairs.items():
        if {r.get('skill') for r in segments} != {'pick', 'place'} or len(segments) != 2:
            pair_errors.append(episode + ': incomplete pair')
            continue
        if any(r['errors'] for r in segments):
            pair_errors.append(episode + ': segment audit failed')
            continue
        if len({r['transaction_id'] for r in segments}) != 1 or len({r['cell'] for r in segments}) != 1:
            pair_errors.append(episode + ': pair metadata differs')
        counts[segments[0]['cell']] += 1
    errors = [r for r in results if r['errors']]
    complete = not args.limit and len(files) == 1000 and len(pairs) == 500 and counts == [50]*10 and not errors and not pair_errors
    summary = dict(run=str(run), full_run_pass=complete, file_count=len(files),
        passed_files=len(files)-len(errors), failed_files=len(errors), pair_count=len(pairs),
        pair_errors=pair_errors, coverage=counts, total_frames=sum(r.get('frames', 0) for r in results),
        camera_frame_count=6*sum(r.get('frames', 0) for r in results),
        class_ids=CLASSES, files_without_all_eight_classes=[r['file'] for r in results if r.get('absent_classes')],
        h5_bytes=sum(r['size_bytes'] for r in results), elapsed_seconds=time.monotonic()-started,
        errors=[dict(file=r['file'], errors=r['errors']) for r in errors],
        scope='All RGB frames and semantic pixels, depths, target masks, calibration, actions, recorded physical evidence, SHA256 commits; not manual pixel annotation verification.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    if errors or (not args.limit and not complete):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
