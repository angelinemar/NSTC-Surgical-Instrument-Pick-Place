"""Export independent full-resolution RGB, semantic PNG and COCO visible boxes.

Split by independently collected run/session, never adjacent frames or cameras.
New recordings include per-body instance masks for duplicate instruments.
Legacy recordings use one instance per class. Bounding boxes
enclose ALL visible instance pixels (including disconnected occluded parts).
These are visible boxes, not amodal boxes or general instance segmentation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase4_storage import require_commit
from phase4_feedback import rgb_has_spatial_detail

CAMERAS = ('front', 'wrist', 'cam_top', 'cam_left', 'cam_right', 'cam_tray')
CLASSES = ('scalpel', 'scissor', 'love_retractor', 'kelly', 'scalpel_type2')


def visible_boxes(mask, min_pixels=8, instances=None, instance_classes=None):
    if mask.ndim != 2 or not np.isin(mask, np.arange(11)).all():
        raise ValueError('Invalid semantic map')
    if instances is not None:
        if instances.shape != mask.shape:
            raise ValueError('Instance/semantic shape mismatch')
        result = []
        for value in np.unique(instances):
            if value == 0:
                continue
            item = instance_classes[str(int(value))]
            semantic_id = int(item['semantic_id'])
            pixels = instances == value
            if not np.all(mask[pixels] == semantic_id):
                raise ValueError('Instance class disagrees with semantic pixels')
            y, x = np.where(pixels)
            if len(x) >= min_pixels:
                result.append(dict(category_id=semantic_id-2, instance_id=int(value),
                    bbox=[int(x.min()),int(y.min()),int(x.max()-x.min()+1),int(y.max()-y.min()+1)],area=int(len(x)),iscrowd=0))
        return result
    result = []
    for category, name in enumerate(CLASSES, 1):
        y, x = np.where(mask == category + 2)
        if len(x) < min_pixels:
            continue
        result.append(dict(category_id=category, bbox=[int(x.min()), int(y.min()),
                           int(x.max()-x.min()+1), int(y.max()-y.min()+1)],
                           area=int(len(x)), iscrowd=0))
    return result


def export(splits, output, stride=20):
    if stride < 1:
        raise ValueError('Stride must be positive')
    roots = [Path(p).resolve() for runs in splits.values() for p in runs]
    if len(set(roots)) != len(roots) or any(a != b and a in b.parents for a in roots for b in roots):
        raise ValueError('Runs must be distinct and non-overlapping across splits')
    inventory = {}
    for split, runs in splits.items():
        files = [p for root in runs for p in sorted(Path(root).glob('*_policy/*/episode_*.h5'))]
        if not files:
            raise ValueError('No committed episode candidates in ' + split)
        inventory[split] = files
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(contract='p4_detection_v1', export_complete=False, production_ready=False,
                    stride=stride, box_convention='visible_xywh_per_rigid_instance',
                    categories=[dict(id=i+1, name=n) for i,n in enumerate(CLASSES)],
                    sources=[], splits={}, split_rule='independent_run_no_frame_split')
    seen_transactions, seen_files, seen_images, seen_seeds = {}, {}, {}, {}
    image_id, annotation_id = 0, 0
    for split, files in inventory.items():
        folder = output / split
        (folder / 'images').mkdir(parents=True)
        (folder / 'masks').mkdir()
        (folder / 'instances').mkdir()
        coco = dict(images=[], annotations=[], categories=manifest['categories'])
        for path in files:
            with path.open('rb') as stream:
                sha = hashlib.file_digest(stream, 'sha256').hexdigest()
            if sha in seen_files:
                raise ValueError('Duplicate H5 content: ' + str(path))
            seen_files[sha] = split
            with h5py.File(path, 'r') as h:
                assigned = h.attrs.get('dataset_split','unassigned')
                if assigned not in ('unassigned',split):
                    raise ValueError('Recorded session split disagrees with export split')
                if h.attrs.get('storage_contract') != 'journaled_episode_v2':
                    raise ValueError('Detector export requires checksum-committed raw episodes')
                require_commit(path, h)
                expected_classes = dict(background=0, robot=1, surgical_tray=2,
                                        **{n:i+3 for i,n in enumerate(CLASSES)})
                actual_classes = json.loads(h.attrs.get('semantic_class_ids', '{}'))
                if actual_classes not in (expected_classes, dict(expected_classes,table=8,floor=9,room=10)):
                    raise ValueError('Unexpected class mapping')
                clutter = json.loads(h.attrs.get('scene_clutter','{}'))
                if clutter.get('instances') and h.attrs.get('instance_contract') != 'rigid_body_visible_masks_v1':
                    raise ValueError('Duplicate objects require recorded instance masks')
                instance_classes = json.loads(h.attrs.get('instance_classes','{}'))
                tx = str(h.attrs['pair_transaction_id'])
                if seen_transactions.setdefault(tx, split) != split:
                    raise ValueError('Pick/place transaction crosses splits')
                metadata = dict(source=str(path.resolve()), sha256=sha, split=split,
                                transaction=tx, target=str(h.attrs['target_object']),
                                has_duplicate_instances=bool(clutter.get('instances')),
                                randomization=json.loads(h.attrs.get('domain_randomization', '{}')))
                manifest['sources'].append(metadata)
                seed = metadata['randomization'].get('session_seed')
                if seed is not None and seen_seeds.setdefault(seed,split) != split:
                    raise ValueError('Randomization session seed crosses dataset splits')
                for camera in CAMERAS:
                    rgb = h['observations/' + camera + '_rgb']
                    sem = h['observations/' + ('grip_b' if camera == 'wrist' else camera) + '_semantic']
                    if rgb.shape[:3] != sem.shape or len(rgb) != len(h['actions']):
                        raise ValueError('Camera/label/action alignment mismatch')
                    for t in range(0, len(rgb), stride):
                        frame, mask = rgb[t], sem[t]
                        if frame.dtype != np.uint8 or not rgb_has_spatial_detail(frame):
                            raise ValueError('Invalid or flat RGB: ' + str(path))
                        fingerprint = hashlib.sha256(frame.tobytes()).hexdigest()
                        previous = seen_images.get(fingerprint)
                        if previous and previous != split:
                            raise ValueError('Identical image leaks across splits')
                        if previous:
                            continue
                        seen_images[fingerprint] = split
                        image_id += 1
                        stem = f'{image_id:08d}'
                        Image.fromarray(frame).save(folder/'images'/f'{stem}.png')
                        Image.fromarray(mask.astype(np.uint8)).save(folder/'masks'/f'{stem}.png')
                        gray = frame.astype(np.float32).mean(2)
                        lap = (-4*gray[1:-1,1:-1]+gray[:-2,1:-1]+gray[2:,1:-1]
                               +gray[1:-1,:-2]+gray[1:-1,2:])
                        coco['images'].append(dict(id=image_id, file_name=f'images/{stem}.png',
                            mask_file=f'masks/{stem}.png', width=frame.shape[1], height=frame.shape[0],
                            source_sha256=sha, frame_index=t, camera=camera, transaction=tx,
                            image_quality=dict(laplacian_variance=float(lap.var()),
                                dark_fraction=float((gray<3).mean()), bright_fraction=float((gray>252).mean()))))
                        instance_key = 'observations/' + ('grip_b' if camera=='wrist' else camera) + '_instance'
                        instances = h[instance_key][t] if instance_key in h else None
                        if instances is not None:
                            Image.fromarray(instances.astype(np.uint16)).save(folder/'instances'/f'{stem}.png')
                            coco['images'][-1]['instance_mask_file'] = f'instances/{stem}.png'
                            coco['images'][-1]['instance_classes'] = instance_classes
                        for box in visible_boxes(mask, instances=instances, instance_classes=instance_classes):
                            annotation_id += 1
                            coco['annotations'].append(dict(id=annotation_id, image_id=image_id, **box))
        counts = {c['name']:sum(a['category_id']==c['id'] for a in coco['annotations']) for c in coco['categories']}
        manifest['splits'][split] = dict(images=len(coco['images']), visible_instances=counts)
        (folder/'annotations.json').write_text(json.dumps(coco, indent=2))
    manifest['export_complete'] = True
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ('train', 'valid', 'test'):
        parser.add_argument('--'+split+'-runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=20)
    args = parser.parse_args()
    report = export({s:getattr(args, s+'_runs') for s in ('train','valid','test')}, args.output, args.stride)
    print(json.dumps(report['splits'], indent=2))
