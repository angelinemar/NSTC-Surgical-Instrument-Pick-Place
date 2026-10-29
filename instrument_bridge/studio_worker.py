"""Evaluate one recorded scene without importing the GUI into the model process."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_api

from .api import CLASSES, InstrumentDetector
from .cli import ground_truth
from .evaluate import compare, table_spawn_counts


DP_CAMERAS = ('front', 'wrist', 'cam_top', 'cam_left', 'cam_right', 'cam_tray')
COLORS = ('#e95756', '#6b63db', '#f2a23a', '#139c87', '#2986d8')


def synthetic_ground_truth(view):
    result = []
    for box in view['boxes']:
        segmentation = box['segmentation']
        if isinstance(segmentation['counts'], list):
            segmentation = mask_api.frPyObjects(segmentation, *segmentation['size'])
        mask = mask_api.decode(segmentation).astype(bool)
        result.append(dict(class_name=CLASSES[box['category_id'] - 1],
                           instance_id=box['instance_id'], mask=mask))
    return result


def draw_preview(rgb, objects, masks, path):
    image = Image.fromarray(rgb).convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    canvas = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    for obj, mask in zip(objects, masks):
        color = COLORS[CLASSES.index(obj['class_name'])]
        color_rgb = tuple(bytes.fromhex(color[1:]))
        canvas[mask] = (*color_rgb, 72)
    overlay = Image.fromarray(canvas, 'RGBA')
    image = Image.alpha_composite(image, overlay).convert('RGB')
    draw = ImageDraw.Draw(image)
    for obj in objects:
        color = COLORS[CLASSES.index(obj['class_name'])]
        draw.rectangle(obj['bbox_xyxy'], outline=color, width=3)
        x, y = obj['bbox_xyxy'][:2]
        draw.text((x, max(0, y - 18)), f"{obj['class_name']} {obj['score']:.2f}",
                  fill='white', stroke_width=2, stroke_fill='black')
    image.save(path)


def draw_ground_truth(rgb, gt, path):
    image = Image.fromarray(rgb).convert('RGBA')
    canvas = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    for row in gt:
        color = COLORS[CLASSES.index(row['class_name'])]
        canvas[row['mask']] = (*bytes.fromhex(color[1:]), 95)
    image = Image.alpha_composite(image, Image.fromarray(canvas, 'RGBA')).convert('RGB')
    draw = ImageDraw.Draw(image)
    for row in gt:
        ys, xs = np.where(row['mask'])
        if len(xs):
            draw.text((int(xs.min()), max(0, int(ys.min()) - 18)), row['class_name'],
                      fill='white', stroke_width=2, stroke_fill='black')
    image.save(path)


def evaluate(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    detector = InstrumentDetector(args.model, args.threshold)
    views = []
    if args.source == 'synthetic':
        scene = json.loads((args.path / 'scene.json').read_text())
        spawn = {name: sum(obj['object'] == name and obj['region'] == 'table'
                           for obj in scene['objects']) for name in CLASSES}
        for view in scene['views']:
            image_path = args.path / f"{view['name']}_rgb.png"
            views.append((view['name'], np.asarray(Image.open(image_path).convert('RGB')),
                          synthetic_ground_truth(view)))
        lighting = scene.get('lighting')
    else:
        with h5py.File(args.path, 'r') as h:
            if not 0 <= args.frame < len(h['actions']):
                raise ValueError('Frame is outside this episode')
            spawn = table_spawn_counts(h.attrs, include_target=h.attrs['policy_skill'] == 'pick')
            for name in DP_CAMERAS:
                views.append((name, np.asarray(h[f'observations/{name}_rgb'][args.frame]),
                              ground_truth(h, name, args.frame)))
        lighting = None
    per_camera = []
    for name, rgb, gt in views:
        prediction, masks = detector.predict(rgb)
        metrics = compare(prediction['objects'], masks, gt)
        image_name = f'{name}_prediction.png'
        gt_image_name = f'{name}_ground_truth.png'
        draw_preview(rgb, prediction['objects'], masks, output / image_name)
        draw_ground_truth(rgb, gt, output / gt_image_name)
        visible = {label: sum(row['class_name'] == label for row in gt) for label in CLASSES}
        per_camera.append(dict(name=name, preview=image_name, ground_truth_preview=gt_image_name,
                               predicted=prediction['counts'],
                               visible_ground_truth=visible, metrics=metrics,
                               objects=prediction['objects']))
    summary = dict(contract='instrument_studio_v1', source=args.source, path=str(args.path),
                   model=args.model, threshold=args.threshold, frame=args.frame if args.source == 'dp' else None,
                   checkpoint=str(detector.checkpoint), spawn_on_table=spawn,
                   lighting=lighting, per_camera=per_camera,
                   note='Each camera is scored independently at mask IoU 0.50. Spawn counts include hidden objects. DP labels may be incomplete. This one-scene diagnostic is not mAP or whole-table accuracy.')
    (output / 'report.json').write_text(json.dumps(summary, indent=2))
    print(output / 'report.json', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', choices=('synthetic', 'dp'), required=True)
    p.add_argument('--path', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', choices=('A', 'B'), default='B')
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--frame', type=int, default=0)
    args = p.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
