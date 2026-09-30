"""Stream native-size PNG previews from saved H5 recordings; never edit the source."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

COLORS = {'background': (20,20,20), 'robot': (60,150,255),
          'surgical_tray': (255,150,40), 'scalpel': (80,220,100),
          'scissor': (245,220,60), 'love_retractor': (225,80,210),
          'kelly': (80,225,220), 'scalpel_type2': (255,90,90),
          'table': (45,150,75), 'floor': (110,110,120), 'room': (180,170,155)}


def color_semantic(mask, classes):
    result = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    for name, ident in classes.items():
        result[mask == int(ident)] = COLORS.get(name, (255,255,255))
    return result


def streams(obs):
    """Resolve legacy aliases once, without exporting duplicate camera images."""
    for camera in ('front', 'wrist', 'cam_top', 'cam_left', 'cam_right', 'cam_tray'):
        for kind in ('rgb', 'depth', 'semantic'):
            keys = [f'{camera}_{kind}']
            if camera == 'front':
                keys += {'rgb': ['images_front', 'images'], 'depth': ['depth_front'],
                         'semantic': ['segmentation_map']}[kind]
            if camera == 'wrist':
                keys += {'rgb': ['grip_b_rgb', 'images_wrist', 'images_grip_b'],
                         'depth': ['depth_wrist', 'grip_b_depth', 'depth_grip_b'],
                         'semantic': ['grip_b_semantic']}[kind]
            for key in keys:
                if key in obs and isinstance(obs[key], h5py.Dataset):
                    yield camera, kind, key
                    break


def png_frame(frame, kind):
    frame = np.asarray(frame)
    if kind == 'rgb':
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            raise ValueError(f'Unsupported RGB shape/dtype: {frame.shape}, {frame.dtype}')
        return frame[..., :3]
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]
    if frame.ndim != 2:
        raise ValueError(f'Expected 2D {kind}, got {frame.shape}')
    if kind == 'semantic':
        if not np.issubdtype(frame.dtype, np.integer) or frame.min() < 0 or frame.max() > 65535:
            raise ValueError('Semantic IDs must be integers in the PNG uint16 range')
        return frame.astype(np.uint16)
    # Fixed 0..2 m display scale, not metric-depth training data.
    valid = np.isfinite(frame) & (frame > 0)
    result = np.zeros(frame.shape, dtype=np.uint8)
    result[valid] = (np.clip(frame[valid], 0, 2) / 2 * 255).astype(np.uint8)
    return result


def export(source, output, stride=1, selected=None, frame_mode='all', colored=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise ValueError('Select a folder containing saved H5 recordings')
    if stride < 1:
        raise ValueError('Frame interval must be at least 1')
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Export destination must be separate from the source dataset')
    if frame_mode not in ('all', 'first', 'middle', 'last'):
        raise ValueError('Invalid frame selection')
    files = sorted(source.rglob('*.h5')) if selected is None else sorted(set((source/p).resolve() for p in selected))
    if any(source not in p.resolve().parents or p.suffix.lower() != '.h5' or not p.is_file() for p in files):
        raise ValueError('Selected H5 must exist inside the source folder')
    if not files:
        raise ValueError('No H5 recordings found in this folder')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(source=str(source), stride=stride, frame_mode=frame_mode, colored=colored, png_count=0, files=[], skipped=[],
                  notes='RGB: native pixels; semantic: uint16 class IDs; depth: 0..2m display only. No GIFs or videos.')
    for path in files:
        before_count = report['png_count']
        relative = path.relative_to(source)
        with h5py.File(path, 'r') as h:
            if 'observations' not in h:
                report['skipped'].append(str(relative))
                continue
            if 'complete' in h.attrs and not bool(h.attrs['complete']):
                report['skipped'].append(str(relative))
                continue
            row = dict(file=str(relative), streams={})
            classes = json.loads(h.attrs.get('semantic_class_ids', '{}'))
            if colored and not classes:
                raise ValueError(f'{relative}: semantic_class_ids metadata missing; refusing guessed colors')
            for camera, kind, key in streams(h['observations']):
                dataset = h['observations'][key]
                folder = output / relative.with_suffix('') / f'{camera}_{kind}'
                folder.mkdir(parents=True, exist_ok=True)
                count = 0
                indices = range(0, len(dataset), stride) if frame_mode == 'all' else (
                    [] if not len(dataset) else [{'first':0, 'middle':len(dataset)//2, 'last':len(dataset)-1}[frame_mode]])
                for index in indices:
                    pixels = png_frame(dataset[index], kind)
                    Image.fromarray(pixels).save(folder / f'frame_{index:06d}.png')
                    count += 1
                    if kind == 'semantic' and colored:
                        color_folder = folder.with_name(f'{camera}_semantic_color')
                        color_folder.mkdir(exist_ok=True)
                        Image.fromarray(color_semantic(pixels, classes)).save(color_folder / f'frame_{index:06d}.png')
                        report['png_count'] += 1
                row['streams'][key] = count
                report['png_count'] += count
            report['files'].append(row)
            if colored and row['streams']:
                legend = {name: {'id': int(ident), 'rgb': COLORS.get(name, (255,255,255))} for name, ident in classes.items()}
                (output / relative.with_suffix('') / 'semantic_legend.json').write_text(json.dumps(legend, indent=2), encoding='utf-8')
        print(f'Exported {relative}: {report["png_count"] - before_count} PNGs', flush=True)
    if not report['png_count']:
        raise ValueError('No supported camera frames found; no completion marker written')
    (output / 'export_complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--files', nargs='+', help='Selected H5 paths relative to source')
    parser.add_argument('--frame-mode', choices=('all','first','middle','last'), default='all')
    parser.add_argument('--color-semantic', action='store_true')
    args = parser.parse_args()
    export(args.source, args.output, args.stride, args.files, args.frame_mode, args.color_semantic)
