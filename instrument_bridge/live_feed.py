"""Atomic RGB feed for a detector GUI; callable from Isaac Sim or any camera producer."""
import json
import os
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image


CAMERA_NAME = re.compile(r'^[a-zA-Z][a-zA-Z0-9_-]{0,63}$')


def publish_frame(folder, camera, rgb, *, scene_id=None, instance_mask=None,
                  instance_classes=None, spawn_on_table=None):
    """Publish one camera frame without exposing a partly written image to readers."""
    if not CAMERA_NAME.fullmatch(camera):
        raise ValueError('Invalid camera name')
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Expected H×W×3 uint8 RGB')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.time_ns()
    final = folder / f'{camera}_{stamp}_rgb.png'
    temporary = folder / f'.{camera}_{os.getpid()}_{time.time_ns()}.png'
    Image.fromarray(rgb).save(temporary, format='PNG')
    os.replace(temporary, final)
    instance_name = None
    if instance_mask is not None:
        instance_mask = np.asarray(instance_mask)
        if instance_mask.shape != rgb.shape[:2] or np.any(instance_mask < 0) or np.max(instance_mask) > 65535:
            raise ValueError('Instance mask must be H×W with IDs in [0, 65535]')
        if not isinstance(instance_classes, dict):
            raise ValueError('instance_classes is required with instance_mask')
        instance_name = f'{camera}_{stamp}_instance.png'
        instance_tmp = folder / f'.{instance_name}'
        Image.fromarray(instance_mask.astype(np.uint16)).save(instance_tmp, format='PNG')
        os.replace(instance_tmp, folder / instance_name)
    meta = dict(camera=camera, scene_id=scene_id, frame_id_ns=stamp, captured_at_unix=stamp / 1e9,
                width=rgb.shape[1], height=rgb.shape[0], image=final.name,
                instance_mask=instance_name, instance_classes=instance_classes,
                spawn_on_table=spawn_on_table)
    tmp_meta = folder / f'.{camera}_latest_{os.getpid()}.json'
    tmp_meta.write_text(json.dumps(meta))
    os.replace(tmp_meta, folder / f'{camera}_latest.json')
    # Keep several immutable frames so a reader finishing inference does not
    # lose the file named by the manifest while the next frame is published.
    old = sorted(folder.glob(f'{camera}_*_rgb.png'))[:-10]
    for path in old:
        path.unlink(missing_ok=True)
        (folder / path.name.replace('_rgb.png', '_instance.png')).unlink(missing_ok=True)
    return final
