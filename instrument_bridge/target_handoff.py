"""Target selection and action handoff. No simulator ground truth enters this contract."""
import json
import re
from pathlib import Path

from .api import CLASSES

ALIASES = {
    'scalpel': 'scalpel',
    'scissors': 'scissor',
    'scissor': 'scissor',
    'love retractor': 'love_retractor',
    'love_retractor': 'love_retractor',
    'kelly': 'kelly',
    'scalpel type 2': 'scalpel_type2',
    'scalpel type2': 'scalpel_type2',
    'scalpel_type2': 'scalpel_type2',
}


def requested_class(instruction):
    """Recognize one of the five known class names; ambiguous text is rejected."""
    text = re.sub(r'[^a-z0-9_ ]+', ' ', instruction.lower())
    matches = set()
    # Longer aliases first so "scalpel type 2" does not also mean scalpel.
    for alias in sorted(ALIASES, key=len, reverse=True):
        if re.search(r'(?<![a-z0-9_])' + re.escape(alias) + r'(?![a-z0-9_])', text):
            matches.add(ALIASES[alias])
    if 'scalpel_type2' in matches and 'scalpel' in matches:
        matches.remove('scalpel')
    if len(matches) != 1:
        raise ValueError('Instruction must name exactly one of: ' + ', '.join(CLASSES))
    return matches.pop()


def matching_indices(objects, class_name):
    if class_name not in CLASSES:
        raise ValueError('Unknown class: ' + class_name)
    return [i for i, obj in enumerate(objects) if obj['class_name'] == class_name]


def pick_at_pixel(objects, x, y):
    """Return highest-score box containing a click, or None; no GT is consulted."""
    choices = [i for i, obj in enumerate(objects)
               if obj['bbox_xyxy'][0] <= x <= obj['bbox_xyxy'][2]
               and obj['bbox_xyxy'][1] <= y <= obj['bbox_xyxy'][3]]
    return max(choices, key=lambda i: objects[i]['score']) if choices else None


def build_request(report, camera, object_index, instruction, selected_by, image_path):
    rows = [row for row in report['per_camera'] if row['name'] == camera]
    if len(rows) != 1:
        raise ValueError('Camera is not part of the report')
    row = rows[0]
    objects = row['objects']
    if not 0 <= object_index < len(objects):
        raise ValueError('Selected instance is outside the camera predictions')
    obj = objects[object_index]
    class_name = requested_class(instruction)
    if obj['class_name'] != class_name:
        raise ValueError(f'Instruction requests {class_name}, but selected instance is {obj["class_name"]}')
    if selected_by not in ('click', 'list', 'language_unique'):
        raise ValueError('Invalid selection method')
    if not Path(image_path).is_file():
        raise FileNotFoundError(image_path)
    camera_observations = []
    for view in report['per_camera']:
        if report['source'] == 'live' and row.get('scene_id') is not None and view.get('scene_id') != row['scene_id']:
            continue
        view_image = Path(image_path).parent / view.get('rgb', Path(image_path).name)
        camera_observations.append(dict(camera=view['name'], image_path=str(view_image.resolve()),
                                        predicted_counts={name: sum(o['class_name'] == name for o in view['objects']) for name in CLASSES},
                                        instances_2d=[dict(class_name=o['class_name'], score=o['score'],
                                                           bbox_xyxy=o['bbox_xyxy'], center_xy=o['center_xy'])
                                                      for o in view['objects']]))
    return dict(contract='nstc_target_request_v1', task='pick', instruction=instruction,
                perception_model=report['model'], score_threshold=report['threshold'],
                source_type=report['source'], source_path=report['path'], source_frame=report['frame'],
                camera=camera, image_path=str(Path(image_path).resolve()),
                source_scene_id=row.get('scene_id'), captured_at_unix=row.get('captured_at_unix'),
                selected_by=selected_by, instance_index=object_index,
                target=dict(class_name=class_name, score=obj['score'],
                            bbox_xyxy=obj['bbox_xyxy'], center_xy=obj['center_xy'],
                            mask_rle=obj['mask_rle'], track_id=obj.get('track_id')),
                camera_observations=camera_observations,
                coordinate_frame='camera_pixels',
                pose_3d=None,
                note='Detector output only. No ground-truth spawn pose or 3D robot target is supplied.')


def save_request(path, request):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request, indent=2))
    return path
