# P4 camera naming

The front pose is loaded from the latest `camera_layout.json`; renaming does
not reset the pose. New tuner saves use `cam_front`. Legacy files with `camera`
remain readable. Conflicting aliases fail explicitly instead of picking one.

| Layer | Front name |
|---|---|
| Preview panel / GIF title | cam_front / CAM_FRONT |
| USD prim | /World/envs/env_0/cam_front |
| New tuner JSON key | cam_front |
| Internal legacy IsaacLab sensor key | camera |
| H5 datasets | observations/front_rgb, front_depth, front_semantic |
| H5 calibration view | camera_calibration/front |

The internal key is a compatibility name, NOT a second camera. All five
recorders still read that same sensor. Existing H5 files/trainers/GIF conversion
stay compatible; saved schema and stage IDs are unchanged. New recordings use
the native square sensor size selected for the session (224 or 448), without a
post-capture center crop. The run's scene_manifest.json records the camera
mapping, poses, intrinsics, and image size explicitly.

Restart the open tuner after this code change. Old running Python cannot adopt
the new code and still autosaves the legacy JSON key. Do not edit active JSON
while that old tuner is running. On restart its latest front pose is loaded,
and the new tuner autosaves the canonical cam_front key.

Validation: `test_camera_names.py` checks all five backend paths, public/legacy
JSON roundtrip, no conflicting aliases, current poses and resolution. The final
task-focused layout was also rendered in Isaac Sim with all six RGB and semantic
streams before publication.
