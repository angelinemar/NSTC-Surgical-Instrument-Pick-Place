"""Record per-body visible masks, merging mesh parts by their rigid asset root."""
import json
import numpy as np

CAMERAS = {'front':'camera', 'grip_b':'grip_cam_b', **{n:n for n in ('cam_top','cam_left','cam_right','cam_tray')}}


def body_mask(raw, labels, roots):
    result = np.zeros(raw.shape, np.uint16)
    for raw_id, value in labels.items():
        path = str(value.get('primPath', value.get('path',''))) if isinstance(value,dict) else str(value)
        for root, instance_id in roots.items():
            if path == root or path.startswith(root + '/'):
                result[raw == int(raw_id)] = instance_id
                break
    return result


def install(ns):
    recorder = ns['EpisodeRecorder']
    original_reset, original_add = recorder._reset, recorder.add_step
    def reset(self):
        original_reset(self)
        self.instance_masks = {name:[] for name in CAMERAS}
        self.instance_classes = {}
    def add(self, env, *args, **kwargs):
        before = len(self.actions)
        result = original_add(self, env, *args, **kwargs)
        if len(self.actions) == before:
            return result
        from phase4_tray_slots import ORDER
        from src.recorder.scene_clutter import pool
        roots = {}
        bodies = [(ns['phase3_scene_key_for_object'](n),n) for n in ORDER] + pool(ns['PHASE3_TARGET_OBJECT'])
        for index, (key,name) in enumerate(bodies, 1):
            root = env.scene[key].cfg.prim_path.replace('{ENV_REGEX_NS}', '/World/envs/env_0').replace('env_.*','env_0')
            roots[root] = index
            self.instance_classes[str(index)] = dict(name=name, semantic_id=ns['PHASE3_SEMANTIC_CLASS_IDS'][name], prim_path=root)
        for public,key in CAMERAS.items():
            camera = env.scene[key]
            raw = camera.data.output['instance_id_segmentation_fast'][0].detach().cpu().numpy().squeeze(-1)
            info = camera.data.info
            info = info[0] if isinstance(info,(list,tuple)) else info
            labels = info['instance_id_segmentation_fast']['idToLabels']
            mask = body_mask(raw, labels, roots)
            # Fail closed if an instrument's visible class pixels lack an instance.
            semantic = self.front_semantic[-1] if public=='front' else (self.grip_b_semantic[-1] if public=='grip_b' else self.extra_camera_semantic[public][-1])
            instrument = (semantic >= 3) & (semantic <= 7)
            if np.any(instrument & (mask == 0)):
                raise RuntimeError('Unmapped visible instrument instance: ' + public)
            self.instance_masks[public].append(mask)
        return result
    recorder._reset, recorder.add_step = reset, add


def append_segment(path, recorder, segment):
    if not hasattr(recorder, 'instance_masks'):
        return
    import h5py
    with h5py.File(path, 'r+') as h:
        # Select exactly the same subsequence as the backend writer.
        saved = [x.decode() if isinstance(x,bytes) else str(x) for x in h['stage_names'][:]] if 'stage_names' in h else None
        if saved is None:
            saved = [x.decode() if isinstance(x,bytes) else str(x) for x in h['observations/stage_names'][:]]
        indices = []
        cursor = 0
        for name in saved:
            while cursor < len(recorder.stage_names) and recorder.stage_names[cursor] != name:
                cursor += 1
            if cursor == len(recorder.stage_names):
                raise ValueError('Instance/frame stage alignment failed')
            indices.append(cursor); cursor += 1
        for camera, frames in recorder.instance_masks.items():
            if len(frames) != len(recorder.actions):
                raise ValueError('Missing instance frame: ' + camera)
            h['observations'].create_dataset(camera+'_instance',data=np.asarray(frames)[indices],compression='gzip')
        h.attrs['instance_classes'] = json.dumps(recorder.instance_classes,sort_keys=True)
        h.attrs['instance_contract'] = 'rigid_body_visible_masks_v1'
        h.attrs['semantic_mapping_version'] = 'p4_scene_v2'
        from src.recorder.scene_semantics import CLASSES
        mapping = json.loads(h.attrs.get('semantic_class_ids','{}'))
        mapping.update(CLASSES)
        h.attrs['semantic_class_ids'] = json.dumps(mapping,sort_keys=True)


def audit_instances(h):
    clutter = json.loads(h.attrs.get('scene_clutter','{}'))
    if h.attrs.get('instance_contract') != 'rigid_body_visible_masks_v1':
        if clutter.get('instances'):
            raise ValueError('Duplicate scene has no per-instance labels')
        return
    classes = json.loads(h.attrs['instance_classes'])
    table = np.zeros(max(map(int,classes))+1,np.uint16)
    for key,item in classes.items():
        table[int(key)] = int(item['semantic_id'])
    for camera in CAMERAS:
        masks = h['observations/'+camera+'_instance']
        semantic = h['observations/'+camera+'_semantic']
        if masks.shape != semantic.shape or masks.dtype != np.uint16:
            raise ValueError('Instance format/alignment mismatch: '+camera)
        for start in range(0,len(masks),32):
            instance, labels = masks[start:start+32], semantic[start:start+32]
            if int(instance.max()) >= len(table):
                raise ValueError('Unknown instance ID: '+camera)
            instrument = (labels>=3) & (labels<=7)
            if not np.array_equal(instance>0,instrument) or np.any(table[instance][instrument] != labels[instrument]):
                raise ValueError('Instance and semantic supervision disagree: '+camera)
