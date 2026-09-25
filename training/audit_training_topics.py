"""Read-only full-topic audit of a frozen inventory of P4 episode H5 files."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'record.py').is_file())
sys.path.insert(0, str(ROOT))
from phase4_feedback import rgb_has_spatial_detail

CAMERAS = [('front','front_rgb','front_depth','front_semantic'),
           ('grip_b','wrist_rgb','wrist_depth','grip_b_semantic')]
CAMERAS += [(n,n+'_rgb',n+'_depth',n+'_semantic') for n in ('cam_top','cam_left','cam_right','cam_tray')]
OBJECTS = ['scalpel','scissor','love_retractor','kelly','scalpel_type2']
ALIASES = dict(images_front='front_rgb', images_wrist='wrist_rgb', images_grip_b='wrist_rgb',
               depth_front='front_depth', depth_wrist='wrist_depth', depth_grip_b='wrist_depth',
               segmentation_map='front_semantic')


def plain(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def inspect(path):
    errors, warnings = [], []
    result = dict(file=str(path.resolve()), errors=errors, warnings=warnings,
                  cameras={}, topics={})
    def check(condition, text):
        if not condition:
            errors.append(text)
    with h5py.File(path, 'r') as h:
        attrs = ('dataset_schema','target_object','policy_skill','success','num_samples',
                 'p4_feedback_version','approach_controller','tcp_jacobian_contract',
                 'action_frame','action_type','action_keys','state_keys','state_frame',
                 'semantic_class_ids','object_type_ids','skill_type_ids','control_tcp_matches_observed_ee',
                 'control_tcp_offset_m','coverage_cell_id','coverage_cycle','coverage_contract',
                 'image_source_hw','image_output_hw','crop_top_left_hw','initial_tray_occupancy',
                 'created_at','grid_cell_id','target_yaw_deg')
        result['metadata'] = {k:plain(h.attrs[k]) for k in attrs if k in h.attrs}
        total = len(h['actions']) if 'actions' in h else 0
        image_hw = tuple(h['observations/front_rgb'].shape[1:3])
        check(image_hw in ((224, 224), (448, 448)), 'unsupported_image_size')
        result['steps'] = total
        def collect(name, node):
            if isinstance(node, h5py.Dataset):
                result['topics'][name] = dict(shape=list(node.shape), dtype=str(node.dtype))
        h.visititems(collect)
        for name, info in result['topics'].items():
            node = h[name]
            temporal = name.startswith(('observations/','camera_calibration/','debug_gt/')) or name in (
                'actions','dones','rewards','stage_names','stage_suffixes','step_ids') or 'conditioning_mask_' in name
            if temporal:
                check(bool(node.ndim) and len(node)==total, f'length_mismatch:{name}')
            if node.dtype.kind in 'fiu' and node.ndim < 3:
                values = node[:]
                nonfinite = int(np.count_nonzero(~np.isfinite(values)))
                info['nonfinite'] = nonfinite
                if nonfinite:
                    if name == 'supervision_gt/teacher_grasp_pose_b' and h.attrs.get('policy_skill')=='place':
                        warnings.append('place_teacher_grasp_pose_is_not_a_valid_label')
                    else:
                        errors.append(f'nonfinite:{name}:{nonfinite}')
        check(total > 0, 'empty_actions')
        if 'actions' in h:
            action = h['actions'][:]
            check(action.shape==(total,8), 'action_shape')
            if action.shape==(total,8):
                check(np.isfinite(action).all(), 'action_nonfinite')
                check(np.allclose(np.linalg.norm(action[:,3:7],axis=1),1,atol=.002), 'action_quaternion_norm')
                check(np.isin(action[:,-1],[-1,1]).all(), 'gripper_not_binary')
                result['quaternion_sign_flips'] = int(np.sum(np.sum(action[1:,3:7]*action[:-1,3:7],axis=1)<0))
        if 'observations/robot_proprio' in h and 'observations/state' in h:
            robot, state = h['observations/robot_proprio'][:], h['observations/state'][:]
            check(robot.shape==(total,16), 'robot_proprio_shape')
            check(state.shape==(total,18), 'state_shape')
            if state.shape==(total,18) and robot.shape==(total,16):
                check(np.array_equal(state[:,:16],robot), 'state_proprio_mismatch')
                for col, key in ((16,'object_type_id'),(17,'skill_id')):
                    if 'observations/'+key in h:
                        check(np.array_equal(state[:,col],h['observations/'+key][:].reshape(-1)), 'state_'+key+'_mismatch')
        else:
            errors.append('missing_policy_state_or_proprio')
        if 'dones' in h and total:
            done = h['dones'][:]
            check(bool(done[-1]) and not np.any(done[:-1]), 'done_boundaries')
        if 'step_ids' in h:
            result['step_id_resets'] = int(np.count_nonzero(np.diff(h['step_ids'][:]) < 0))
        target = h.attrs.get('target_object')
        target_id = OBJECTS.index(target)+3 if target in OBJECTS else None
        if target_id is not None and 'observations/object_type_id' in h:
            check(np.all(h['observations/object_type_id'][:] == target_id-3), 'object_id_mapping')
        if 'observations/skill_id' in h and h.attrs.get('policy_skill') in ('pick','place'):
            check(np.all(h['observations/skill_id'][:] == int(h.attrs['policy_skill']=='place')), 'skill_id_mapping')
        crop = json.loads(h.attrs.get('crop_top_left_hw','[0,0]'))
        for view, rgbkey, depthkey, semkey in CAMERAS:
            keys = ['observations/'+k for k in (rgbkey,depthkey,semkey)]
            if not all(k in h for k in keys):
                errors.append('missing_camera:'+view)
                continue
            rgb, depth, semantic = [h[k] for k in keys]
            check(rgb.shape==(total,*image_hw,3) and rgb.dtype==np.uint8, 'rgb_format:'+view)
            check(depth.shape==(total,*image_hw), 'depth_format:'+view)
            check(semantic.shape==(total,*image_hw) and semantic.dtype==np.uint16, 'semantic_format:'+view)
            class_count = len(json.loads(h.attrs['semantic_class_ids']))
            stats = dict(flat_rgb_frames=[], semantic_pixel_counts=[0]*class_count,
                         target_visible_frames=0, target_pixels=0, depth_valid_pixels=0,
                         depth_total_pixels=0, depth_negative_pixels=0, invalid_semantic_pixels=0,
                         conditioning_mask_mismatch_pixels=0)
            counts = np.zeros(class_count,dtype=np.int64)
            for start in range(0,total,32):
                stop = min(total,start+32)
                images, depths, labels = rgb[start:stop], depth[start:stop], semantic[start:stop]
                stats['flat_rgb_frames'].extend(start+i for i,im in enumerate(images) if not rgb_has_spatial_detail(im))
                stats['depth_valid_pixels'] += int(np.sum(np.isfinite(depths)&(depths>0)))
                stats['depth_negative_pixels'] += int(np.sum(np.isfinite(depths)&(depths<0)))
                stats['depth_total_pixels'] += depths.size
                unique, freq = np.unique(labels,return_counts=True)
                for value, count in zip(unique,freq):
                    if int(value) in range(class_count):
                        counts[int(value)] += count
                    else:
                        stats['invalid_semantic_pixels'] += int(count)
                if target_id is not None:
                    target_mask = labels==target_id
                    stats['target_visible_frames'] += int(np.count_nonzero(np.any(target_mask,axis=(1,2))))
                    stats['target_pixels'] += int(np.sum(target_mask))
                maskkey = 'supervision_gt/conditioning_mask_'+view
                if maskkey in h:
                    expected = int(h['supervision_gt'].attrs['conditioning_semantic_id'])
                    stats['conditioning_mask_mismatch_pixels'] += int(np.sum(h[maskkey][start:stop] != (labels==expected)))
                else:
                    check(False, 'missing_conditioning_mask:'+view) if start==0 else None
                for alias, canonical in ALIASES.items():
                    alias_key = 'observations/'+alias
                    if alias_key in h and canonical in (rgbkey,depthkey,semkey):
                        source = images if canonical==rgbkey else depths if canonical==depthkey else labels
                        check(np.array_equal(h[alias_key][start:stop],source,equal_nan=True), 'alias_mismatch:'+alias)
            stats['semantic_pixel_counts'] = counts.tolist()
            stats['depth_valid_fraction'] = stats['depth_valid_pixels']/max(stats['depth_total_pixels'],1)
            check(not stats['flat_rgb_frames'], 'flat_rgb:'+view)
            check(not stats['invalid_semantic_pixels'], 'unknown_semantic_ids:'+view)
            check(not stats['conditioning_mask_mismatch_pixels'], 'conditioning_mask_mismatch:'+view)
            check(not stats['depth_negative_pixels'], 'negative_depth:'+view)
            if stats['depth_valid_fraction'] < .95:
                warnings.append('invalid_or_zero_depth_requires_mask:'+view)
            if stats['target_visible_frames']==0:
                warnings.append('target_never_visible:'+view)
            result['cameras'][view] = stats
            group = 'camera_calibration/'+view
            required = [group+'/'+k for k in ('intrinsics_raw','intrinsics_output','position_b','quaternion_b_ros')]
            if all(k in h for k in required):
                raw, saved, pos, quat = [h[k][:] for k in required]
                check(all(np.isfinite(a).all() for a in (raw,saved,pos,quat)), 'calibration_nonfinite:'+view)
                expected = raw.copy()
                expected[:,0,2] -= crop[1]
                expected[:,1,2] -= crop[0]
                check(np.allclose(expected,saved,atol=1e-5), 'crop_intrinsics_mismatch:'+view)
                check(np.allclose(np.linalg.norm(quat,axis=1),1,atol=.002), 'camera_quaternion_norm:'+view)
            else:
                errors.append('missing_calibration:'+view)
        signature = {name:[info['shape'][1:] if (name.startswith(('observations/','camera_calibration/','debug_gt/'))
                      or name in ('actions','dones','rewards','stage_names','stage_suffixes','step_ids')
                      or 'conditioning_mask_' in name) else info['shape'], info['dtype']]
                     for name,info in result['topics'].items()}
        result['schema_hash'] = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        result['topic_count'] = len(result['topics'])
        result['errors'] = sorted(set(errors))
        result['warnings'] = sorted(set(warnings))
        result['topic_audit_pass'] = not errors
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True, help='One explicit recording run')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    files = sorted(args.source.resolve().glob('*_policy/*/episode_*.h5'))
    if not files:
        raise ValueError('No episode H5 in source run')
    (out/'inventory.json').write_text(json.dumps([str(p.relative_to(ROOT)) for p in files],indent=2))
    reports = []
    for i,path in enumerate(files):
        print(f'AUDIT {i+1}/{len(files)} {path.relative_to(ROOT)}',flush=True)
        try:
            report = inspect(path)
        except Exception as error:
            report = dict(file=str(path.relative_to(ROOT)),topic_audit_pass=False,
                          errors=[type(error).__name__+': '+str(error)])
        reports.append(report)
        (out/f'file_{i:03d}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    summary = dict(files=len(reports), topic_pass=sum(r['topic_audit_pass'] for r in reports),
                   schema_groups=dict(Counter(r.get('schema_hash','error') for r in reports)),
                   object_skill_counts=dict(Counter(str((r.get('metadata',{}).get('target_object'),
                                                         r.get('metadata',{}).get('policy_skill'))) for r in reports)),
                   errors=dict(Counter(e for r in reports for e in r.get('errors',[]))),
                   reports=[{k:v for k,v in r.items() if k!='topics'} for r in reports],
                   note='Topic quality is not a physical-success certificate or a sufficient training sample count.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f'COMPLETE: {summary["topic_pass"]}/{len(reports)} topic audits passed',flush=True)


if __name__=='__main__':
    main()
