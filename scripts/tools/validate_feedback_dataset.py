"""Read-only v4 H5 alignment/control audit; does not certify physical success."""
import argparse
import json
from collections import Counter
from pathlib import Path
import h5py
import numpy as np


def flat_rgb_frames(node,batch_size=32):
    """Read contiguous batches so compressed H5 chunks aren't inflated per frame."""
    from phase4_feedback import rgb_has_spatial_detail
    bad=[]
    for start in range(0,len(node),batch_size):
        frames=node[start:start+batch_size]
        bad.extend(start+i for i,frame in enumerate(frames) if not rgb_has_spatial_detail(frame))
    return bad


def audit(path):
    with h5py.File(path, 'r') as h5:
        actions = h5['actions'][:]
        total = len(actions)
        image_hw = tuple(h5['observations/front_rgb'].shape[1:3])
        assert image_hw in ((224, 224), (448, 448)), (path, image_hw)
        assert bool(h5.attrs['success']), (path,'failed episode saved')
        assert total > 0 and actions.shape == (total,8), (path,actions.shape)
        assert np.isfinite(actions).all(), (path,'nonfinite actions')
        assert np.allclose(np.linalg.norm(actions[:,3:7],axis=1),1.,atol=.002), (path,'quaternions')
        assert np.isin(actions[:,-1],[-1.,1.]).all(), (path,'nonbinary gripper')
        version=h5.attrs['p4_feedback_version']
        assert version in ('20260915-v5','20260916-tray-slots-v1'), (path,'version')
        evidence = json.loads(h5.attrs['expert_grasp_evidence'])
        assert evidence['lift_verified'] and evidence['place_verified'], (path,'missing physical evidence')
        if version=='20260916-tray-slots-v1':
            from phase4_tray_slots import ORDER, CENTER_TOL, ANGLE_TOL
            assert evidence['tray_contract']=='tray_slots_v1'
            slot=int(h5.attrs['tray_slot_id'])
            assert slot==int(h5.attrs['object_type_id']) and h5.attrs['target_object']==ORDER[slot]
            occupancy=json.loads(h5.attrs['initial_tray_occupancy'])
            assert json.loads(h5.attrs['tray_slot_order'])==list(ORDER)
            assert evidence['slot']['slot_id']==slot and evidence['slot']['object']==ORDER[slot]
            assert len(occupancy)==5 and set(occupancy)<={0,1} and occupancy[slot]==0
            assert [n for n,on in zip(ORDER,occupancy) if on]==evidence['initial_tray_objects']
            final=evidence['final_placement']
            assert final['inside_slot'] and final['center_error_m']<=CENTER_TOL and final['axis_error_deg']<=ANGLE_TOL
            assert abs(final['bottom_m']-evidence['tray_support_z'])<=.005
            assert .003<=evidence['release_measurement']['gap_above_support_m']<=.018
            assert set(evidence['final_other_tray_geometry'])==set(evidence['initial_tray_objects'])
            for other in evidence['final_other_tray_geometry'].values():
                assert other['inside_slot'] and other['center_error_m']<=CENTER_TOL and other['axis_error_deg']<=ANGLE_TOL
        assert h5.attrs['control_tcp_matches_observed_ee'], (path,'TCP mismatch')
        assert np.allclose(json.loads(h5.attrs['control_tcp_offset_m']),[0.,0.,.1034]), (path,'TCP offset')
        topics, rgb, semantics = {}, [], []
        def visit(name, node):
            if isinstance(node,h5py.Dataset):
                topics[name] = {'shape':list(node.shape),'dtype':str(node.dtype)}
                if name.startswith(('observations/','camera_calibration/')) and node.ndim:
                    assert len(node) == total, (path,name,node.shape,total)
                if name.endswith('_rgb'):
                    assert node.shape == (total,*image_hw,3), (path,name,node.shape)
                    assert node.dtype == np.uint8, (path,name,node.dtype)
                    # A stream can initialize correctly then become a flat
                    # gray render buffer. Check EVERY recorded/cropped frame.
                    bad=flat_rgb_frames(node)
                    assert not bad, (path,name,'flat RGB frames',bad[:20],len(bad))
                    rgb.append(name)
                if name.endswith('_semantic'):
                    assert node.shape == (total,*image_hw), (path,name,node.shape)
                    assert node.dtype == np.uint16, (path,name,node.dtype)
                    ids = set()
                    for i in {0,total//2,total-1}:
                        ids.update(int(v) for v in np.unique(node[i]))
                    assert ids <= set(json.loads(h5.attrs['semantic_class_ids']).values()), (path,name,ids)
                    semantics.append(name)
        h5.visititems(visit)
        from src.recorder.instance_labels import audit_instances
        audit_instances(h5)
        assert len(rgb) == len(semantics) == 6, (path,rgb,semantics)
        assert np.isfinite(h5['observations/robot_proprio'][:]).all(), (path,'proprio')
        stages = [v.decode() if isinstance(v,bytes) else str(v) for v in h5['stage_names'][:]]
        assert len(stages) == total, (path,'stage alignment')
        assert not any('MICRO' in n or 'HOLD_AFTER_CLOSE' in n for n in stages), (path,'obsolete stage')
        for i,name in enumerate(stages):
            if name.endswith('_CLOSE'):
                assert actions[i,-1] == -1, (path,'close delayed')
        close = np.array([n.endswith('_CLOSE') for n in stages])
        close_drift = None
        if close.any():
            assert np.max(np.ptp(actions[close,:3],axis=0)) < 1e-6, (path,'moving close command')
            ee = h5['observations/robot_proprio'][:][close,9:12]
            close_drift = float(np.linalg.norm(ee-ee[0],axis=1).max()*1000)
        return {'file':str(path),'steps':total,'stage_counts':dict(Counter(stages)),
                'close_measured_ee_drift_mm':close_drift,
                'rgb_streams':rgb,'semantic_streams':semantics,'topic_count':len(topics),'topics':topics,'result':'PASS'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--details',action='store_true')
    args = parser.parse_args()
    files = sorted(args.root.rglob('episode_*.h5'))
    if not files:
        raise SystemExit('No completed episode H5 files found')
    reports = [audit(path) for path in files]
    signatures = [{name:(tuple(info['shape'][1:] if info['shape'] and info['shape'][0] == report['steps'] else info['shape']),info['dtype'])
                   for name,info in report['topics'].items()} for report in reports]
    assert all(s == signatures[0] for s in signatures), 'H5 topic/shape/dtype schema differs across episodes'
    if not args.details:
        for report in reports:
            report.pop('topics')
    print(json.dumps(reports,indent=2))
