"""Static physics-settled scenes, DLAA 448, ring + top PNG and COCO capture."""
import argparse
import json
import os
from pathlib import Path
import random
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'compat')]
from detection.dataset import CLASSES, camera_views, write_json, export_index
from detection.placement import ScenePlacementError, retry_placement
from detection.paths import object_datasets_root
from datetime import datetime


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=object_datasets_root()/'detection_runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    p.add_argument('--scenes',type=int,default=100)
    p.add_argument('--min-objects',type=int,default=8)
    p.add_argument('--max-objects',type=int,default=18)
    p.add_argument('--ring-cameras',type=int,default=8)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--robot-visible-probability',type=float,default=.25)
    p.add_argument('--tray',choices=('random','empty','full'),default='random')
    p.add_argument('--randomization',choices=('original','wide'),default='original')
    p.add_argument('--randomization-config',type=Path,help='JSON overrides for wide mode')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-file',type=Path)
    a=p.parse_args()
    lo,cap,views=(1,64,32) if a.randomization=='wide' else (5,30,16)
    if not (1<=a.scenes and lo<=a.min_objects<=a.max_objects<=cap and 3<=a.ring_cameras<=views and 0<=a.robot_visible_probability<=1):
        p.error(f'Use scenes >=1, {lo}<=min<=max<={cap}, 3..{views} ring cameras, robot probability 0..1')
    if a.randomization_config and a.randomization!='wide': p.error('JSON overrides require --randomization wide')
    from detection.randomization import load_profile
    a.wide=load_profile(a.randomization_config) if a.randomization=='wide' else None
    a.output=a.output.resolve()
    return a


def main():
    args=arguments()
    contract=dict(contract='p4_static_detection_v1',resolution=448,antialiasing='DLAA',materials='recorder',
                  seed=args.seed,min_objects=args.min_objects,max_objects=args.max_objects,
                  ring_cameras=args.ring_cameras,robot_visible_probability=args.robot_visible_probability,tray=args.tray,
                  split='train',robot_pose='home_static',labels='visible_rigid_instance_boxes')
    if args.wide is not None:
        contract.update(randomization='wide',wide_profile=args.wide,capacity_policy='stop_at_fit_after_minimum')
        from detection.randomization import install_wide_lighting
        install_wide_lighting(args.wide)
    config=args.output/'config.json'
    if args.resume:
        if not config.is_file() or json.loads(config.read_text())!=contract:
            raise ValueError('Resume configuration mismatch or missing config')
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        write_json(config,contract)
    (args.output/'train'/'scenes').mkdir(parents=True,exist_ok=True)
    os.environ.update(P4_CAMERA_SIZE='448',P4_ANTIALIASING_MODE='DLAA',P4_INSTRUMENT_MATERIALS='recorder',
                      P4_RANDOMIZATION='train',P4_RANDOMIZATION_SEED=str(args.seed),P4_TRAY_OCCUPANCY='empty')
    # Ignore DP panel environment settings in this separate process.
    for key in ('P4_SESSION_CONFIG','P4_SESSION_FILE'):
        os.environ.pop(key,None)
    backend=ROOT/'backends/phase3_grid_split_scalpel_recorder.py'
    sys.argv=[str(backend),'--episodes','0','--layout_tuner','--out_dir',str(args.output/'bootstrap')]
    if args.headless: sys.argv.append('--headless')
    loaded=runpy.run_path(str(backend),run_name='detection_scene_bootstrap')
    ns=loaded['main'].__globals__
    from phase3_shared_env_cfg import RecorderEnvRequest
    ns['RecorderEnvReque…6636 tokens truncated…ksum'):require_commit(first,h)
            manifest.write_text(original)
            self.assertEqual(assignment(root,'scalpel',3,{})['dataset_split'],'train')
            from src.recorder.phase4_coverage import audit_directory
            self.assertTrue(audit_directory(root,'scalpel','both',1,1,3)['complete'])
            from src.recorder.episode_split import committed_assignments
            self.assertEqual(max(r['split_randomization_attempt'] for r in committed_assignments(root,'scalpel')),3)
            documents={
                'capture_contract.json':dict(split='auto',split_contract='episode_grid_balanced_v1',
                    consumers=['detection','dp_joint'],session_seed=42,saved_skill='both'),
                'session.json':dict(target='scalpel',collection='single'),
                'session.status.json':dict(state='complete',requested=3),
                'run_metrics.json':dict(runtime_error=None)}
            for name, payload in documents.items():
                (root/name).write_text(json.dumps(payload))
            splits,pairs=inventory(root,'both')
            self.assertEqual([len(splits[s]) for s in ('train','valid','test')],[1,1,1])
            self.assertEqual([p[1] for p in pairs],['train','valid','test'])
            # Cross-split movement must not silently relabel a committed pair.
            first=episode_files(root,'pick','scalpel')[0]
            wrong=root/'valid'/'pick_policy'/'scalpel'/first.name
            first.rename(wrong)
            with h5py.File(wrong) as h:
                with self.assertRaises(ValueError): require_commit(wrong,h)
            wrong.rename(first)
            from src.recorder.verified_exit import verify_and_exit
            exits=[]
            verify_and_exit(root,'scalpel','both',3,exit_process=exits.append)
            self.assertEqual(exits,[0])


if __name__=='__main__': unittest.main()
