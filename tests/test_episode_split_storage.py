import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'compat'))
from src.recorder.episode_split import assignment, episode_files, summary
from src.recorder.phase4_storage import EpisodeTransaction, require_commit
from training.export_recordings import inventory


def writer(recorder, ep_idx, segment, out_dir, success, object_name, meta=None):
    folder=Path(out_dir); folder.mkdir(parents=True,exist_ok=True)
    with h5py.File(folder/f'episode_{ep_idx:06d}.h5','w') as h:
        h.attrs.update(meta)
        h.attrs.update(target_object=object_name,success=True,policy_skill=segment)
        h.create_dataset('actions',data=np.zeros((1,8)))
    return True


class SplitStorageTests(unittest.TestCase):
    @patch('validate_feedback_dataset.audit')
    @patch('src.recorder.instance_labels.append_segment')
    def test_pair_commit_resume_inventory_and_exit(self, append, audit):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            root=Path(tmp)
            for i in range(3):
                meta=dict(coverage_cell_id=i,coverage_cycle=1,coverage_contract='successful_cell_cycles_v1',
                          spawn_requested_all=json.dumps({'scalpel':{'yaw_deg':i*80}}),
                          domain_randomization=json.dumps(dict(seed=42+i*1009,session_seed=42,attempt=i+1)))
                allocated=assignment(root,'scalpel',i,meta)
                tx=EpisodeTransaction(root/allocated['dataset_split'],'scalpel',i,'both',allocated)
                for skill in ('pick','place'):
                    bound=inspect.signature(writer).bind(None,i,skill,str(root),True,'scalpel',dict(meta))
                    tx.save(writer,bound)
                    if skill=='pick':
                        self.assertEqual(sum(summary(root,'scalpel')['counts'].values()),i)
                for skill in ('pick','place'):
                    path=episode_files(root,skill,'scalpel')[i]
                    with h5py.File(path) as h:
                        require_commit(path,h)
                        self.assertEqual(h.attrs['dataset_split'],allocated['dataset_split'])
            self.assertEqual(summary(root,'scalpel')['counts'],dict(train=1,valid=1,test=1))
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
