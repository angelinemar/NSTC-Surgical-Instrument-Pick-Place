import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.recorder.capture_contract import capture_contract,storage_budget
from training.export_recordings import inventory


class CaptureTests(unittest.TestCase):
    def test_all_purpose_skill_combinations(self):
        for purpose in ('detection','dp','both'):
            for skill in ('pick','place','both'):
                c=capture_contract(purpose,skill,448,41,'train')
                self.assertFalse(c['duplicate_raw'])
                self.assertEqual(len(c['consumers']),2 if purpose=='both' else 1)
                self.assertEqual(c['saved_skill'],skill)

    def test_disk_guard_never_counts_both_consumers_twice(self):
        with patch('src.recorder.capture_contract.shutil.disk_usage',return_value=Mock(free=227_000_000_000)):
            self.assertTrue(storage_budget(ROOT,150,448)['capacity_pass'])
            self.assertFalse(storage_budget(ROOT,200,448)['capacity_pass'])
            self.assertFalse(storage_budget(ROOT,500,448)['capacity_pass'])
            self.assertTrue(storage_budget(ROOT,500,224)['capacity_pass'])

    def test_explicit_split_inventory_supports_pick_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for seed,split in enumerate(('train','valid','test')):
                folder=root/split; folder.mkdir()
                (folder/'capture_contract.json').write_text(json.dumps(capture_contract('both','pick',224,seed,split)))
                (folder/'run_metrics.json').write_text('{}')
                (folder/'session.json').write_text(json.dumps(dict(target='scalpel',collection='single')))
                (folder/'session.status.json').write_text(json.dumps(dict(state='complete',requested=1)))
                target=folder/'pick_policy/scalpel';target.mkdir(parents=True)
                (target/'episode_000000.h5').touch()
            splits,pairs=inventory(root,'both')
            self.assertEqual(len(pairs),3)
            self.assertTrue(all(len(paths)==1 for _,_,paths in pairs))
            self.assertTrue(all(case['saved_skills']==['pick'] for case,_,_ in pairs))
            path=root/'test/capture_contract.json'
            c=json.loads(path.read_text());c['session_seed']=0;path.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'leakage'): inventory(root,'both')

    def test_background_deterministic_and_no_physics_scale_change(self):
        from src.recorder.domain_randomization import sample_background
        self.assertEqual(sample_background(5),sample_background(5))
        self.assertEqual(len({sample_background(i)['profile'] for i in range(40)}),3)
        self.assertTrue(all(sample_background(i)['physical_scale']==1 for i in range(40)))


class ExitTests(unittest.TestCase):
    def test_verified_exit_requires_commit_and_exact_skills(self):
        import h5py
        from src.recorder.verified_exit import verify_and_exit
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); target=root/'pick_policy/scalpel';target.mkdir(parents=True)
            path=target/'episode_000000.h5'
            with h5py.File(path,'w') as h:
                h.attrs.update(storage_contract='journaled_episode_v2',success=True,policy_skill='pick',
                               target_object='scalpel',pair_transaction_id='unit_transaction')
            relative=str(path.relative_to(root)); digest=hashlib.sha256(path.read_bytes()).hexdigest()
            (root/'.commits').mkdir()
            commit=root/'.commits/scalpel_episode_000000.json'
            commit.write_text(json.dumps(dict(transaction_id='unit_transaction',files=[relative],sha256={relative:digest})))
            exit_mock=Mock()
            verify_and_exit(root,'scalpel','pick',1,exit_mock)
            exit_mock.assert_called_once_with(0)
            self.assertTrue(json.loads((root/'run_completion.json').read_text())['recording_complete'])
            exit_mock.reset_mock()
            with self.assertRaises(RuntimeError): verify_and_exit(root,'scalpel','both',1,exit_mock)
            exit_mock.assert_not_called()
            with h5py.File(path,'r+') as h: h.attrs['tampered']=True
            with self.assertRaisesRegex(ValueError,'checksum'): verify_and_exit(root,'scalpel','pick',1,exit_mock)
            exit_mock.assert_not_called()


if __name__=='__main__': unittest.main()
