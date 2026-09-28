"""CPU coverage/log tests; no simulator or GPU required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from phase4_coverage import progress
from phase4_panel_log import LogFilter,LogTail

class CoverageTests(unittest.TestCase):
    def test_batched_rgb_audit_still_checks_every_frame(self):
        import numpy as np
        from validate_feedback_dataset import flat_rgb_frames
        frames=np.random.default_rng(12).integers(0,256,(70,8,8,3),dtype=np.uint8)
        frames[[0,31,32,69]]=128
        self.assertEqual(flat_rgb_frames(frames),[0,31,32,69])

    def test_session_progress_only_advances_on_saved_count(self):
        import phase4_session as s
        cfg=dict(collection='grid_cycles',cycles=3,mode='auto',target='love_retractor',auto_start=True)
        ns=dict(PHASE3_TARGET_OBJECT='love_retractor')
        with patch.object(s,'load_session',return_value=cfg),patch.object(s,'status'):
            for attempt,saved,cell in ((5,4,4),(6,4,4),(7,5,5)):
                s.before_attempt(ns,attempt,saved,30)
                self.assertEqual(ns['_p4_coverage']['cell_id'],cell)
                self.assertTrue(s.wait_for_start(None,ns))

    def test_disk_audit_rejects_fake_or_missing_cell_success(self):
        import h5py
        from phase4_coverage import audit_directory
        with tempfile.TemporaryDirectory() as tmp:
            for skill in ('pick','place'):
                folder=Path(tmp)/(skill+'_policy')/'scalpel'; folder.mkdir(parents=True)
                for index in range(2):
                    with h5py.File(folder/f'episode_{index:06d}.h5','w') as h5:
                        h5.attrs.update(success=True,coverage_contract='successful_cell_cycles_v1',coverage_cell_id=index,coverage_cycle=1)
            self.assertTrue(audit_directory(tmp,'scalpel','both',1,1,2)['complete'])
            with h5py.File(Path(tmp)/'place_policy/scalpel/episode_000001.h5','r+') as h5: h5.attrs['coverage_cell_id']=0
            report=audit_directory(tmp,'scalpel','both',1,1,2)
            self.assertFalse(report['complete']); self.assertEqual(report['counts'],[1,0]); self.assertTrue(report['errors'])

    def test_success_does_not_flip_working_manual_wrist(self):
        from phase4_feedback import manual_wrist_key,record_wrist_outcome
        ns=dict(PHASE3_TARGET_OBJECT='love_retractor',_p4_manual_positions={'love_retractor':dict(x=.42,y=-.59,yaw_deg=0)})
        key=manual_wrist_key(ns); ns['_p4_active_wrist_key']=key
        record_wrist_outcome(ns,True)
        self.assertFalse(ns.get('_p4_wrist_branches',{}).get(key,False))
        record_wrist_outcome(ns,False)
        self.assertTrue(ns['_p4_wrist_branches'][key])
        record_wrist_outcome(ns,True)
        self.assertTrue(ns['_p4_wrist_branches'][key])
        record_wrist_outcome(ns,False)
        self.assertFalse(ns['_p4_wrist_branches'][key])

    def test_three_rounds_cover_every_cell(self):
        seen=[]
        for saved in range(30):
            c=progress(saved,3,5,2)
            seen.append(c['cell_id'])
            self.assertEqual(c,progress(saved,3,5,2)) # failure leaves saved unchanged
        self.assertEqual(seen,list(range(10))*3)
        done=progress(30,3,5,2)
        self.assertTrue(done['complete']); self.assertIsNone(done['cell_id'])
        self.assertEqual(done['counts'],[3]*10)

    def test_forced_target_cell_unique_and_bounded(self):
        import numpy as np
        import phase4_cell_spawn as s
        from phase4_scene import LAYOUT
        names=['scalpel','scissor','love_retractor','kelly','scalpel_type2']
        env={n:{'radius_m':.0824} for n in names}
        rng=np.random.default_rng(17)
        with patch.object(s,'instrument_envelopes',return_value=env):
            for cell in range(LAYOUT['grid_rows']*LAYOUT['grid_cols']):
                for target in names:
                    spawn={n:{'yaw_deg':float(rng.uniform(0,360))} for n in names}; spawn['grid']={}
                    out=s.fit_spawn_to_cells(spawn,target,rng,cell)
                    self.assertEqual(out[target]['cell_id'],cell)
                    self.assertEqual(len({out[n]['cell_id'] for n in names}),5)
                    for n in names:
                        p=out[n]; r,c=p['row'],p['col']
                        for k,idx,bounds,num in [('center_x' if n=='scalpel' else 'x',c,LAYOUT['grid_x'],LAYOUT['grid_cols']),('center_y' if n=='scalpel' else 'y',r,LAYOUT['grid_y'],LAYOUT['grid_rows'])]:
                            size=(bounds[1]-bounds[0])/num
                            self.assertGreaterEqual(p[k]-.0824,bounds[0]+idx*size)
                            self.assertLessEqual(p[k]+.0824,bounds[0]+(idx+1)*size)

    def test_log_filters_noise_and_keeps_fail_save(self):
        f=LogFilter()
        self.assertIsNone(f.feed('[REC step=0100] stage=LOVE_CLOSE'))
        self.assertEqual(f.feed('[PHASE FAIL] object did not lift')[0],'error')
        self.assertIn('did not lift',f.last_reason)
        self.assertEqual(f.feed('[RESULT] success=True steps=300')[0],'info')
        self.assertEqual(f.feed('[SAVE SPLIT] 1/30 -> pick+place')[0],'success')
        f.feed('[P4 STAGE] LOVE_CLOSE')
        self.assertIn('LOVE_CLOSE',f.feed('Steps 14 | sim 0.280s | wall 6.2s')[1])

    def test_tail_partial_lines_and_no_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'console.log'; p.write_bytes(b'first\npar')
            t=LogTail(); self.assertEqual(t.read(p),['first'])
            with p.open('ab') as f: f.write(b'tial\n')
            self.assertEqual(t.read(p),['partial']); self.assertEqual(t.read(p),[])

if __name__=='__main__': unittest.main()
