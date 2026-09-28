"""CPU tests for session flow, canonical spawn math, and responsive GUI."""
import unittest
from unittest.mock import patch
import types
import itertools

class RetryTests(unittest.TestCase):
    def test_canonical_pose_centers_and_clears_support_at_any_yaw(self):
        import numpy as np
        from phase4_reset import table_pose
        from phase4_grasp_validation import rotate
        points=np.array(list(itertools.product([-.01,.01],[.06,.07],[-.075,.075])))
        t=dict(yaw=20.,quat=np.array([1.,0,0,0]),points=points)
        for yaw in (0,90,180,270,359):
            pose=table_pose(t,.42,-.59,yaw)
            actual=rotate(pose[3:],points)+pose[:3]
            np.testing.assert_allclose((actual.min(0)+actual.max(0))[:2]/2,[.42,-.59],atol=1e-8)
            self.assertAlmostEqual(actual[:,2].min(),.0015)

    def test_autostart_retries_without_prepare_or_start_click(self):
        import phase4_session as s
        cfg=dict(mode='manual',target='scalpel',auto_start=True,prepare_id=1,positions={},discard_id=3)
        ns=dict(PHASE3_TARGET_OBJECT='scalpel',_p4_prepare_id=1)
        with patch.object(s,'load_session',return_value=cfg),patch.object(s,'status'),patch.object(s,'validate_positions',return_value={'scalpel':{}}),patch.object(s,'validate_target_workspace'):
            s.before_attempt(ns,2,1,3)
            self.assertTrue(s.wait_for_start(None,ns))
            self.assertEqual(ns['_p4_discard_id'],3)
            self.assertEqual(ns['_p4_saved_count'],1)

    def test_user_failed_near_base_pose_is_rejected(self):
        from phase4_session import validate_target_workspace
        layout=dict(robot_pos=[0,-.1924,.003])
        with self.assertRaisesRegex(ValueError,'too close'):
            validate_target_workspace('love_retractor',{'love_retractor':dict(x=.218028,y=-.061935)},layout)
        validate_target_workspace('love_retractor',{'love_retractor':dict(x=.42,y=-.59)},layout)

    def test_stop_not_ignored_in_unlimited_mode(self):
        import phase4_session as s
        with patch.object(s,'load_session',return_value=dict(stop=True)):
            with self.assertRaisesRegex(RuntimeError,'stopped'):
                s.before_attempt({},5,0,10)

class PanelTests(unittest.TestCase):
    def test_grid_launch_requests_saved_goal_and_no_attempt_limit(self):
        import tkinter as tk
        import tempfile,json
        from pathlib import Path
        from unittest.mock import Mock
        import control_panel as ui
        root=tk.Tk(); root.withdraw(); panel=ui.Panel(root)
        try:
            panel.collection.set('grid_cycles'); panel.episodes.set(3)
            proc=Mock(); proc.poll.return_value=None
            with tempfile.TemporaryDirectory() as tmp,patch.object(ui,'ROOT',Path(tmp)),patch.object(ui.subprocess,'Popen',return_value=proc) as launch,patch.object(ui.messagebox,'showerror') as error:
                panel.destination.set(tmp)
                panel.prepare()
                error.assert_not_called()
                args=launch.call_args.args[0]
                self.assertEqual(args[args.index('--episodes')+1],'30')
                self.assertEqual(args[args.index('--max-attempts')+1],'0')
                config=json.loads(panel.session_path.read_text())
                self.assertTrue(config['auto_start']); self.assertEqual(config['cycles'],3)
                self.assertEqual(config['mode'],'auto')
                panel.log_file.close(); panel.log_file=None; panel.process=None
        finally:
            if panel.log_file: panel.log_file.close()
            panel.close()

    def test_responsive_mapping_and_dark_theme(self):
        import tkinter as tk
        from control_panel import Panel,NAMES,RADII
        from phase4_session import validate_positions,validate_target_workspace
        root=tk.Tk(); root.withdraw()
        try:
            panel=Panel(root)
            validate_positions(panel.positions,panel.layout,RADII)
            for n in NAMES: validate_target_workspace(n,panel.positions,panel.layout)
            for w,h in ((1020,780),(1920,1080)):
                root.geometry(f'{w}x{h}'); root.deiconify(); root.update()
                for xy in ((.42,-.59),(0,-.1924)):
                    actual=panel.canvas_to_xy(*panel.xy_to_canvas(*xy))
                    for a,b in zip(actual,xy): self.assertAlmostEqual(a,b,places=10)
                scale,ox,oy=panel.view_transform()
                self.assertAlmostEqual(ox+1.6*scale/2,panel.canvas.winfo_width()/2)
            self.assertEqual(root.cget('bg'),'#151b24')
            self.assertFalse(hasattr(panel,'max_attempts'))
        finally: panel.close()

if __name__=='__main__': unittest.main()
