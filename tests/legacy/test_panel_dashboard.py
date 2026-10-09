import unittest,math
from phase4_panel_state import Dashboard,rotate_outline

class DashboardTests(unittest.TestCase):
    def test_fail_deduplicates_and_only_saved_counts_success(self):
        d=Dashboard(30)
        d.feed('ATTEMPT 1 | SUCCESS 0/30')
        d.feed('[RESULT] success=True steps=100')
        self.assertEqual(d.saved,0); self.assertEqual(d.phase,'end')
        d.feed('[SENSOR FAIL] flat RGB'); d.feed('[PHASE FAIL] pick | no proof')
        self.assertEqual(d.failures,1)
        d.feed('ATTEMPT 2 | SUCCESS 0/30')
        d.feed('[P4 ACTIVE STAGE] LOVE_CLOSE')
        self.assertEqual(d.phase,'recording'); self.assertEqual(d.stage,'LOVE_CLOSE')
        d.feed('[SAVE SPLIT] 1/30 -> pick+place')
        self.assertEqual((d.saved,d.episode,d.attempt,d.failures),(1,2,2,1))
        d.feed('[SAVE SPLIT] 1/30 -> pick+place')
        self.assertEqual(d.saved,1)
        d.finish(dict(success_count=1,failure_count=1,total_attempts=2),1)
        self.assertIn('incomplete',d.stage)

    def test_complete_does_not_show_nonexistent_next_episode(self):
        d=Dashboard(1); d.feed('ATTEMPT 1 | SUCCESS 0/1'); d.feed('[SAVE SPLIT] 1/1 -> both')
        self.assertEqual(d.episode,1)
        d.finish(dict(success_count=1,failure_count=0,total_attempts=1),0)
        self.assertEqual(d.stage,'Complete')

    def test_setup_stages_are_not_shown_as_recording(self):
        d=Dashboard(1)
        d.feed('[P4 ACTIVE STAGE] SCALPEL_OPEN_HOVER')
        self.assertEqual(d.phase,'reset')
        d.feed('[P4 ACTIVE STAGE] SCALPEL_LOWER_PRE')
        self.assertEqual(d.phase,'recording')

    def test_asset_axis_not_universal_yaw_line(self):
        item=dict(hull_xy_m=[[-.01,-.075],[.01,-.075],[.01,.075],[-.01,.075]],axis_xy=[0,1])
        a=rotate_outline(item,dict(x=.42,y=-.59,yaw_deg=0))
        b=rotate_outline(item,dict(x=.42,y=-.59,yaw_deg=90))
        self.assertEqual(a['axis_xy'],[0,1])
        self.assertAlmostEqual(b['axis_xy'][0],-1); self.assertAlmostEqual(b['axis_xy'][1],0)
        self.assertAlmostEqual(math.dist(a['hull_xy'][0],a['hull_xy'][1]),math.dist(b['hull_xy'][0],b['hull_xy'][1]))

class DashboardGuiTests(unittest.TestCase):
    def test_counters_and_three_lights(self):
        import tkinter as tk
        from control_panel import Panel
        root=tk.Tk(); root.withdraw(); panel=Panel(root)
        try:
            panel.collection.set('grid_cycles'); panel.episodes.set(3)
            self.assertIn('Saved 0/30',panel.counter_text.get())
            panel.dashboard.feed('ATTEMPT 2 | SUCCESS 1/30')
            panel.dashboard.feed('[P4 ACTIVE STAGE] SCALPEL_CLOSE'); panel.update_dashboard()
            self.assertIn('Attempt 2',panel.counter_text.get())
            self.assertEqual(panel.active_stage.get(),'SCALPEL_CLOSE')
            circles=[i for i in panel.lamps.find_all() if panel.lamps.type(i)=='oval']
            self.assertEqual(len(circles),3)
            self.assertEqual(sum(panel.lamps.itemcget(i,'fill')=='#ff424f' for i in circles),1)
            self.assertEqual(len(panel.preview_geometry),5)
            panel.session_path=__import__('pathlib').Path('previous_session.json')
            panel.episodes.set(2)
            self.assertIn('Saved 0/20',panel.counter_text.get())
            panel.session_path=None
        finally: panel.close()

if __name__=='__main__': unittest.main()
