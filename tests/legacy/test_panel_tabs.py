import tkinter as tk
import unittest
from control_panel import Panel
from phase4_coverage import progress
from phase4_panel_widgets import cell_style


class CoverageAndTabsTests(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk();self.panel=Panel(self.root);self.root.update()
    def tearDown(self): self.panel.close()
    def test_rounds_cover_every_cell_once_before_repeating(self):
        self.assertEqual([progress(i,2,5,2)['cell_id'] for i in range(20)],list(range(10))*2)
        self.assertEqual(progress(10,1,5,2)['counts'],[1]*10)
        self.assertEqual(progress(20,2,5,2)['counts'],[2]*10)
        self.assertIsNone(progress(20,2,5,2)['cell_id'])
    def test_current_done_partial_markers_and_no_yellow_robot_circle(self):
        p=self.panel;p.collection.set('grid_cycles');p.episodes.set(1)
        p.coverage=progress(4,1,5,2);p.redraw()
        self.assertEqual(len(p.canvas.find_withtag('coverage_cell')),10)
        self.assertEqual(len(p.canvas.find_withtag('DONE')),4)
        self.assertEqual(p.canvas.find_withtag('CURRENT'),p.canvas.find_withtag('cell_4'))
        for item in p.canvas.find_all():
            if p.canvas.type(item)=='oval': self.assertNotEqual(p.canvas.itemcget(item,'outline'),'#d99b4a')
        p.episodes.set(2);p.coverage=progress(14,2,5,2);p.redraw()
        self.assertEqual(len(p.canvas.find_withtag('DONE')),4)
        self.assertEqual(len(p.canvas.find_withtag('PARTIAL')),5)
        self.assertIn('1/2 saved',[p.canvas.itemcget(i,'text') for i in p.canvas.find_withtag('coverage_label')][4])
        p.coverage=progress(20,2,5,2);p.redraw()
        self.assertEqual(len(p.canvas.find_withtag('DONE')),10);self.assertFalse(p.canvas.find_withtag('CURRENT'))
    def test_tabs_controls_tooltips_and_size(self):
        p=self.panel;self.assertEqual(len(p.notebook.tabs()),3);self.assertGreaterEqual(len(p.tooltips),17)
        for width,height in ((1100,780),(1920,1080)):
            self.root.geometry(f'{width}x{height}');self.root.update()
            for page in p.pages.values():
                p.notebook.select(page);self.root.update()
                self.assertTrue(page.winfo_ismapped())
                for button in p.action_buttons.values():
                    self.assertTrue(button.winfo_ismapped())
                    self.assertLessEqual(button.winfo_rootx()+button.winfo_width(),self.root.winfo_rootx()+self.root.winfo_width())
                    self.assertLessEqual(button.winfo_rooty()+button.winfo_height(),self.root.winfo_rooty()+self.root.winfo_height())
            p.notebook.select(p.pages['workspace']);self.root.update()
            self.assertGreater(p.canvas.winfo_height(),200)
        tip=p.tooltips[-1];tip.show();self.assertIsNotNone(tip.popup);tip.hide();self.assertIsNone(tip.popup)


if __name__=='__main__': unittest.main()
