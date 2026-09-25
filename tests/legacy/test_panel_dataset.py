"""Run with system Python (Tk), not the Isaac runtime."""
import json
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import Mock,patch
import control_panel as ui
from src.recorder.capture_contract import LABELS


class DatasetPanelTests(unittest.TestCase):
    def test_every_purpose_and_motion_reaches_recorder(self):
        for label,purpose in LABELS.items():
            for skill in ('pick','place','both'):
                with self.subTest(purpose=purpose,skill=skill),tempfile.TemporaryDirectory() as tmp:
                    root=tk.Tk();root.withdraw();panel=ui.Panel(root)
                    try:
                        panel.destination.set(tmp);panel.mode.set('auto')
                        panel.dataset_purpose.set(label);panel.skill.set(skill)
                        panel.dataset_split.set('train');panel.session_seed.set(101)
                        panel.camera_size.set('448')
                        proc=Mock();proc.poll.return_value=None
                        with patch.object(ui,'ROOT',Path(tmp)),patch.object(ui.subprocess,'Popen',return_value=proc) as launch,patch.object(ui.messagebox,'showerror') as error:
                            panel.prepare();error.assert_not_called()
                            args=launch.call_args.args[0]
                            for key,value in (('--record_mode',skill),('--dataset-purpose',purpose),('--camera-size','448'),('--dataset-split','train'),('--randomization-seed','101')):
                                self.assertEqual(args[args.index(key)+1],value)
                            contract=json.loads((panel.output_dir/'capture_contract.json').read_text())
                            self.assertEqual(contract['purpose'],purpose)
                            self.assertTrue(Path(panel.session['console_log']).is_relative_to(Path(tmp)/'debug/logs'))
                            self.assertEqual(panel.session['tray_mode'],'random')
                            self.assertEqual((panel.session['distractor_min'],panel.session['distractor_max']),(12,18))
                    finally:
                        if panel.log_file: panel.log_file.close()
                        panel.log_file=None;panel.process=None;panel.close()

    def test_dp_rejects_ambiguous_tray_and_low_capacity_stops_launch(self):
        root=tk.Tk();root.withdraw();panel=ui.Panel(root)
        try:
            panel.tray_mode.set('manual')
            panel.tray_counts['scalpel'].set(1)
            with patch.object(ui.messagebox,'showerror') as error,patch.object(ui.subprocess,'Popen') as launch:
                panel.prepare();error.assert_called_once();launch.assert_not_called()
            panel.tray_counts['scalpel'].set(0);panel.tray_mode.set('random');panel.mode.set('auto')
            with patch.object(ui,'storage_budget',return_value=dict(capacity_pass=False,estimated_bytes=900000000,free_bytes=100)),patch.object(ui.messagebox,'showerror') as error,patch.object(ui.subprocess,'Popen') as launch:
                panel.prepare();error.assert_called_once();launch.assert_not_called()
        finally: panel.close()

    def test_gui_defaults_auto_split_and_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=tk.Tk();root.withdraw();panel=ui.Panel(root)
            try:
                panel.destination.set(tmp);panel.mode.set('auto')
                proc=Mock();proc.poll.return_value=None
                with patch.object(ui,'ROOT',Path(tmp)),patch.object(ui.subprocess,'Popen',return_value=proc) as launch,patch.object(ui.messagebox,'showerror') as error:
                    panel.prepare();error.assert_not_called()
                    args=launch.call_args.args[0]
                    self.assertEqual(args[args.index('--dataset-split')+1],'train')
                    self.assertNotEqual(args[args.index('--randomization-seed')+1],'')
                    self.assertEqual(panel.output_dir.parents[1],Path(tmp)/'train')
            finally:
                if panel.log_file: panel.log_file.close()
                panel.log_file=None;panel.process=None;panel.close()


if __name__=='__main__': unittest.main()
