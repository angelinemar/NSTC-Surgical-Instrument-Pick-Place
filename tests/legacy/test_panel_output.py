import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch
from control_panel import Panel


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk(); self.root.withdraw(); self.panel=Panel(self.root)
        self.tmp=tempfile.TemporaryDirectory(); self.folder=Path(self.tmp.name)
        self.panel.output_dir=self.folder

    def tearDown(self):
        self.panel.close(); self.tmp.cleanup()

    def test_buttons_open_correct_subfolders(self):
        for kind,subdir in [('run',''),('pick','pick_policy'),('place','place_policy'),('gif','gifs'),('failure','failure_previews')]:
            target=self.folder/subdir; target.mkdir(exist_ok=True)
            with patch('control_panel.os.startfile') as launch:
                self.panel.open_output(kind); launch.assert_called_once_with(str(target))

    def test_absent_gif_explained_not_fabricated(self):
        with patch('control_panel.os.startfile') as launch,patch('control_panel.messagebox.showinfo') as info:
            self.panel.open_output('gif'); launch.assert_not_called()
            self.assertIn('No GIF',info.call_args.args[1])

    def test_browse_existing_run(self):
        with patch('control_panel.filedialog.askdirectory',return_value=str(self.folder)):
            self.panel.choose_output()
        self.assertEqual(self.panel.output_text.get(),str(self.folder))

    def test_nested_episode_gif(self):
        target=self.folder/'pick_policy'/'scissor';target.mkdir(parents=True)
        (target/'episode_000000.gif').touch()
        with patch('control_panel.os.startfile') as launch:
            self.panel.open_output('gif');launch.assert_called_once_with(str(target))

    def test_no_run_and_missing_directory(self):
        self.panel.output_dir=None
        with patch('control_panel.messagebox.showinfo') as info:
            self.panel.open_output();info.assert_called_once()
        self.panel.output_dir=self.folder/'missing'
        with patch('control_panel.messagebox.showerror') as error:
            self.panel.open_output();error.assert_called_once()


if __name__=='__main__': unittest.main()
