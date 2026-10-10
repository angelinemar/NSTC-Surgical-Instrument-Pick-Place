import json
from pathlib import Path
import tempfile
import unittest
from detection.progress import read_collection

class ProgressTests(unittest.TestCase):
    def test_committed_progress_wins_over_stale_status_and_ignores_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'config.json').write_text(json.dumps({'contract':'p4_static_detection_v1'}))
            (root/'status.json').write_text(json.dumps(dict(goal=500,scenes=0)))
            scene=root/'train/scenes/scene_000000';scene.mkdir(parents=True)
            (scene/'scene.json').write_text(json.dumps(dict(scene_id=0,views=[dict(boxes=[{},{}])])) )
            (root/'.pending_scene_000001').mkdir()
            _,stats=read_collection(root)
            self.assertEqual((stats['scenes'],stats['images'],stats['annotations'],stats['goal']),(1,1,2,500))
            self.assertEqual(json.loads((root/'status.json').read_text())['scenes'],0)

    def test_missing_committed_scene_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'config.json').write_text(json.dumps({'contract':'p4_static_detection_v1'}))
            scene=root/'train/scenes/scene_000001';scene.mkdir(parents=True)
            (scene/'scene.json').write_text(json.dumps(dict(scene_id=1,views=[])))
            with self.assertRaises(ValueError):read_collection(root)
