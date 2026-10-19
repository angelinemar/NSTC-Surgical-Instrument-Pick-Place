import json,tempfile,unittest
from pathlib import Path
from training.segmentation_checkpoint import checkpoint_resolution
class ResolutionTests(unittest.TestCase):
 def test_export_without_resolution_uses_training_sidecar(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'checkpoint_best_total.pth';(p.parent/'trial_config.json').write_text(json.dumps({'resolution':768}))
   self.assertEqual(checkpoint_resolution(p),768)
   with self.assertRaises(ValueError):checkpoint_resolution(p,384)
 def test_missing_metadata_cannot_silently_default_384(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'model.pth'
   with self.assertRaises(ValueError):checkpoint_resolution(p)
   self.assertEqual(checkpoint_resolution(p,768),768)
