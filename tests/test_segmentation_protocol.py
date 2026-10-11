import random,unittest
import numpy as np
from pycocotools import mask as coco_mask
from detection.segmentation_protocol import encode_rle,hemisphere_views,lighting,AZIMUTH_BANDS,LIGHT_MASKS

class ProtocolTests(unittest.TestCase):
 def test_exact_rle_holes_disconnected_and_empty(self):
  rng=np.random.default_rng(3)
  for mask in [np.zeros((30,40),bool),np.ones((30,40),bool),rng.random((30,40))>.7]:
   rle=encode_rle(mask);decoded=coco_mask.decode(coco_mask.frPyObjects(rle,*mask.shape))
   np.testing.assert_array_equal(mask,decoded)
 def test_disjoint_camera_and_lighting_support(self):
  layout=dict(grid_x=[.12,.52],grid_y=[-.6924,.3076])
  for split in ('train','valid','test'):
   config=dict(split=split,reserved_eyes=[[.32,-.1924,1.]],dp_exclusion_radius_m=.12)
   a=hemisphere_views(random.Random(42),layout,99,config)
   self.assertEqual(a,hemisphere_views(random.Random(42),layout,99,config))
   for v in a:
    lo,hi=AZIMUTH_BANDS[split];self.assertTrue(lo<=v['azimuth_deg']%60<=hi)
    self.assertGreaterEqual(v['dp_clearance_m'],.14)
   self.assertEqual(set(v['distance_band'] for v in a),{'close','medium','far'})
   for seed in range(100):
    light=lighting(random.Random(seed),layout,split)
    self.assertIn(light['enabled_mask'],LIGHT_MASKS[split]);self.assertEqual(len(light['lights']),4)
    for row in light['lights']:self.assertEqual(row['intensity']>0,row['enabled'])
  for x,y in [('train','valid'),('train','test'),('valid','test')]:
   self.assertFalse(set(LIGHT_MASKS[x])&set(LIGHT_MASKS[y]))

if __name__=='__main__':unittest.main()
