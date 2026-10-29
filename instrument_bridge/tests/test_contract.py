import unittest
import numpy as np
from instrument_bridge.api import validate_rgb
from instrument_bridge.evaluate import compare,table_spawn_counts

class ContractTest(unittest.TestCase):
 def test_visible_instance_matching_and_counts(self):
  mask=np.zeros((20,20),bool);mask[2:8,2:8]=True
  pred=[dict(class_name='kelly',score=.9,bbox_xyxy=[2,2,8,8])]
  result=compare(pred,[mask],[dict(class_name='kelly',mask=mask)])
  self.assertEqual(result['overall']['TP'],1)
  self.assertEqual(result['per_class']['kelly']['count_error'],0)
 def test_spawn_is_separate_from_visible(self):
  attrs=dict(scene_clutter='{"instances":[{"object":"scalpel","region":"table"},{"object":"scissor","region":"tray"}]}',target_object='kelly')
  result=table_spawn_counts(attrs,include_target=True)
  self.assertEqual(result['scalpel'],1);self.assertEqual(result['scissor'],0);self.assertEqual(result['kelly'],1)
 def test_sensor_contract(self):
  with self.assertRaises(ValueError):validate_rgb(np.zeros((10,10,3),np.float32))
if __name__=='__main__':unittest.main()
