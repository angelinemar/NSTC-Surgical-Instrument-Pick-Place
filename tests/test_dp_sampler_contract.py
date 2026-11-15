import unittest
from training.sensor_policy import SensorPolicy

class SamplerContractTests(unittest.TestCase):
 def test_unsafe_skipping_is_rejected_before_inference(self):
  model=SensorPolicy()
  for steps in (2,20,50):
   with self.assertRaisesRegex(ValueError,'all training timesteps'):
    model.act(None,None,inference_steps=steps)
if __name__=='__main__':unittest.main()
