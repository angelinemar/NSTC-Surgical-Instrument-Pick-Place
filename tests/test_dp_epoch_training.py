import unittest
from training.train_dp_epochs import stopping_state
class EarlyStoppingTests(unittest.TestCase):
 def test_plateau_and_real_improvement(self):
  best,bad,changed=stopping_state(.5,float('inf'),0,.01)
  self.assertEqual((best,bad,changed),(.5,0,True))
  for _ in range(15):best,bad,changed=stopping_state(.495,best,bad,.01)
  self.assertEqual((best,bad,changed),(.5,15,False))
  self.assertEqual(stopping_state(.48,best,bad,.01),(.48,0,True))
 def test_nonfinite_rejected(self):
  for value in [float('nan'),float('inf')]:
   with self.assertRaises(RuntimeError):stopping_state(value,.5,0,.01)
if __name__=='__main__':unittest.main()
