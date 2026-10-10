import unittest
from detection.prepare_training import scene_splits

class SplitTests(unittest.TestCase):
    def test_scene_assignment_is_complete_disjoint_and_reproducible(self):
        a=scene_splits(500)
        self.assertEqual(a,scene_splits(500))
        self.assertEqual(set(a),set(range(500)))
        self.assertEqual([sum(s==split for s in a.values()) for split in ('train','valid','test')],[400,50,50])
        self.assertNotEqual(a,scene_splits(500,seed=1))

    def test_small_collection_rejected(self):
        with self.assertRaises(ValueError):scene_splits(3)
