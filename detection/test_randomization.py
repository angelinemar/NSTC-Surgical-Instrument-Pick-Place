"""CPU tests for reproducible wide profiles and conservative camera framing."""
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from detection.randomization import load_profile,sample_wide_lighting
from detection.dataset import camera_views

LAYOUT=dict(grid_x=[.12,.52],grid_y=[-.69,.31],tray_xy=[.15,-.89])

class WideTests(unittest.TestCase):
    def test_light_sampling_reproducible_and_bounded(self):
        profile=load_profile()
        for seed in range(100):
            a=sample_wide_lighting(seed,LAYOUT,profile)
            self.assertEqual(a,sample_wide_lighting(seed,LAYOUT,profile))
            self.assertTrue(100<=a['ambient_intensity']<=900)
            for light in a['lights']:
                self.assertTrue(800<=light['intensity']<=12000)
                self.assertTrue(.65<=light['position'][2]<=2.4)
                self.assertTrue(all(.55<=c<=1 for c in light['color']))
        self.assertNotEqual(sample_wide_lighting(1,LAYOUT,profile),sample_wide_lighting(2,LAYOUT,profile))

    def test_override_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'wide.json'
            for bad in ({'unknown':1},{'light_height':[3,1]},{'camera_distance':[.5,1]}, {'aim_jitter':float('nan')},{'light_color':[0,2]}):
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): load_profile(path)
            path.write_text(json.dumps({'light_height':[1,2]}))
            self.assertEqual(load_profile(path)['light_height'],[1,2])
            self.assertEqual(load_profile()['light_height'],[.65,2.4])

    def test_wide_cameras_keep_framing_distance(self):
        original=camera_views(random.Random(1),LAYOUT,32)
        views=camera_views(random.Random(1),LAYOUT,32,load_profile())
        base=math.dist(original[-1]['eye'],original[-1]['target'])
        self.assertEqual(len(views),33)
        self.assertEqual(len({tuple(v['eye']) for v in views}),33)
        for view in views[:-1]:
            distance=math.dist(view['eye'],view['target'])
            self.assertGreaterEqual(distance,base-1e-5)
            self.assertLessEqual(distance,base*1.2)
            elevation=math.degrees(math.asin((view['eye'][2]-view['target'][2])/distance))
            self.assertTrue(25<=elevation<=75)

if __name__=='__main__': unittest.main()
