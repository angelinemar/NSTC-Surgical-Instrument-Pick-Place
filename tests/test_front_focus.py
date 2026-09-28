import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
import phase3_camera_tuning as tuning

ROOT=Path(__file__).resolve().parents[1]

class FrontFocusTests(unittest.TestCase):
    def test_tabletop_work_grid_fits_native_front(self):
        with patch.dict(os.environ,{'P4_RESUME_CAMERA_MANIFEST':'','P4_CAMERA_FOV_SCALE':'0.75'}):
            importlib.reload(tuning)
            c=tuning.PHASE3_CAMERAS['camera']
            q=c['rot']; rotation=Rotation.from_quat([*q[1:],q[0]]).as_matrix()
            layout=json.loads((ROOT/'env/scene_layout.json').read_text())
            points=np.array([[x,y,z] for x in layout['grid_x'] for y in layout['grid_y'] for z in (0,.10)])
            local=(points-np.array(c['pos']))@rotation
            xy=local[:,:2]/local[:,2:]*c['focal_length']/c['horizontal_aperture']
            self.assertTrue(np.all(local[:,2]>0))
            self.assertLess(float(np.abs(xy).max()),.5)

    def test_resume_preserves_original_front_pose_and_focal_length(self):
        original=json.loads((ROOT/'env/camera_layout.json').read_text())
        original['cameras']['cam_front']['pos']=[1.5,0,1.1]
        original['cameras']['cam_front'].pop('focal_length')
        with tempfile.TemporaryDirectory() as tmp:
            manifest=Path(tmp)/'scene_manifest.json'
            manifest.write_text(json.dumps({'camera_layout':original}))
            with patch.dict(os.environ,{'P4_RESUME_CAMERA_MANIFEST':str(manifest)}):
                importlib.reload(tuning)
                self.assertEqual(tuning.PHASE3_CAMERAS['camera']['pos'],(1.5,0,1.1))
                self.assertEqual(tuning.PHASE3_CAMERAS['camera']['focal_length'],16.)
        importlib.reload(tuning)

if __name__=='__main__': unittest.main()
