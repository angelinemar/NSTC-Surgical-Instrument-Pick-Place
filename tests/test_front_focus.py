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
    def test_camera_installer_loads_active_layout_not_defaults(self):
        import phase3_recorder_camera_patch as camera_patch
        active=json.loads((ROOT/'env/camera_layout.json').read_text())['cameras']
        installed=camera_patch.phase3_load_camera_tuning().PHASE3_CAMERAS
        self.assertEqual(list(installed['camera']['pos']),active['cam_front']['pos'])
        self.assertEqual(installed['camera']['focal_length'],active['cam_front']['focal_length'])
        self.assertEqual(list(installed['cam_tray']['pos']),active['cam_tray']['pos'])

    def test_full_task_surface_fits_four_static_views(self):
        with patch.dict(os.environ,{'P4_RESUME_CAMERA_MANIFEST':'','P4_CAMERA_FOV_SCALE':'0.75'}):
            importlib.reload(tuning)
            layout=json.loads((ROOT/'env/scene_layout.json').read_text())
            length,width=layout['tray_dimensions_local_xy']
            # Tray yaw=90 degrees, so its local width becomes world X and its
            # local length becomes world Y. Include instrument footprint and
            # lift-height margins around the complete spawn + tray envelope.
            tx,ty=layout['tray_xy']
            xs=(min(layout['grid_x'][0]-.05,tx-width/2-.025),
                max(layout['grid_x'][1]+.05,tx+width/2+.025))
            ys=(min(layout['grid_y'][0]-.05,ty-length/2-.03),
                max(layout['grid_y'][1]+.08,ty+length/2+.03))
            points=np.array([[x,y,z] for x in xs for y in ys for z in (0,.18)])
            for name in ('camera','cam_top','cam_left','cam_right'):
                c=tuning.PHASE3_CAMERAS[name]
                q=c['rot']; rotation=Rotation.from_quat([*q[1:],q[0]]).as_matrix()
                local=(points-np.array(c['pos']))@rotation
                xy=local[:,:2]/local[:,2:]*c['focal_length']/c['horizontal_aperture']
                self.assertTrue(np.all(local[:,2]>0),name)
                self.assertLess(float(np.abs(xy).max()),.48,name)

    def test_tray_fills_but_does_not_clip_tray_camera(self):
        with patch.dict(os.environ,{'P4_RESUME_CAMERA_MANIFEST':'','P4_CAMERA_FOV_SCALE':'0.75'}):
            importlib.reload(tuning)
            layout=json.loads((ROOT/'env/scene_layout.json').read_text())
            c=tuning.PHASE3_CAMERAS['cam_tray']; q=c['rot']
            rotation=Rotation.from_quat([*q[1:],q[0]]).as_matrix()
            length,width=layout['tray_dimensions_local_xy']; tx,ty=layout['tray_xy']
            points=np.array([[x,y,z] for x in (tx-width/2-.02,tx+width/2+.02)
                                      for y in (ty-length/2-.02,ty+length/2+.02)
                                      for z in (0,.10)])
            local=(points-np.array(c['pos']))@rotation
            xy=local[:,:2]/local[:,2:]*c['focal_length']/c['horizontal_aperture']
            self.assertTrue(np.all(local[:,2]>0))
            self.assertGreater(float(np.abs(xy).max()),.35)
            self.assertLess(float(np.abs(xy).max()),.48)

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
