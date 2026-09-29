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

    def test_directional_views_cover_spawn_and_preserve_distinct_directions(self):
        with patch.dict(os.environ,{'P4_RESUME_CAMERA_MANIFEST':'','P4_CAMERA_FOV_SCALE':'0.75'}):
            importlib.reload(tuning)
            layout=json.loads((ROOT/'env/scene_layout.json').read_text())
            # Overlapping side-view regions jointly cover all spawn cells.
            regions={'camera':(-.5224,.1376), 'cam_left':(-.2124,.3076),
                     'cam_right':(-.6924,-.1724)}
            self.assertLessEqual(regions['cam_right'][0],layout['grid_y'][0])
            self.assertGreaterEqual(regions['cam_left'][1],layout['grid_y'][1])
            self.assertGreater(regions['cam_right'][1],regions['cam_left'][0])
            directions={}
            for name in ('camera','cam_left','cam_right'):
                ys=regions[name]
                points=np.array([[x,y,z] for x in layout['grid_x']
                                 for y in ys for z in (0,.16)])
                c=tuning.PHASE3_CAMERAS[name]
                q=c['rot']; rotation=Rotation.from_quat([*q[1:],q[0]]).as_matrix()
                directions[name]=rotation[:,2]
                local=(points-np.array(c['pos']))@rotation
                xy=local[:,:2]/local[:,2:]*c['focal_length']/c['horizontal_aperture']
                self.assertTrue(np.all(local[:,2]>0),name)
                self.assertGreater(float(np.abs(xy).max()),.34,name)
                self.assertLess(float(np.abs(xy).max()),.48,name)
                self.assertGreater(np.linalg.norm(rotation[:2,2]),.5,name)
            self.assertLess(directions['camera'][0],-.5)
            self.assertLess(directions['cam_left'][1],-.5)
            self.assertGreater(directions['cam_right'][1],.5)

    def test_top_is_vertical_and_covers_upper_cells(self):
        c=tuning.PHASE3_CAMERAS['cam_top'];q=c['rot']
        rotation=Rotation.from_quat([*q[1:],q[0]]).as_matrix()
        np.testing.assert_allclose(rotation[:,2],[0,0,-1],atol=1e-6)
        points=np.array([[x,y,z] for x in (.12,.52)
                         for y in (-.1924,.3076) for z in (0,.12)])
        local=(points-np.array(c['pos']))@rotation
        self.assertLess(float(np.abs(local[:,:2]/local[:,2:]*c['focal_length']/c['horizontal_aperture']).max()),.48)

    def test_tray_fills_but_does_not_clip_tray_camera(self):
        with patch.dict(os.environ,{'P4_RESUME_CAMERA_MANIFEST':'','P4_CAMERA_FOV_SCALE':'0.75'}):
            importlib.reload(tuning)
            layout=json.loads((ROOT/'env/scene_layout.json').read_text())
            c=tuning.PHASE3_CAMERAS['cam_tray']; q=c['rot']
            rotation=Rotation.from_quat([*q[1:],q[0]]).as_matrix()
            length,width=layout['tray_dimensions_local_xy']; tx,ty=layout['tray_xy']
            points=np.array([[x,y,z] for x in (tx-width/2-.006,tx+width/2+.006)
                                      for y in (ty-length/2-.006,ty+length/2+.006)
                                      for z in (0,.06)])
            local=(points-np.array(c['pos']))@rotation
            xy=local[:,:2]/local[:,2:]*c['focal_length']/c['horizontal_aperture']
            self.assertTrue(np.all(local[:,2]>0))
            self.assertGreater(float(np.abs(xy).max()),.45)
            self.assertLess(float(np.abs(xy).max()),.49)

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
