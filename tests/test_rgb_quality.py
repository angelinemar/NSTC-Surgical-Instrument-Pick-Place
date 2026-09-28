import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class RGBQualityTests(unittest.TestCase):
    def test_every_backend_exports_six_rgb_previews_without_resize(self):
        for path in (ROOT/'backends').glob('phase3_grid_split_*_recorder.py'):
            with self.subTest(backend=path.name), tempfile.TemporaryDirectory() as tmp:
                tree=ast.parse(path.read_text(encoding='utf-8-sig'))
                node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='export_segment_preview')
                ns=dict(os=os,json=json,np=np,Image=Image,omni=Mock(),
                        phase3_hide_all_debug_visuals=Mock(),draw_grid_preview_png=Mock(),
                        CAMERA_WIDTH=448,CAMERA_HEIGHT=448,CAMERA_EXPORT_STRIDE=1,
                        _safe_name=lambda x:x,_depth_to_vis_uint8=lambda x:x,
                        colorize_semantic=lambda x:x)
                exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
                rgb=np.zeros((448,448,3),np.uint8);rgb[::2,:,1]=255
                depth=np.zeros((448,448),np.uint8)
                rec=SimpleNamespace(stage_names=['LOWER_PRE'],front_rgb=[rgb],grip_b_rgb=[rgb],
                    front_depth=[depth],grip_b_depth=[depth],
                    extra_camera_rgb={name:[rgb] for name in ('cam_top','cam_left','cam_right','cam_tray')})
                ns['export_segment_preview'](rec,tmp,0,[0],{})
                folders=('front_rgb','grip_b_rgb','cam_top_rgb','cam_left_rgb','cam_right_rgb','cam_tray_rgb')
                for name in folders:
                    file=next((Path(tmp)/'episode_000000_preview'/name).glob('*.png'))
                    with Image.open(file) as image:
                        np.testing.assert_array_equal(np.asarray(image),rgb)

    def test_dp_downsamples_448_while_raw_is_unchanged(self):
        import h5py
        from training.export_sensor_only import copy_policy_images
        with tempfile.TemporaryDirectory() as tmp, h5py.File(Path(tmp)/'test.h5','w') as h:
            pixels=np.zeros((1,448,448,3),np.uint8);pixels[:,::2,:,1]=255
            raw=h.create_dataset('raw',data=pixels)
            out=h.create_group('dp')
            copy_policy_images(raw,out,'rgb')
            self.assertEqual(out['rgb'].shape,(1,224,224,3))
            np.testing.assert_array_equal(raw[:],pixels)
            self.assertGreater(float(out['rgb'][0,:,:,1].mean()),100.)


if __name__=='__main__':
    unittest.main()
