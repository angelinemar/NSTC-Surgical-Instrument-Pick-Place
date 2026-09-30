import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('preview_export', ROOT/'scripts/tools/export_preview_png.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PreviewExportTests(unittest.TestCase):
    def test_all_cameras_native_png_and_alias_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source'; source.mkdir()
            path=source/'episode.h5'
            rgb=np.full((2,8,10,3),123,dtype=np.uint8)
            with h5py.File(path,'w') as h:
                h.attrs['complete']=True
                obs=h.create_group('observations')
                for camera in ('front','wrist','cam_top','cam_left','cam_right','cam_tray'):
                    obs.create_dataset(camera+'_rgb',data=rgb)
                    obs.create_dataset(camera+'_depth',data=np.ones((2,8,10),dtype=np.float32))
                    key='grip_b_semantic' if camera=='wrist' else camera+'_semantic'
                    obs.create_dataset(key,data=np.full((2,8,10),8,dtype=np.uint16))
                obs['images_front']=obs['front_rgb']
            original=path.read_bytes()
            report=module.export(source,root/'pngs')
            self.assertEqual(report['png_count'],36)
            self.assertEqual(path.read_bytes(),original)
            frame=Image.open(root/'pngs/episode/front_rgb/frame_000000.png')
            self.assertEqual(frame.size,(10,8))
            np.testing.assert_array_equal(np.array(frame),rgb[0])
            mask=Image.open(root/'pngs/episode/wrist_semantic/frame_000000.png')
            self.assertTrue(np.all(np.array(mask)==8))
            self.assertEqual(module.export(source,root/'sampled',2)['png_count'],18)
            with self.assertRaises(FileExistsError): module.export(source,root/'pngs')
            with self.assertRaises(ValueError): module.export(source,source/'nested')

    def test_incomplete_file_is_not_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source'; source.mkdir()
            with h5py.File(source/'episode.h5','w') as h:
                h.attrs['complete']=False
                h.create_group('observations').create_dataset('front_rgb',data=np.zeros((1,4,4,3),dtype=np.uint8))
            with self.assertRaises(ValueError): module.export(source,root/'pngs')
            self.assertFalse((root/'pngs/export_complete.json').exists())

    def test_selected_skill_episode_middle_frame_and_metadata_colors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source'; source.mkdir()
            for skill in ('pick','place'):
                folder=source/f'{skill}_policy/scalpel'; folder.mkdir(parents=True)
                with h5py.File(folder/'episode_000003.h5','w') as h:
                    h.attrs['semantic_class_ids']='{"robot": 42}'
                    obs=h.create_group('observations')
                    obs.create_dataset('front_rgb',data=np.zeros((3,4,4,3),dtype=np.uint8))
                    obs.create_dataset('front_semantic',data=np.full((3,4,4),42,dtype=np.uint16))
            selected=['place_policy/scalpel/episode_000003.h5']
            report=module.export(source,root/'pngs',selected=selected,frame_mode='middle',colored=True)
            self.assertEqual(report['png_count'],3)
            self.assertEqual(len(report['files']),1)
            self.assertFalse((root/'pngs/pick_policy').exists())
            folder=root/'pngs/place_policy/scalpel/episode_000003'
            with Image.open(folder/'front_semantic_color/frame_000001.png') as image:
                self.assertEqual(tuple(np.array(image)[0,0]),module.COLORS['robot'])
            self.assertFalse((folder/'front_rgb/frame_000000.png').exists())
            with self.assertRaises(ValueError):
                module.export(source,root/'bad',selected=['../escape.h5'])

    def test_all_five_backends_stop_automatic_preview_calls(self):
        paths=list((ROOT/'backends').glob('phase3_grid_split_*_recorder.py'))
        self.assertEqual(len(paths),5)
        for path in paths:
            tree=ast.parse(path.read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node,ast.Call):
                    name=getattr(node.func,'id',getattr(node.func,'attr',''))
                    self.assertNotIn(name,('_export_preview_folders','export_segment_preview'),str(path))


if __name__=='__main__': unittest.main()
