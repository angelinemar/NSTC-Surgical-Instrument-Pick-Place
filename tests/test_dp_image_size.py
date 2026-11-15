import tempfile, unittest
from pathlib import Path
import h5py
import numpy as np
from training.export_sensor_only import copy_policy_images

class ResolutionTests(unittest.TestCase):
 def test_native_copy_and_downsample(self):
  from PIL import Image
  with tempfile.TemporaryDirectory() as d, h5py.File(Path(d)/'source.h5','w') as src, h5py.File(Path(d)/'out.h5','w') as out:
   rgb=np.random.default_rng(42).integers(0,256,(1,448,448,3),dtype=np.uint8)
   source=src.create_dataset('rgb',data=rgb)
   copy_policy_images(source,out,'native',image_size=448)
   np.testing.assert_array_equal(out['native'][:],rgb)
   copy_policy_images(source,out,'small',image_size=224)
   np.testing.assert_array_equal(out['small'][0],np.asarray(Image.fromarray(rgb[0]).resize((224,224),Image.Resampling.LANCZOS)))
   with self.assertRaises(ValueError):copy_policy_images(out['small'],out,'upscaled',image_size=448)
if __name__=='__main__':unittest.main()
