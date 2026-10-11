import contextlib
import io
import unittest

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from training.evaluate_dp_cameras import CLASSES, coco_records, summarize, instance_boxes
from training.export_detection import visible_boxes


class CameraMetricsTests(unittest.TestCase):
    def test_fast_boxes_preserve_disconnected_pixels_and_instance_identity(self):
        mask=np.zeros((20,20),np.uint16);instances=np.zeros_like(mask)
        instances[1:4,2:6]=1;instances[10:12,12:15]=1
        instances[5:8,4:9]=2;instances[0,19]=3
        mask[instances>0]=3
        mapping={str(i):dict(semantic_id=3) for i in (1,2,3)}
        self.assertEqual(instance_boxes(mask,instances,mapping),
            visible_boxes(mask,instances=instances,instance_classes=mapping))
        mask[1,2]=4
        with self.assertRaisesRegex(ValueError,'disagrees'):instance_boxes(mask,instances,mapping)

    def test_pooled_matches_equal_reference_coco_and_count_errors(self):
        images=[dict(id=i,width=448,height=448,camera='front') for i in (1,2,3)]
        annotations=[dict(id=i,image_id=i,category_id=1,bbox=[10,10,20,20],area=400,iscrowd=0) for i in (1,2)]
        predictions=[dict(image_id=1,category_id=1,bbox=[10,10,20,20],score=.95),
                     dict(image_id=1,category_id=1,bbox=[10,10,20,20],score=.85),
                     dict(image_id=3,category_id=1,bbox=[10,10,20,20],score=.75)]
        records=coco_records(images,annotations,predictions)
        result=summarize([records['front']],.5)
        self.assertEqual((result['TP'],result['FP'],result['FN']),(1,2,1))
        self.assertAlmostEqual(result['precision'],100/3)
        self.assertAlmostEqual(result['recall'],50)
        self.assertAlmostEqual(result['F1'],40)
        with contextlib.redirect_stdout(io.StringIO()):
            gt=COCO();gt.dataset=dict(info={},images=images,annotations=annotations,
                categories=[dict(id=i+1,name=n) for i,n in enumerate(CLASSES)]);gt.createIndex()
            e=COCOeval(gt,gt.loadRes(predictions),'bbox');e.evaluate();e.accumulate()
        precision=e.eval['precision'][:,:,:,0,-1]
        self.assertAlmostEqual(result['mAP_50_95'],100*precision[precision>=0].mean())
        self.assertEqual(summarize([records['front']],.99)['recall'],0)

    def test_missing_detections_and_empty_camera(self):
        images=[dict(id=1,width=448,height=448,camera='front')]
        annotations=[dict(id=1,image_id=1,category_id=1,bbox=[0,0,20,20],area=400,iscrowd=0)]
        records=coco_records(images,annotations,[])
        result=summarize([records['front']],.5)
        self.assertEqual(result['mAP_50_95'],0)
        self.assertEqual(result['FN'],1)
        self.assertIsNone(summarize([records['wrist']],.5)['mAP_50_95'])

if __name__=='__main__':unittest.main()
