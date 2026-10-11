import unittest
import numpy as np
from training.evaluate_segmentation import matches,rle
from training.evaluate_dp_cameras import summarize

class SegmentationEvaluationTests(unittest.TestCase):
    def test_box_overlap_does_not_imply_mask_overlap(self):
        mask=np.zeros((10,10),np.uint8);mask[:,:5]=1
        images=[dict(id=1,width=10,height=10,group='far')]
        gt=[dict(id=1,image_id=1,category_id=1,bbox=[0,0,10,10],area=50,iscrowd=0,segmentation=rle(mask))]
        pred=[dict(image_id=1,category_id=1,bbox=[0,0,10,10],score=.9,segmentation=rle(1-mask))]
        box=summarize([matches(images,gt,pred,'bbox')['far']],.5)
        seg=summarize([matches(images,gt,pred,'segm')['far']],.5)
        self.assertAlmostEqual(box['F1'],100)
        self.assertEqual(seg['F1'],0)
    def test_empty_predictions_count_false_negatives(self):
        mask=np.ones((10,10),np.uint8)
        gt=[dict(id=1,image_id=1,category_id=1,bbox=[0,0,10,10],area=100,iscrowd=0,segmentation=rle(mask))]
        score=summarize([matches([dict(id=1,width=10,height=10,group='front')],gt,[],'segm')['front']],.5)
        self.assertEqual(score['FN'],1)
        self.assertEqual(score['recall'],0)
