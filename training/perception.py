"""Independent RGB perception checkpoint; no proprioception or simulator inputs.

Semantic detector for the current one-instance-per-class scene contract.
Returns visible class boxes; multiple same-class objects need instance labels.
"""
import torch
from torch import nn
from torch.nn import functional as F


class PerceptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,32,5,2,2), nn.GroupNorm(8,32), nn.SiLU(),
            nn.Conv2d(32,64,3,2,1), nn.GroupNorm(8,64), nn.SiLU(),
            nn.Conv2d(64,64,3,2,1), nn.GroupNorm(8,64), nn.SiLU())
        self.segmentation = nn.Conv2d(64,8,1)

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError('Expected B,3,H,W RGB in [0,1]')
        return F.interpolate(self.segmentation(self.encoder(rgb)), rgb.shape[-2:],
                             mode='bilinear', align_corners=False)

    @torch.no_grad()
    def detect(self, rgb, confidence=.5, min_pixels=8):
        probabilities = self(rgb).softmax(1)
        scores, masks = probabilities.max(1)
        results = []
        for mask, score in zip(masks, scores):
            boxes = []
            for label in range(3,8):
                visible = (mask == label) & (score >= confidence)
                y,x = visible.nonzero(as_tuple=True)
                if len(x) >= min_pixels:
                    boxes.append(dict(semantic_id=label, category_id=label-2,
                        bbox_xywh=[int(x.min()),int(y.min()),int(x.max()-x.min()+1),int(y.max()-y.min()+1)],
                        confidence=float(score[visible].mean())))
            results.append(dict(mask=mask, detections=boxes))
        return results


def save_from_joint(joint, path, smoke_only):
    model = PerceptionModel()
    model.encoder.load_state_dict(joint.encoder.state_dict())
    model.segmentation.load_state_dict(joint.segmentation.state_dict())
    torch.save(dict(model=model.state_dict(), input_contract='rgb_only_v1',
                    smoke_only=smoke_only, ready_for_deployment=False,
                    source='joint_dp_auxiliary_segmentation'), path)


class PerceptionRuntime:
    def __init__(self, checkpoint, device='cpu', allow_smoke=False):
        data = torch.load(checkpoint, map_location=device, weights_only=True)
        if data.get('input_contract') != 'rgb_only_v1':
            raise ValueError('Incompatible perception checkpoint')
        if data.get('smoke_only', True) and not allow_smoke:
            raise ValueError('Smoke checkpoint is for computation testing only')
        self.model = PerceptionModel().to(device).eval()
        self.model.load_state_dict(data['model'])
        self.device = device

    def predict(self, rgb, confidence=.5):
        import numpy as np
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError('Expected H,W,3 uint8 RGB')
        value=torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1).float()[None].to(self.device)/255.
        result=self.model.detect(value,confidence=confidence)[0]
        result['mask']=result['mask'].cpu().numpy()
        return result
