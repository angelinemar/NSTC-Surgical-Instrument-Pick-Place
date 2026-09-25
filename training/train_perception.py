"""Train independent RGB semantic detection from export_detection output."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
try:
    from .perception import PerceptionModel
    from .detection_metrics import ap50
    from .export_detection import visible_boxes
except ImportError:
    from perception import PerceptionModel
    from detection_metrics import ap50
    from export_detection import visible_boxes


class Frames(Dataset):
    def __init__(self, root, split, size):
        self.root, self.size = Path(root)/split, size
        self.rows = json.loads((self.root/'annotations.json').read_text())['images']
        if not self.rows:
            raise ValueError('Empty split: ' + split)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root/row['file_name']) as image:
            rgb = np.array(image.convert('RGB').resize((self.size,self.size), Image.Resampling.LANCZOS))
        with Image.open(self.root/row['mask_file']) as image:
            mask = np.array(image.resize((self.size,self.size), Image.Resampling.NEAREST))
        if not np.isin(mask, np.arange(11)).all():
            raise ValueError('Unknown semantic label')
        return torch.from_numpy(rgb).permute(2,0,1).float()/255., torch.from_numpy(mask.astype(np.int64))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--size', type=int, choices=(224,448), default=448)
    p.add_argument('--device', default='cpu')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('steps and batch-size must be positive')
    manifest = json.loads((args.dataset/'manifest.json').read_text())
    if not manifest.get('export_complete') or manifest.get('contract') != 'p4_detection_v1':
        raise ValueError('Incomplete or incompatible detector export')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(17)
    torch.set_num_threads(4)
    model = PerceptionModel().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loader = DataLoader(Frames(args.dataset,'train',args.size), batch_size=args.batch_size, shuffle=True)
    iterator = iter(loader)
    weights = torch.tensor([.1,.3,.5,1.,1.,1.,1.,1.,.2,.1,.1],device=args.device)
    duplicates = any(s.get('has_duplicate_instances') for s in manifest['sources'])
    for step in range(args.steps):
        try:
            rgb, mask = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            rgb, mask = next(iterator)
        loss = F.cross_entropy(model(rgb.to(args.device)),mask.to(args.device),weight=weights)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite perception loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if step % 25 == 0 or step == args.steps-1:
            print(json.dumps(dict(step=step+1,loss=float(loss))),flush=True)
    model.eval()
    report = dict(smoke_only=args.smoke, steps=args.steps, splits={}, ready_for_deployment=False,
                  limitation='Visible class boxes, one instance per class. AP50 is not COCO AP50:95; smoke metrics use only one batch.')
    with torch.no_grad():
        for split in ('valid','test'):
            confusion = torch.zeros(11,11,dtype=torch.int64)
            batches = 0
            detections = []
            for rgb, mask in DataLoader(Frames(args.dataset,split,args.size),batch_size=args.batch_size):
                prediction = model(rgb.to(args.device)).argmax(1).cpu()
                confusion += torch.bincount((mask*11+prediction).flatten(),minlength=121).reshape(11,11)
                if not duplicates:
                    for predicted,truth in zip(model.detect(rgb.to(args.device),confidence=0.),mask):
                        detections.append(dict(predictions=predicted['detections'],truth=visible_boxes(truth.numpy())))
                batches += 1
                if args.smoke:
                    break
            union = confusion.sum(0)+confusion.sum(1)-confusion.diag()
            report['splits'][split] = dict(batches=batches, complete=not args.smoke,
                iou=[float(confusion[i,i]/union[i]) if union[i] else None for i in range(11)],
                detection=({'unavailable':'Semantic head cannot separate same-class instances; train an instance-aware detector using the COCO boxes.'} if duplicates else ap50(detections)))
    torch.save(dict(model=model.cpu().state_dict(),input_contract='rgb_only_v1',
        smoke_only=args.smoke,ready_for_deployment=False,size=args.size,
        dataset_manifest_sha256=hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest()),args.output/'perception.pt')
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
