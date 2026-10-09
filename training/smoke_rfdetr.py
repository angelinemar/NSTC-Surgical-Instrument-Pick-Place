"""One CPU RF-DETR Nano epoch on a small exported dataset, with held-out evaluation."""
import argparse
import json
import os
import sys
from pathlib import Path
import torch
from rfdetr import RFDETRNano
from rfdetr_pipeline import validate_dataset, _jsonable

def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    audit = validate_dataset(dataset)
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    model = RFDETRNano(device='cpu')
    config = dict(dataset_dir=str(dataset), dataset_file='roboflow', device='cpu',
                  epochs=1, batch_size=1, grad_accum_steps=1, num_workers=0,
                  multi_scale=False, output_dir=str(output), tensorboard=False,
                  wandb=False, run_test=False, seed=42)
    (output/'smoke_config.json').write_text(json.dumps(config, indent=2))
    model.train(**config)
    checkpoint = output/'checkpoint_best_total.pth'
    if not checkpoint.is_file():
        raise RuntimeError('Training did not produce the expected best checkpoint')
    best = RFDETRNano(device='cpu', pretrain_weights=str(checkpoint))
    metrics = best.evaluate(dataset_dir=str(dataset), dataset_file='roboflow',
                            device='cpu', split='test', batch_size=1, num_workers=0,
                            output_dir=str(output/'test'), tensorboard=False, wandb=False)
    result = dict(smoke_only=True, training_complete=True, deployment_ready=False,
                  dataset_audit=audit, test_metrics=_jsonable(metrics), checkpoint=str(checkpoint))
    (output/'p4_rfdetr_result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)

if __name__ == '__main__':
    main()
