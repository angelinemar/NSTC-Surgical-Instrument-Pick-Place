"""Train or smoke-test the sensor-only DP + perception pipeline."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
if __package__:
    from .sensor_policy import SensorDataset, SensorPolicy, decode_actions, fit_training_stats
else:
    from sensor_policy import SensorDataset, SensorPolicy, decode_actions, fit_training_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--skill', choices=['pick', 'place'], required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.steps < 1:
        p.error('steps must be positive')
    torch.set_num_threads(4)
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    args.output.mkdir(parents=True, exist_ok=False)
    stats = fit_training_stats(args.dataset / (args.skill + '.hdf5'))
    train = SensorDataset(args.dataset, args.skill, 'train', stats, stride=4 if args.smoke else 1)
    valid = SensorDataset(args.dataset, args.skill, 'valid', stats, stride=16)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = SensorPolicy().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    history = []
    iterator = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {k: v.to(args.device) for k, v in batch.items()}
        loss, dp, perception = model.losses(batch)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite gradient')
        optimizer.step()
        row = dict(step=step + 1, loss=float(loss), diffusion=float(dp), perception=float(perception))
        history.append(row)
        print(json.dumps(row), flush=True)
    model.eval()
    metrics = []
    confusion = torch.zeros(11, 11, dtype=torch.int64)
    with torch.no_grad():
        for index, batch in enumerate(DataLoader(valid, batch_size=1)):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss, dp, perception = model.losses(batch)
            _, logits = model.encode(batch['rgb'], batch['proprio'])
            pred = logits.argmax(2)
            confusion += torch.bincount((batch['semantic'] * 11 + pred).flatten().cpu(), minlength=121).reshape(11, 11)
            metrics.append([float(loss), float(dp), float(perception)])
            if index == 0:
                # Inference has no labels/actions/metadata arguments.
                sampled, predicted_mask = model.act(batch['rgb'], batch['proprio'], inference_steps=10,task_target=batch['task_target'])
                decoded = decode_actions(sampled, stats)
                if decoded.shape != (1, 16, 8) or not torch.isfinite(decoded).all():
                    raise RuntimeError('Invalid inference output')
            if args.smoke and index >= 1:
                break
    union = confusion.sum(0) + confusion.sum(1) - confusion.diag()
    iou = [float(confusion[i, i] / union[i]) if union[i] else None for i in range(11)]
    checkpoint = dict(model=model.state_dict(), stats=stats, skill=args.skill,
                      observation_horizon=2, prediction_horizon=16, control_dt_s=.02,
                      input_contract='p4_sensor_task_v2', smoke_only=args.smoke)
    checkpoint['controller_contract'] = json.loads((args.dataset / 'manifest.json').read_text())['controller_contract']
    torch.save(checkpoint, args.output / 'checkpoint.pt')
    # Independently loadable perception artifact even when trained jointly.
    if __package__:
        from .perception import save_from_joint
    else:
        from perception import save_from_joint
    save_from_joint(model, args.output / 'perception.pt', args.smoke)
    report = dict(smoke_only=args.smoke, training_steps=args.steps, history=history,
                  validation_batches=len(metrics), validation_loss=np.mean(metrics, axis=0).tolist(),
                  validation_iou_by_semantic_id=iou, confusion_matrix=confusion.tolist(),
                  inference_output_finite=True, physical_rollout_verified=False,
                  ready_for_deployment=False,
                  note='A smoke run checks computation only, not learned competence or rollout success.')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'history'}), flush=True)


if __name__ == '__main__':
    main()
