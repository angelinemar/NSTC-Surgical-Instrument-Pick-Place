"""Paired, held-out Isaac rollouts of the coupled and separate-gripper DP checkpoints."""
import argparse
import csv
import hashlib
import html
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INVENTORY = ROOT / 'datasets/Object datasets/trials/trial_03C_rfdetr_seg_syn_dp_balanced/dataset/source_episodes.json'
RUNS = ROOT / 'datasets/DP_training/runs'
CHECKPOINTS = {
    'v1_coupled': {
        skill: RUNS / f'dp_{skill}_all5_rgb448_b24_20261025_025052/checkpoint_best.pt'
        for skill in ('pick', 'place')
    },
    'v2_separate_gripper': {
        skill: RUNS / f'gripper_v2_{skill}_rgb448_b24_20261028_223111/checkpoint_best.pt'
        for skill in ('pick', 'place')
    },
}
CLASSES = ('kelly', 'love_retractor', 'scalpel', 'scalpel_type2', 'scissor')


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def selected_starts(per_class, repeats):
    inventory = json.loads(INVENTORY.read_text())
    starts = []
    for skill in ('pick', 'place'):
        for cls in CLASSES:
            matching = sorted((row for row in inventory if row['split'] == 'test'
                               and row['target'] == cls and f'/{skill}_policy/' in row['path']),
                              key=lambda row: row['path'])
            if len(matching) < per_class:
                raise ValueError(f'Only {len(matching)} held-out {skill}/{cls} starts')
            for index, row in enumerate(matching[:per_class]):
                for repeat in range(repeats):
                    starts.append(dict(skill=skill, instrument=cls, source=row['path'],
                                       repeat=repeat, seed=810000 + 10000 * (skill == 'place')
                                       + 1000 * CLASSES.index(cls) + 10 * index + repeat))
    return starts


def summarize(output, starts):
    rows = []
    for i, start in enumerate(starts):
        for model in CHECKPOINTS:
            path = output / f'{i:03d}_{start["skill"]}_{start["instrument"]}_r{start["repeat"]}_{model}' / 'result.json'
            if path.exists():
                row = json.loads(path.read_text())
                valid = bool(row.get(start['skill'] + '_started')) and not str(row.get('failure', '')).startswith('RESET_INVALID:')
                rows.append(dict(start_index=i, model=model, skill=start['skill'], instrument=start['instrument'],
                                 seed=start['seed'], source=start['source'], valid=valid,
                                 success=bool(row.get('lift_success' if start['skill'] == 'pick' else 'place_success')) if valid else False,
                                 failure=row.get('failure', ''), gif=row.get('gif', ''),
                                 steps=sum(1 for _ in (path.parent / 'actions.jsonl').open()) if (path.parent / 'actions.jsonl').exists() else 0))
    with (output / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('start_index', 'model', 'skill', 'instrument', 'seed', 'source', 'valid', 'success', 'failure', 'gif', 'steps'))
        writer.writeheader(); writer.writerows(rows)
    by_model = {}
    for model in CHECKPOINTS:
        by_model[model] = {}
        for skill in ('pick', 'place'):
            subset = [r for r in rows if r['model'] == model and r['skill'] == skill]
            valid = [r for r in subset if r['valid']]
            won = sum(r['success'] for r in valid)
            by_model[model][skill] = dict(planned=len([s for s in starts if s['skill'] == skill]),
                                          finished=len(subset), valid=len(valid), success=won,
                                          fail=len(valid) - won, invalid=len(subset) - len(valid),
                                          success_percent=100 * won / len(valid) if valid else None)
    paired = {}
    for skill in ('pick', 'place'):
        pairs = []
        for i, start in enumerate(starts):
            if start['skill'] != skill:
                continue
            old = next((r for r in rows if r['start_index'] == i and r['model'] == 'v1_coupled'), None)
            new = next((r for r in rows if r['start_index'] == i and r['model'] == 'v2_separate_gripper'), None)
            if old and new and old['valid'] and new['valid']:
                pairs.append((old, new))
        paired[skill] = dict(valid_pairs=len(pairs), both_success=sum(a['success'] and b['success'] for a, b in pairs),
                             old_only=sum(a['success'] and not b['success'] for a, b in pairs),
                             new_only=sum(b['success'] and not a['success'] for a, b in pairs),
                             neither=sum(not a['success'] and not b['success'] for a, b in pairs))
    result = dict(protocol='paired_held_out_skill_only_v1', planned_starts=len(starts),
                  planned_rollouts=2 * len(starts), finished_rollouts=len(rows),
                  complete=len(rows) == 2 * len(starts), models=by_model, paired=paired,
                  action_horizon=8, denoising_steps=100, timeout_steps=300,
                  note='Skill-only closed loop from identical held-out H5 starts and DDPM seeds; 20 ms/step. No FSM skill actions.')
    atomic_json(output / 'summary.json', result)
    gallery = ['<!doctype html><meta charset="utf-8"><title>Paired DP rollouts</title>',
               '<style>body{font:16px sans-serif;max-width:1100px;margin:2em auto;color:#17212b}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.5em}video,img{max-width:100%}</style>',
               '<h1>Coupled V1 vs separate-gripper V2</h1><p>Held-out test starts; same H5 and DDPM seed per pair. GIFs show real front and top camera frames. A missing GIF means that rollout has not completed.</p>',
               '<table><tr><th>Skill</th><th>Model</th><th>Success / valid</th><th>Rate</th></tr>']
    for skill in ('pick', 'place'):
        for model in CHECKPOINTS:
            stat = by_model[model][skill]
            rate = f'{stat["success_percent"]:.1f}%' if stat['success_percent'] is not None else '—'
            gallery.append(f'<tr><td>{skill}</td><td>{model}</td><td>{stat["success"]}/{stat["valid"]}</td><td>{rate}</td></tr>')
    gallery.append('</table>')
    for row in rows:
        if row['gif']:
            gif = Path(row['gif'])
            relative = gif.relative_to(output) if gif.is_relative_to(output) else gif
            gallery.append(f'<h2>{row["skill"].upper()} · {html.escape(row["instrument"])} · {html.escape(row["model"])} · {"SUCCESS" if row["success"] else "FAIL"}</h2><img src="{html.escape(str(relative))}" loading="lazy">')
    (output / 'GIF_INDEX.html').write_text('\n'.join(gallery))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=1, help='Held-out starts per class and skill (1–10)')
    parser.add_argument('--repeats', type=int, default=1, help='DDPM seeds per held-out start')
    parser.add_argument('--output', type=Path, default=ROOT / 'datasets/DP_evaluation/gripper_v2_vs_v1/paired_test_1_per_class')
    args = parser.parse_args()
    if not 1 <= args.per_class <= 10 or args.repeats < 1:
        parser.error('per-class must be 1–10; repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    starts = selected_starts(args.per_class, args.repeats)
    for pair in CHECKPOINTS.values():
        for checkpoint in pair.values():
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
    plan = dict(starts=starts, checkpoints={m: {s: str(p) for s, p in pair.items()} for m, pair in CHECKPOINTS.items()},
                checkpoint_sha256={m: {s: hashlib.sha256(p.read_bytes()).hexdigest() for s, p in pair.items()} for m, pair in CHECKPOINTS.items()},
                inventory_sha256=hashlib.sha256(INVENTORY.read_bytes()).hexdigest())
    frozen = args.output / 'plan.json'
    if frozen.exists() and json.loads(frozen.read_text()) != plan:
        raise ValueError('Frozen plan differs; choose a new output folder')
    atomic_json(frozen, plan)
    summarize(args.output, starts)
    for i, start in enumerate(starts):
        for model, pair in CHECKPOINTS.items():
            dest = args.output / f'{i:03d}_{start["skill"]}_{start["instrument"]}_r{start["repeat"]}_{model}'
            if (dest / 'result.json').exists():
                continue
            if dest.exists():
                raise RuntimeError(f'Incomplete rollout: {dest}; inspect before resuming')
            command = [sys.executable, '-m', 'evaluation.dp_trial', '--pick', str(pair['pick']), '--place', str(pair['place']),
                       '--skill', start['skill'], '--object', start['instrument'], '--mode', 'grid', '--seed', str(start['seed']),
                       '--snapshot', start['source'], '--output', str(dest), '--headless']
            print(f'[{i + 1}/{len(starts)}] {start["skill"]} {start["instrument"]} {model}', flush=True)
            with (args.output / f'{i:03d}_{model}.log').open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    process.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise RuntimeError(f'Simulator timeout: {dest}')
            if not (dest / 'result.json').exists():
                raise RuntimeError(f'Missing result: {dest}')
            row = json.loads((dest / 'result.json').read_text())
            failure = str(row.get('failure', ''))
            if failure and failure not in ('pick_timeout', 'place_timeout') and not failure.startswith('RESET_INVALID:'):
                raise RuntimeError(f'Infrastructure error in {dest}: {failure}')
            summary = summarize(args.output, starts)
            print(json.dumps(summary['models']), flush=True)
    print('Paired test complete:', args.output, flush=True)


if __name__ == '__main__':
    main()
