"""Clear live status for the matched 400-rollout trained-DP benchmark."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from .workspace_root import workspace_root
except ImportError:
    from workspace_root import workspace_root
ROOT = workspace_root()
BASE = ROOT / 'datasets/DP_evaluation/target_relative_individual_vs_joint_100_20261103'
CLASSES = ('scalpel', 'scissor', 'love_retractor', 'kelly', 'scalpel_type2')
GROUPS = (('individual', 'pick'), ('joint', 'pick'),
          ('individual', 'place'), ('joint', 'place'))
LABELS = {('individual', 'pick'): 'Individual Pick',
          ('joint', 'pick'): 'Joint Pick',
          ('individual', 'place'): 'Individual Place',
          ('joint', 'place'): 'Joint Place'}


def summaries() -> dict:
    result = {}
    for key in GROUPS:
        path = BASE / key[0] / key[1] / 'summary.json'
        try:
            result[key] = json.loads(path.read_text()) if path.exists() else {}
        except json.JSONDecodeError:
            result[key] = {}
    return result


def last_log_event(path: Path) -> str:
    with path.open('rb') as handle:
        handle.seek(max(0, path.stat().st_size - 12000))
        lines = handle.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        if '[CONTROL]' in line:
            match = re.search(r"'step':\s*(\d+)", line)
            if match:
                return f'policy step {int(match.group(1)):>3}/300'
            if 'RF-DETR B' in line:
                return 'RF-DETR target pose'
            if 'SUCCESS' in line or 'FAILED' in line:
                return 'saving result/GIF'
            if 'snapshot' in line.lower() or 'reset' in line.lower():
                return 'restoring scene'
    return 'starting simulator'


def active_trials(now: float) -> list[str]:
    candidates = []
    for variant, skill in GROUPS:
        folder = BASE / variant / skill
        if not folder.exists():
            continue
        for log in folder.glob('trial_*.log'):
            run = folder / log.stem
            if (run / 'result.json').exists():
                continue
            age = now - log.stat().st_mtime
            if age > 180:
                continue
            match = re.match(r'trial_(\d+)', log.stem)
            if not match:
                continue
            candidates.append((log.stat().st_mtime,
                f'{LABELS[(variant, skill)]:17} #{int(match.group(1)):03d}  '
                f'{last_log_event(log)}  (log {int(age)}s ago)'))
    return [line for _, line in sorted(candidates, reverse=True)[:4]]


def short_reason(value: object) -> str:
    reason = str(value or 'physical gate passed')
    if 'estimate_scissor_pose.py' in reason:
        return 'RF-DETR pose failed (see pose_estimator.log)'
    if reason == 'pick_timeout':
        return 'Pick timeout (300 steps)'
    if reason == 'place_timeout':
        return 'Place timeout (300 steps)'
    if 'RESET_INVALID' in reason:
        return 'Scene reset invalid'
    return reason if len(reason) <= 55 else reason[:52] + '...'


def observed_gate(row: dict, key: tuple[str, str]) -> str:
    """Describe a measured failure symptom; do not claim a causal diagnosis."""
    reason = str(row.get('failure') or '')
    run = Path(row['result_file']).parent
    if 'estimate_scissor_pose.py' in reason:
        path = run / 'pose_estimator.log'
        if path.exists():
            tail = path.read_text(errors='replace').splitlines()[-5:]
            if any('No target candidate has enough valid depth points' in line for line in tail):
                return 'RF-DETR target mask/depth unusable'
        return 'RF-DETR pose estimate failed'
    if reason not in ('pick_timeout', 'place_timeout'):
        return short_reason(reason)
    path = run / 'actions.jsonl'
    if not path.exists():
        return short_reason(reason)
    try:
        actions = [json.loads(line) for line in path.read_text().splitlines()]
    except (OSError, json.JSONDecodeError):
        return short_reason(reason)
    if not actions:
        return short_reason(reason)
    if key[1] == 'pick':
        close_sent = any(action['action'][7] < 0 for action in actions)
        initial_z = actions[0]['object_center_for_scoring'][2]
        rise_mm = max(action['object_center_for_scoring'][2] for action in actions)
        rise_mm = max(0, round((rise_mm - initial_z) * 1000))
        if not close_sent:
            return f'no CLOSE; max lift {rise_mm} mm'
        if rise_mm < 100:
            return f'CLOSE sent; max lift {rise_mm} mm (<100)'
        return f'max lift {rise_mm} mm; stable-hold gate failed'
    last = actions[-1].get('place_components') or {}
    if not last:
        return 'Place timeout; no final gate measurements'
    if not last.get('inside_slot'):
        return 'last state outside assigned tray slot'
    center_mm = round(last.get('center_error_m', 0) * 1000, 1)
    if center_mm > 8:
        return f'last center error {center_mm} mm (>8)'
    if last.get('axis_error_deg', 0) > 8:
        return f'last axis error {last["axis_error_deg"]:.1f}° (>8°)'
    if not last.get('fingers_open'):
        return 'fingers not open at final state'
    if not last.get('ee_clear'):
        return 'TCP not clear of placed tool'
    if not last.get('settled'):
        return 'tool not settled at final state'
    return 'strict Place gate not stable for 12 steps'


def frame(compact: bool = False) -> str:
    data = summaries()
    done = sum(item.get('completed', 0) for item in data.values())
    finished = done == 400 and all(item.get('complete') for item in data.values())
    now = time.time()
    progress_log = BASE / 'progress.log'
    pause_file = BASE / 'PAUSE_NOTE_20261104.json'
    last_progress = progress_log.stat().st_mtime if progress_log.exists() else 0
    paused = (not finished and pause_file.exists()
              and pause_file.stat().st_mtime > last_progress)
    stale = not finished and not paused and last_progress and now - last_progress > 3600
    state = ('COMPLETE' if finished else 'PAUSED' if paused else
             'NO RECENT PROGRESS' if stale else 'RUNNING / CHECKING')
    lines = [f'DP BENCHMARK  |  {datetime.now():%Y-%m-%d %H:%M:%S}  |  '
             f'{state}',
             f'{done}/400 completed • 100 Pick + 100 Place per approach • 5 instruments']
    if paused:
        note = json.loads(pause_file.read_text())
        lines.append('Paused: ' + note.get('reason', 'see pause note'))
        lines.append('Resume: ' + note.get('resume_command', 'see benchmark documentation'))
    elif stale:
        lines.append('Last saved result: ' + datetime.fromtimestamp(last_progress).strftime('%Y-%m-%d %H:%M:%S'))
        lines.append('Check the runner process and trial logs before resuming.')
    if not compact:
        lines += ['Individual = per instrument | Joint = one model for all five',
                  'Trained checkpoints • RF-DETR B live target pose for Pick', '']
    lines.append(f'{"Approach / skill":18} {"Done":>8} {"Success":>9} {"Invalid":>9} {"So far":>9}')
    for key in GROUPS:
        row = data[key]
        n = row.get('completed', 0)
        success = row.get('successes', 0)
        invalid = row.get('technical_invalid', 0)
        rate = f'{success/n:.1%}' if n else '—'
        lines.append(f'{LABELS[key]:18} {n:>4}/100 {success:>9} {invalid:>9} {rate:>9}')
    reasons = Counter()
    for key in GROUPS:
        for row in data[key].get('trials', []):
            if row['success']:
                continue
            failure = str(row.get('failure') or '')
            if row['technical_invalid']:
                reasons['invalid reset'] += 1
            elif 'estimate_scissor_pose.py' in failure:
                reasons['pose'] += 1
            elif failure == 'pick_timeout':
                reasons['pick timeout'] += 1
            elif failure == 'place_timeout':
                reasons['place timeout'] += 1
            else:
                reasons['other'] += 1
    lines.append('Failures: ' + ' | '.join(f'{title} {reasons[key]}' for title,key in
                 (('Pick timeout','pick timeout'), ('Place timeout','place timeout'),
                  ('Pose','pose'), ('Invalid','invalid reset'), ('Other','other'))))
    if not compact:
        lines += ['', 'Per instrument: successes / completed (target: 20 in every column)']
    else:
        lines += ['Per instrument (success/completed; target 20 each)']
    lines += [f'{"Instrument":17} {"Ind Pick":>10} {"Joint Pick":>11} '
              f'{"Ind Place":>11} {"Joint Place":>12}']
    for name in CLASSES:
        cells = []
        for key in GROUPS:
            class_row = data[key].get('per_class', {}).get(name, {})
            cells.append(f"{class_row.get('successes',0)}/{class_row.get('completed',0)}")
        lines.append(f'{name:17} {cells[0]:>10} {cells[1]:>11} {cells[2]:>11} {cells[3]:>12}')
    running = active_trials(now)
    lines += ['Active now:'] if compact else ['', 'Active now:']
    lines += running[:2] if compact and running else running if running else [
        '  None (benchmark paused).' if paused else '  None (no recent progress).' if stale
        else '  Starting next trial or saving results.']
    recent = []
    for key in GROUPS:
        for row in data[key].get('trials', []):
            path = Path(row['result_file'])
            if path.exists():
                recent.append((path.stat().st_mtime, key, row))
    lines += ['Latest finished:'] if compact else ['', 'Latest finished:']
    for _, key, row in sorted(recent, key=lambda x: x[0], reverse=True)[:2 if compact else 5]:
        outcome = ('INVALID' if row['technical_invalid'] else
                   'SUCCESS' if row['success'] else 'FAIL')
        why = 'physical gate passed' if row['success'] else observed_gate(row, key)
        if compact:
            lines.append(f"  #{row['index']:03d} {row['instrument']:16} "
                         f'{LABELS[key]:17} {outcome:7} {why[:25]}')
        else:
            lines.append(f"  #{row['index']:03d} {row['instrument']:16} "
                         f'{LABELS[key]:17} {outcome:7} {why}')
    if not recent:
        lines.append('  Waiting for first completed trial.')
    if compact:
        lines += ['Invalid = reset error; difficult spawn may still be valid.',
                  'So far = completed only; final at 100/100 each. Ctrl+C closes view.']
    else:
        lines += ['', 'Invalid means failed reset checks; difficult/occluded spawn can still be valid.',
                  'So far = completed trials only. Final percentages require every row at 100/100.',
                  'Ctrl+C exits this display; the benchmark keeps running.',
                  f'Results: {BASE}']
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true', help='Refresh terminal every 2 seconds')
    args = parser.parse_args()
    try:
        while True:
            if args.watch and sys.stdout.isatty():
                sys.stdout.write('\033[H\033[2J')
            print(frame(compact=args.watch), flush=True)
            if not args.watch:
                return
            time.sleep(2)
    except KeyboardInterrupt:
        print('\nDisplay closed; check the status above for the benchmark state.', flush=True)


if __name__ == '__main__':
    main()
