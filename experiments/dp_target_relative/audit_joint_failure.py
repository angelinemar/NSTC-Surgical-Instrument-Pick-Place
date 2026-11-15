"""Classify observed failure endpoints from saved Isaac Sim trial evidence.

Categories identify the first unmet measured gate in a fixed order. They are
descriptive, mutually exclusive plot labels, not proven root causes. All
failed scheduled trials remain in the denominator.
"""
from __future__ import annotations

import json
from pathlib import Path


PICK_LABELS = {
    'pose_unavailable': 'RF-DETR pose unavailable',
    'no_close': 'No CLOSE command',
    'no_lift': 'CLOSE, lift <35 mm',
    'partial_lift': 'Lift 35–100 mm',
    'hold_unstable': 'Reached 100 mm; hold gate failed',
    'reset_invalid': 'Invalid technical reset',
    'other': 'Other / incomplete log',
}
PLACE_LABELS = {
    'no_open': 'No OPEN command',
    'outside_slot': 'Outside assigned slot',
    'center_error': 'Center error >8 mm',
    'axis_error': 'Axis error >8°',
    'bottom_error': 'Height error >5 mm',
    'not_settled': 'Tool not settled',
    'fingers_closed': 'Fingers not open',
    'tcp_not_clear': 'TCP not clear',
    'gate_not_stable': 'Gate not stable 12 steps',
    'reset_invalid': 'Invalid technical reset',
    'other': 'Other / incomplete log',
}


def read_actions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_trial(row: dict, variant: str, skill: str) -> dict:
    run = Path(row['result_file']).parent
    result_path = run / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    actions = read_actions(run / 'actions.jsonl')
    failure_raw = str(row.get('failure') or '')
    evidence = {}
    violations = []

    if row.get('success'):
        category = 'success'
    elif row.get('technical_invalid'):
        category = 'reset_invalid'
    elif skill == 'pick' and ('estimate_scissor_pose.py' in failure_raw or
                              not actions and not (run / 'live_camera_pose.json').exists()):
        category = 'pose_unavailable'
        estimator = run / 'pose_estimator.log'
        if estimator.exists():
            errors = [line for line in estimator.read_text(errors='replace').splitlines()
                      if line.startswith(('RuntimeError:', 'ValueError:'))]
            evidence['pose_estimator_error'] = errors[-1] if errors else None
    elif not actions:
        category = 'other'
    elif skill == 'pick':
        close_sent = any(action['action'][7] < 0 for action in actions)
        z0 = actions[0]['object_center_for_scoring'][2]
        max_rise_mm = max(0., 1000. * (max(action['object_center_for_scoring'][2]
                                          for action in actions) - z0))
        evidence.update(close_commanded=close_sent,
                        max_center_rise_mm=round(max_rise_mm, 2),
                        executed_action_steps=len(actions))
        if not close_sent:
            category = 'no_close'
        elif max_rise_mm < 35:
            category = 'no_lift'
        elif max_rise_mm < 100:
            category = 'partial_lift'
        else:
            category = 'hold_unstable'
    else:
        open_sent = any(action['action'][7] > 0 for action in actions)
        last = actions[-1].get('place_components') or {}
        evidence.update(open_commanded=open_sent,
                        executed_action_steps=len(actions),
                        final_place_gates=last)
        if not open_sent:
            violations.append('no_open')
        if last:
            if not last.get('inside_slot'):
                violations.append('outside_slot')
            if last.get('center_error_m', float('inf')) > .008:
                violations.append('center_error')
            if last.get('axis_error_deg', float('inf')) > 8.:
                violations.append('axis_error')
            if last.get('bottom_error_m', float('inf')) > .005:
                violations.append('bottom_error')
            if not last.get('settled'):
                violations.append('not_settled')
            if not last.get('fingers_open'):
                violations.append('fingers_closed')
            if not last.get('ee_clear'):
                violations.append('tcp_not_clear')
        category = violations[0] if violations else 'gate_not_stable' if last else 'other'

    return dict(variant=variant, skill=skill, trial_index=row['index'],
                instrument=row['instrument'], success=bool(row.get('success')),
                technical_invalid=bool(row.get('technical_invalid')),
                category=category, all_unmet_final_gates=violations,
                failure_raw=failure_raw or None, evidence=evidence,
                result_file=str(result_path), actions_file=str(run / 'actions.jsonl'),
                gif=row.get('gif') or result.get('gif'),
                reset_diagnostics=str(run / 'reset_diagnostics.json'))
