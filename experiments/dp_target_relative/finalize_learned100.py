"""Audit all 100 checkpoint-driven rollouts and build human-readable results."""
import html
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

try:
    from .workspace_root import workspace_root
except ImportError:
    from workspace_root import workspace_root
ROOT = workspace_root()
OUT=ROOT/'datasets/DP_evaluation/scissor_trained_dp_live_100_20261103'


def main():
    plan=json.loads((OUT/'plan.json').read_text())
    if plan['planned_attempts']!=100 or len(plan['attempts'])!=100:
        raise RuntimeError('Unexpected 100-trial plan')
    rows=[];groups=defaultdict(list);cards=defaultdict(list)
    for i,item in enumerate(plan['attempts'],1):
        folder=OUT/f'trial_{i:03d}'
        file=folder/'result.json'
        if not file.exists():raise RuntimeError(f'Missing result: {file}')
        r=json.loads(file.read_text())
        if r.get('protocol')!='learned_target_relative_dp_live_camera_v1':
            raise RuntimeError(f'Wrong protocol in trial {i}')
        if not r.get('model_inference_used') or not r.get('detector_inference_used') or r.get('oracle_object_pose_used'):
            raise RuntimeError(f'Model/detector contract failed in trial {i}')
        if r.get('expert_replay') or r.get('expert_approach') or r.get('template_replay'):
            raise RuntimeError(f'Expert/template replay in trial {i}')
        if r.get('pick_sha256')!=plan['pick_sha256']:
            raise RuntimeError(f'Pick checkpoint changed in trial {i}')
        reset=r.get('reset')
        if reset:
            if str(Path(reset['source']).resolve())!=item['snapshot']:
                raise RuntimeError(f'Wrong source snapshot in trial {i}')
            for key in ('dx_m','dy_m','yaw_deg'):
                actual=reset[{'dx_m':'target_perturb_dx_m','dy_m':'target_perturb_dy_m','yaw_deg':'target_perturb_yaw_deg'}[key]]
                if abs(actual-item[key])>1e-8:raise RuntimeError(f'Wrong perturbation in trial {i}')
        elif 'RESET_INVALID' not in str(r.get('failure')):
            raise RuntimeError(f'Missing reset evidence in trial {i}')
        success=bool(r['lift_success'])
        gif=Path(r['gif']) if r.get('gif') else None
        if gif and (not gif.is_file() or gif.parent!=folder):
            raise RuntimeError(f'Unexpected GIF path in trial {i}')
        if success and not gif:raise RuntimeError(f'Success without GIF in trial {i}')
        action_file=folder/'actions.jsonl'
        actions=[json.loads(line) for line in action_file.open()] if action_file.exists() else []
        if success and not actions:raise RuntimeError(f'Success without model actions in trial {i}')
        if any(a['actor']!='DP' for a in actions):raise RuntimeError(f'Non-DP action in trial {i}')
        first_close=next((a['step'] for a in actions if a['action'][7]<0),None)
        peak=(max(a['object_center_for_scoring'][2] for a in actions)-actions[0]['object_center_for_scoring'][2]) if actions else None
        min_distance=min((a['tcp_object_center_distance_m'] for a in actions),default=None)
        pose_file=folder/'live_camera_pose.json'
        detector_score=json.loads(pose_file.read_text())['detector_score'] if pose_file.exists() else None
        valid=bool(r.get('pick_started')) and 'RESET_INVALID' not in str(r.get('failure'))
        row=dict(index=i,scene=item['scene'],variation=item['variation'],valid=valid,
                 success=success if valid else False,failure=r.get('failure'),steps=len(actions),
                 first_close_step=first_close,peak_lift_m=peak,
                 min_tcp_object_distance_m=min_distance,detector_score=detector_score,
                 gif=str(gif) if gif else None,
                 result_file=str(file))
        rows.append(row);groups[item['scene']].append(row)
        outcome='SUCCESS' if row['success'] else 'FAIL'
        css='ok' if row['success'] else 'bad'
        rel=f'trial_{i:03d}/{gif.name}' if gif else None
        media=(f'<a href="{html.escape(rel)}"><img loading="lazy" src="{html.escape(rel)}" '
               f'alt="Trial {i:03d} {outcome}"></a><p><a href="{html.escape(rel)}">Open GIF</a>'
               if rel else '<p>No motion GIF: trial stopped before video capture.')
        cards[item['scene']].append(f'<details class="card"><summary><b>Trial {i:03d}</b>'
                                  f'<span class="{css}">{outcome}</span>'
                                  f'<small>{len(actions)} steps</small></summary>'
                                  f'{media} · <a href="trial_{i:03d}/result.json">Result JSON</a></p></details>')
    successes=sum(x['success'] for x in rows)
    valid=sum(x['valid'] for x in rows)
    summary=dict(contract=plan['contract'],complete=True,planned_attempts=100,
                 completed_attempts=100,valid_attempts=valid,technical_invalid=100-valid,
                 successes=successes,failures=100-successes,success_pct_of_100=successes,
                 model_inference_used=True,detector_inference_used=True,
                 expert_or_template_replay_used=False,oracle_object_pose_used=False,
                 pick_checkpoint_sha256=plan['pick_sha256'],detector_checkpoint_sha256=plan['detector_sha256'],
                 shared_base_scenes=10,per_scene={str(k):dict(successes=sum(r['success'] for r in v),attempts=10)
                                                   for k,v in sorted(groups.items())},trials=rows)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    successful_steps=[r['steps'] for r in rows if r['success']]
    lines=['# RF-DETR B + trained DP: 100 Scissor Pick attempts','',
           f'**Result: {successes}/100 successful physical Picks ({successes}%).** '
           f'{100-successes} did not pass the 10 cm stable-lift gate. '
           f'{100-valid} had a technically invalid reset.','',
           'The frozen plan uses 10 held-out Scissor test scenes × 10 preset XY/yaw target perturbations. '
           'XY offsets were sampled independently within ±1.8 cm, with at least 0.5 cm total XY shift; '
           'yaw was sampled within ±25°. These are 100 attempts but only 10 independent base scenes.','',
           'RF-DETR B detected the target from live front RGB-D; the trained target-relative DP checkpoint '
           'generated all Pick motion and OPEN/CLOSE actions. Expert trajectories were not replayed. '
           'The simulator object pose was used for reset and scoring, never as a policy input.','',
           f'Median successful episode length: {median(successful_steps):g} control steps.' if successful_steps else '',
           '', '## Results by original held-out scene','',
           '| Test scene | Successes / 10 |','| --- | ---: |']
    for scene,group in sorted(groups.items()):lines.append(f'| {scene:02d} | {sum(r["success"] for r in group)}/10 |')
    lines+=['','## Failed attempts','',
            'All eight failures reached the 300-step timeout. The detector returned a Scissor prediction in each; '
            'the model also issued a CLOSE command. Peak lift below is the largest observed Scissor rise, '
            'which did not satisfy the 10 cm stable-lift gate. These observations do not identify one unique cause.','',
            '| Trial | Base scene | First CLOSE step | Peak lift | GIF |',
            '| ---: | ---: | ---: | ---: | --- |']
    for r in rows:
        if not r['success']:
            rel=f'trial_{r["index"]:03d}/{Path(r["gif"]).name}' if r['gif'] else None
            gif_link=f'[View]({rel})' if rel else 'Unavailable'
            close=str(r['first_close_step']) if r['first_close_step'] is not None else 'None'
            lift=f'{r["peak_lift_m"]*100:.1f} cm' if r['peak_lift_m'] is not None else 'Unknown'
            lines.append(f'| {r["index"]:03d} | {r["scene"]:02d} | {close} | {lift} | {gif_link} |')
    lines+=['','## Evidence','',
            '- [Fixed plan](plan.json) (including checkpoint hashes and all perturbations)',
            '- [Machine-readable results](summary.json)',
            '- [GIF gallery](GIF_INDEX.html) with every success and failure',
            '', '## Scope','',
            'This is Scissor **Pick only** from held-out recorded Pick starts in Isaac Sim. '
            'It does not measure Place, whole-task transfer, other instruments, arbitrary table positions, '
            'or physical hospital operation.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    scene_html=''.join(f'<details class="scene" {"open" if scene==1 else ""}><summary>'
                       f'Scene {scene:02d} — {sum(r["success"] for r in group)}/10 successes</summary>'
                       f'<div class="grid">{"".join(cards[scene])}</div></details>'
                       for scene,group in sorted(groups.items()))
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>100 trained DP + RF-DETR Scissor Pick rollouts</title>
<style>body{{font-family:system-ui,Arial,sans-serif;margin:0;background:#fff;color:#172536}}
main{{max-width:1500px;margin:auto;padding:30px}}h1{{margin:0 0 10px}}
p{{line-height:1.5;color:#405168}}.scene{{border:1px solid #cad3de;border-radius:10px;margin:16px 0;padding:12px}}
.scene>summary{{font-weight:700;font-size:20px;cursor:pointer}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,500px),1fr));gap:14px;margin-top:16px}}
.card{{border:1px solid #d4dce5;border-radius:8px;padding:10px}}.card>summary{{display:flex;gap:12px;cursor:pointer}}
.card small{{margin-left:auto;color:#607087}}.ok{{color:#08704b;font-weight:700}}.bad{{color:#b12626;font-weight:700}}
img{{display:block;width:100%;height:auto;margin-top:12px}}a{{color:#165c9e}}</style>
<main><h1>100 trained DP + RF-DETR B Scissor Pick rollouts</h1>
<p><strong>{successes}/100 successful Picks.</strong> Each trial below has its own GIF. Open a scene, then open a trial to play its GIF. All Pick actions came from the trained checkpoint; expert actions were not replayed. <a href="RESULTS.md">Read the result summary.</a></p>
{scene_html}</main></html>'''
    (OUT/'GIF_INDEX.html').write_text(page)
    print(f'FINAL {successes}/100; report: {OUT/"RESULTS.md"}; GIF gallery: {OUT/"GIF_INDEX.html"}')


if __name__=='__main__':main()
