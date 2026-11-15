"""Create readable local GIF galleries for trained Scissor Pick evaluations."""
import html
import json
from pathlib import Path

try:
    from .workspace_root import workspace_root
except ImportError:
    from workspace_root import workspace_root
ROOT = workspace_root()


def build(folder, title):
    summary=json.loads((folder/'summary.json').read_text())
    cards=[]
    for trial in summary['trials']:
        index=trial['scene']
        result_file=folder/f'trial_{index:02d}/result.json'
        result=json.loads(result_file.read_text())
        gif=Path(result['gif'])
        if not gif.is_file() or gif.parent!=result_file.parent:
            raise RuntimeError(f'Missing or unexpected GIF: {gif}')
        rel=f'trial_{index:02d}/{gif.name}'
        outcome='SUCCESS' if result['lift_success'] else 'FAIL'
        cls='success' if result['lift_success'] else 'fail'
        steps=len((result_file.parent/'actions.jsonl').read_text().splitlines())
        cards.append(f'<article><header><strong>Trial {index:02d}</strong><span class="{cls}">{outcome}</span>'
                     f'<small>{steps} model-controlled steps</small></header>'
                     f'<a href="{html.escape(rel)}"><img src="{html.escape(rel)}" '
                     f'alt="Scissor Pick trial {index:02d}: {outcome}" loading="lazy"></a>'
                     f'<footer><a href="{html.escape(rel)}">Open GIF</a> · '
                     f'<a href="trial_{index:02d}/result.json">Result JSON</a></footer></article>')
    page=f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,Arial,sans-serif;margin:0;background:#fff;color:#172536}}
main{{max-width:1500px;margin:auto;padding:32px 24px 60px}}
h1{{margin:0 0 8px;font-size:clamp(26px,3vw,40px)}}
p{{font-size:17px;line-height:1.5;color:#405168;margin:0 0 24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,560px),1fr));gap:22px}}
article{{border:1px solid #cdd6df;border-radius:12px;overflow:hidden;background:#fff}}
header{{display:flex;align-items:center;gap:12px;padding:14px 18px;background:#f5f8fb;flex-wrap:wrap}}
header strong{{font-size:20px}}header small{{margin-left:auto;color:#56677b}}
.success{{color:#086b43;font-weight:700}}.fail{{color:#b12626;font-weight:700}}
img{{display:block;width:100%;height:auto;background:#eef1f5}}
footer{{padding:12px 18px}}a{{color:#155a9e}}
</style><main><h1>{html.escape(title)}</h1>
<p>{summary['successes']}/{summary['valid']} physical Pick successes. Every GIF shows a separate live-camera RF-DETR B + trained DP rollout; recorded expert actions were not replayed. Click a GIF to open it full size.</p>
<div class="grid">{''.join(cards)}</div></main></html>'''
    (folder/'GIF_INDEX.html').write_text(page)


if __name__=='__main__':
    base=ROOT/'datasets/DP_evaluation'
    build(base/'scissor_trained_dp_live_test_20261103','Scissor Pick — trained DP — 10 test rollouts')
    build(base/'scissor_trained_dp_live_test_perturbed_20261103','Scissor Pick — trained DP — 10 shifted test rollouts')
