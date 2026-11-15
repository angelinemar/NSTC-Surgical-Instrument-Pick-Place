"""CPU-only evaluation contract and fail-closed backend adaptation."""
import ast
from pathlib import Path

def replace_policy_windows(source, skill="both"):
    """Keep recorder approach/transport verbatim; remove ALL expert skill actions."""
    tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pick_and_place_object')
    def stage(node,name):
        return any(isinstance(n,ast.Constant) and n.value==name for n in ast.walk(node))
    starts=[i for i,n in enumerate(fn.body) if stage(n,'LOWER_PRE')]
    transfers=[i for i,n in enumerate(fn.body) if stage(n,'MOVE_TO_TARGET')]
    places=[i for i,n in enumerate(fn.body) if stage(n,'LOWER_PLACE')]
    finals=[i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='final_pos' for t in n.targets)]
    if any(len(v)!=1 for v in (starts,transfers,places,finals)):
        raise ValueError('Unsupported backend skill boundaries; refusing expert fallback')
    a,b,c,d=starts[0],transfers[0],places[0],finals[0]
    if not a<b<c<d:raise ValueError('Invalid skill order')
    pick=ast.parse("close_z = float(get_pos_fn(env)[2])\nmz = max(mz, _dp_eval_skill(env, 'pick'))").body
    place=ast.parse("_dp_eval_skill(env, 'place')\n_dp_eval_finish()").body
    if skill=='both':fn.body=fn.body[:a]+pick+fn.body[b:c]+place+fn.body[d:]
    elif skill=='pick':fn.body=fn.body[:a]+pick+ast.parse("_dp_eval_finish()").body
    elif skill=='place':fn.body=fn.body[:c]+place+fn.body[d:]
    else:raise ValueError('Unknown evaluation skill')
    ast.fix_missing_locations(tree)
    return compile(tree,'<DP evaluation: recorder FSM + learned skills>','exec')

def summarize(rows):
    def metric(success,total):
        return dict(successes=success,total=total,percent=100*success/total if total else None)
    pick=[r for r in rows if r.get('pick_started')]
    place=[r for r in rows if r.get('place_started')]
    return dict(attempts=len(rows),
        lift=metric(sum(bool(r.get('lift_success')) for r in pick),len(pick)),
        place_given_model_lift_and_FSM_transfer=metric(sum(bool(r.get('place_success')) for r in place),len(place)),
        hybrid_chain=metric(sum(bool(r.get('place_success')) for r in rows),len(rows)),
        setup_failures=sum(not r.get('pick_started',False) for r in rows),
        transfer_failures=sum(bool(r.get('lift_success')) and not r.get('place_started',False) for r in rows))
