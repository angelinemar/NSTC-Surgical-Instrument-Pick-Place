import ast,unittest
from pathlib import Path
from evaluation.dp_contract import replace_policy_windows,summarize

class ContractTests(unittest.TestCase):
 def test_all_backends_remove_expert_skill_actions(self):
  for path in (Path(__file__).resolve().parents[1]/'backends').glob('*recorder.py'):
   tree=ast.parse(path.read_text(encoding='utf-8-sig'))
   fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pick_and_place_object')
   source=ast.unparse(fn)
   ns={};exec(replace_policy_windows(source),ns)
   constants=ns['pick_and_place_object'].__code__.co_consts
   self.assertIn('OPEN_HOVER',constants);self.assertIn('MOVE_TO_TARGET',constants)
   for stage in ['LOWER_PRE','LOWER_GRASP','CLOSE','LIFT_CLEAR','LOWER_PLACE','OPEN','RETREAT']:
    self.assertNotIn(stage,constants)
   self.assertIn('_dp_eval_skill',ns['pick_and_place_object'].__code__.co_names)
 def test_independent_place_keeps_fsm_pick(self):
  for path in (Path(__file__).resolve().parents[1]/'backends').glob('*recorder.py'):
   tree=ast.parse(path.read_text(encoding='utf-8-sig'))
   fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pick_and_place_object')
   ns={};exec(replace_policy_windows(ast.unparse(fn),'place'),ns)
   constants=ns['pick_and_place_object'].__code__.co_consts
   self.assertIn('LIFT_CLEAR',constants)
   self.assertIn('MOVE_TO_TARGET',constants)
   self.assertNotIn('LOWER_PLACE',constants)
 def test_fail_closed_on_wrong_boundaries(self):
  with self.assertRaises(ValueError):replace_policy_windows('def pick_and_place_object(): pass')
 def test_failures_and_conditional_denominators(self):
  rows=[{},dict(pick_started=True),dict(pick_started=True,lift_success=True),dict(pick_started=True,lift_success=True,place_started=True),dict(pick_started=True,lift_success=True,place_started=True,place_success=True)]
  r=summarize(rows)
  self.assertEqual(r['lift']['percent'],75)
  self.assertEqual(r['place_given_model_lift_and_FSM_transfer']['percent'],50)
  self.assertEqual(r['hybrid_chain']['percent'],20)
  self.assertEqual(r['setup_failures'],1);self.assertEqual(r['transfer_failures'],1)
if __name__=='__main__':unittest.main()
