"""Exercise the real backend selection code without starting Isaac Sim."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class PolicyWindowTests(unittest.TestCase):
    def test_all_backends_select_only_requested_windows(self):
        paths = list((Path(__file__).resolve().parents[1]/'backends').glob('phase3_grid_split_*_recorder.py'))
        self.assertEqual(len(paths), 5)
        for path in paths:
            with self.subTest(backend=path.name):
                tree = ast.parse(path.read_text(encoding='utf-8-sig'))
                nodes = []
                for node in tree.body:
                    if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id in
                            ('PICK_STAGE_SUFFIXES','PLACE_STAGE_SUFFIXES') for t in node.targets):
                        nodes.append(node)
                    if isinstance(node,ast.FunctionDef) and node.name=='save_realcompat_segment':
                        # Keep the real selection logic; stop immediately before filesystem I/O.
                        stop = next(i for i,n in enumerate(node.body) if isinstance(n,ast.Expr)
                                    and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
                                    and n.value.func.attr=='makedirs')
                        node.body = node.body[:stop] + [ast.Return(value=ast.Name(id='indices',ctx=ast.Load()))]
                        nodes.append(node)
                ns = dict(canonical_object_name=lambda x:x, stage_prefix_for_object=lambda x:'TARGET',
                          stage_suffix_from_name=lambda x:x.removeprefix('TARGET_'))
                exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),ns)
                stages = ['OPEN_HOVER','LOWER_PRE','LOWER_GRASP','LOWER_EXTRA','CLOSE',
                          'LIFT_CLEAR','MOVE_TO_TARGET','LOWER_PLACE','OPEN','RETREAT']
                recorder = SimpleNamespace(actions=[0]*20,stage_names=['TARGET_'+s for s in stages for _ in range(2)])
                for skill,expected in [('pick',list(range(2,12))),('place',list(range(14,20)))]:
                    selected=ns['save_realcompat_segment'](recorder,0,skill,'unused',True,'target')
                    self.assertEqual(selected,expected)
                    self.assertEqual(len(selected),len(set(selected)))


if __name__=='__main__':
    unittest.main()
