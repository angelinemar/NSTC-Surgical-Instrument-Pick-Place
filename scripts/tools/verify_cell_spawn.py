"""Offline randomized contract check; no Isaac application launch."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'compat'))
from phase4_cell_spawn import fit_spawn_to_cells,instrument_envelopes
from phase4_scene import LAYOUT

def main():
    rng=np.random.default_rng(904224)
    sizes=instrument_envelopes()
    x0,x1=LAYOUT['grid_x'];y0,y1=LAYOUT['grid_y']
    w=(x1-x0)/LAYOUT['grid_cols'];h=(y1-y0)/LAYOUT['grid_rows']
    for target in sizes:
        for _ in range(1000):
            spawn={name:{'yaw_deg':float(rng.uniform(0,360))} for name in sizes}
            spawn['grid']={}
            yaws={n:spawn[n]['yaw_deg'] for n in sizes}
            fit_spawn_to_cells(spawn,target,rng)
            cells=json.loads(spawn['grid']['object_cell_assignments'])
            assert len(set(a['cell_id'] for a in cells.values()))==5
            for name,a in cells.items():
                assert spawn[name]['yaw_deg']==yaws[name]
                cx=x0+(a['col']+.5)*w;cy=y0+(a['row']+.5)*h
                assert abs(a['x']-cx)+a['radius_m']<=w/2-LAYOUT['cell_object_margin']+1e-9
                assert abs(a['y']-cy)+a['radius_m']<=h/2-LAYOUT['cell_object_margin']+1e-9
            assert spawn['grid']['cell_id']==cells[target]['cell_id']
    print('PASS: 5000 spawn sets, all five targets, distinct cells, unchanged random yaw, bounded full-asset envelopes.')
    print(json.dumps(sizes,indent=2))

if __name__=='__main__':
    main()
