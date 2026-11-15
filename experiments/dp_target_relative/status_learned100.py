"""Read-only progress for the 100 trained-model Scissor Pick trials."""
import json
from pathlib import Path

try:
    from .workspace_root import workspace_root
except ImportError:
    from workspace_root import workspace_root
ROOT = workspace_root()
folder=ROOT/'datasets/DP_evaluation/scissor_trained_dp_live_100_20261103'
rows=[]
for i in range(1,101):
    file=folder/f'trial_{i:03d}/result.json'
    if file.exists():
        r=json.loads(file.read_text())
        rows.append((i,bool(r.get('lift_success')),r.get('failure')))
success=sum(r[1] for r in rows)
print(f'TRAINED DP + RF-DETR B | Scissor Pick | {len(rows)}/100 completed')
print(f'Success: {success}   Failure: {len(rows)-success}   Remaining: {100-len(rows)}')
if rows:
    print('Latest completed:', ', '.join(f'{i:03d} {"SUCCESS" if ok else "FAIL"}' for i,ok,_ in rows[-8:]))
print(f'Results: {folder}')
