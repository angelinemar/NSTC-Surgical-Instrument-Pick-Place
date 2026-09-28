"""Local pre-publish hygiene check. Reports filenames only, never secret values.

Heuristic defense, not proof that no secret exists. Scan tracked and unignored
source; recordings/checkpoints/credentials are prohibited in the publish set.
"""
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    raw=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT)
    names=sorted(set(raw.decode().split('\0'))-{''})
    prohibited=re.compile(r'(^|/)(\.env(?:\..*)?|credentials\.json|service-account[^/]*\.json)$|\.(h5|hdf5|pt|pth|pem|key|log)$',re.I)
    patterns=[re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
              re.compile(r'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b'),
              re.compile(r'\bgithub_pat_[A-Za-z0-9_]{50,}\b'),
              re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
              re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{35,}\b')]
    issues=[]
    for name in names:
        if prohibited.search(name) and not name.endswith('.env.example'):
            issues.append((name,'prohibited publish artifact')); continue
        path=ROOT/name
        if not path.is_file(): continue
        if path.suffix.lower() not in ('.py','.md','.json','.yaml','.yml','.toml','.ps1','.txt','.svg','.usda','.gitignore'):
            continue
        content=path.read_text(encoding='utf-8',errors='replace')
        if any(p.search(content) for p in patterns): issues.append((name,'possible secret signature'))
    for name,reason in issues: print(reason+': '+name)
    if issues: raise SystemExit(1)
    print(f'PASS: {len(names)} publish candidates; no prohibited artifacts or recognized secret signatures')


if __name__=='__main__': main()
