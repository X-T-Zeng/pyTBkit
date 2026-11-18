"""Reproducible candidate index of symbolic/presentation use in pinned MATLAB.

This is a conservative lexical index, not an API parser or parity measurement.
Only full comment lines are removed; inline comments/strings can still match.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess


SIGNALS = ('sym','symvar','simplify','subs','collect','expand','rewrite','taylor',
           'diff','int','solve','equationsToMatrix','null','latex','pretty',
           'matlabFunction','vpa','assume','assumeAlso','Ynm','legendreP')


def inventory(root):
    config=json.loads((root/'python/upstream.json').read_text())
    source=root/config['checkout']
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if revision!=config['commit']:
        raise RuntimeError('official source is not at the pinned commit')
    files=sorted((source/'src').rglob('*.m'))
    candidates=[];counts=Counter()
    for path in files:
        lines=path.read_text(errors='replace').splitlines()
        hits={}
        for number,line in enumerate(lines,1):
            if line.lstrip().startswith('%'):
                continue
            for signal in SIGNALS:
                if re.search(r'(?<!\w)'+signal+r'\s*\(',line):
                    hits.setdefault(signal,[]).append(number)
            if re.search(r'^\s*syms\b',line):
                hits.setdefault('syms',[]).append(number)
        if hits:
            candidates.append({'path':path.relative_to(source).as_posix(),'signals':hits})
            counts.update(hits.keys())
    return {'repository':config['repository'],'commit':revision,
            'source_files':len(files),'candidate_files':len(candidates),
            'meaning':'lexical candidates, not completed APIs or verified physics',
            'limitations':'full comment lines excluded; inline comments and strings may match; dynamic calls may be missed',
            'signal_file_counts':dict(sorted(counts.items())),'files':candidates}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='work/symbolic-source-inventory.json')
    p.add_argument('--check',action='store_true',help='check committed index, retaining a fresh report')
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    result=inventory(root)
    output=root/args.output;output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if args.check:
        expected=json.loads((root/'python/audit/symbolic-source-inventory.json').read_text())
        if result!=expected:
            raise RuntimeError('symbolic source index is stale or reference differs')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
