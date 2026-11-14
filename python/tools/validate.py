"""Run finite cloud validation cycles, retain logs and a machine-readable result.

This script runs tests/examples/optional benchmarks. It does not autonomously
write new implementation code or schedule a background agent.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='work/python-validation')
    parser.add_argument('--benchmarks',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];out=(root/args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.setdefault('NUMBA_NUM_THREADS','2');env['OPENBLAS_NUM_THREADS']='1'
    env['MKL_NUM_THREADS']='1';env['MPLBACKEND']='Agg';env['PYTHONHASHSEED']='0'
    env['NUMBA_CACHE_DIR']=str(root/'work/numba-cache');env['MPLCONFIGDIR']=str(root/'work/matplotlib')
    tasks=[('upstream',[sys.executable,'python/tools/prepare_upstream.py']),
           ('tests',[sys.executable,'-m','pytest','python/tests','-q',f'--junitxml={out / "junit.xml"}']),
           ('group_theory',[sys.executable,'python/examples/group_theory.py','--output',str(out/'group_theory')]),
           ('group_generators',[sys.executable,'python/examples/group_generators.py','--output',str(out/'group_generators')]),
           ('examples',[sys.executable,'python/examples/full_suite.py','--output',str(out/'examples')]),
           ('plot_example',[sys.executable,'python/examples/research.py']),
           ('external_example',[sys.executable,'python/examples/external_data.py','--output',str(out/'external')]),
           ('advanced_example',[sys.executable,'python/examples/advanced.py','--output',str(out/'advanced')]),
           ('response_sweep',[sys.executable,'python/examples/response_sweep.py','--output',str(out/'response_sweep')])]
    if args.benchmarks:
        tasks.append(('group_closure_benchmark',[sys.executable,'python/benchmarks/group_closure.py','--output',str(out/'group-closure-benchmark.json')]))
        tasks.extend([('benchmark',[sys.executable,'python/benchmarks/benchmark.py']),('sparse_benchmark',[sys.executable,'python/benchmarks/sparse_benchmark.py']),
                      ('response_reuse_benchmark',[sys.executable,'python/benchmarks/response_reuse.py'])])
    results=[]
    for name,command in tasks:
        start=time.perf_counter()
        with (out/f'{name}.log').open('w') as log:
            child=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        item={'name':name,'exit_code':child.returncode,'seconds':time.perf_counter()-start,'log':f'{name}.log'}
        results.append(item);print(json.dumps(item),flush=True)
    report={'python':sys.version,'numba_threads':env['NUMBA_NUM_THREADS'],'passed':all(r['exit_code']==0 for r in results),'steps':results}
    if (out/'junit.xml').exists():
        suites=ET.parse(out/'junit.xml').getroot()
        report['tests']={key:sum(int(s.attrib.get(key,0)) for s in suites.iter('testsuite')) for key in ('tests','failures','errors','skipped')}
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['passed'] else 1

if __name__=='__main__':sys.exit(main())
