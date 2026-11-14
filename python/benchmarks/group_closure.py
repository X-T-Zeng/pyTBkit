"""Prepared finite-group closures: Python reference versus Numba array kernel.

This measures closures, not full minimum-generator search or MATLAB. First
invocation includes JIT or disk-cache loading; warmed medians are separate.
"""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from tbkit import FiniteGroup
from tbkit.group_kernels import generated_subgroup_indices


def python_closure(table,generators,identity):
    generators=set(int(i) for i in generators)
    known={identity};pending=[identity]
    while pending:
        a=pending.pop()
        for b in sorted(generators):
            c=int(table[a,b])
            if c not in known:known.add(c);pending.append(c)
    return np.array(sorted(known),dtype=np.int64)


def median_seconds(function,repeats=5):
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter();function();samples.append(time.perf_counter()-start)
    return float(np.median(samples))


def run(output='work/group-closure-benchmark.json'):
    values=np.arange(64)
    group=FiniteGroup(np.bitwise_xor(values[:,None],values),name='C2^6')
    rng=np.random.default_rng(41)
    generators=[np.array(sorted(rng.choice(values[1:],size=int(rng.integers(1,7)),replace=False)),dtype=np.int64) for _ in range(1024)]
    table=group.table;identity=group.index(group.identity)
    start=time.perf_counter();generated_subgroup_indices(table,generators[0],identity)
    first=time.perf_counter()-start
    reference=lambda:[python_closure(table,g,identity) for g in generators]
    compiled=lambda:[generated_subgroup_indices(table,g,identity) for g in generators]
    expected=reference();actual=compiled()
    assert all(np.array_equal(a,b) for a,b in zip(expected,actual))
    compiled()  # warm the measured workload
    python_seconds=median_seconds(reference);numba_seconds=median_seconds(compiled)
    result={'group':'C2^6','order':64,'closure_count':len(generators),'seed':41,
            'first_jit_or_cache_seconds':float(first),'warm_python_median_seconds':python_seconds,
            'warm_numba_median_seconds':numba_seconds,'speedup':python_seconds/numba_seconds,
            'all_closures_equal':True,'kernel_serial':True,
            'numba_threads_env':os.environ.get('NUMBA_NUM_THREADS'),
            'blas_threads_env':os.environ.get('OPENBLAS_NUM_THREADS'),
            'scope':'prepared integer-table closures, not end-to-end rank search or MATLAB comparison'}
    output=Path(output);output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='work/group-closure-benchmark.json')
    run(parser.parse_args().output)
