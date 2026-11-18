"""Prepared left-regular compressions: NumPy reference versus Numba kernel."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from tbkit import dihedral_group,irreducible_representations
from tbkit.group_kernels import regular_blocks


def numpy_blocks(table,basis):
    # Optimized vectorized gather and batched products, with no dense L array.
    # Correctness is separately tested against explicit permutation matrices.
    return basis[table].conj().swapaxes(-1,-2)@basis


def median_seconds(function,repeats=5):
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter();function();samples.append(time.perf_counter()-start)
    return float(np.median(samples))


def run(output='work/regular-block-benchmark.json'):
    group=dihedral_group(12);table=group.table;degree=2
    rng=np.random.default_rng(59)
    bases=[np.ascontiguousarray(np.linalg.qr(rng.normal(size=(group.order,degree))+1j*rng.normal(size=(group.order,degree)))[0]) for _ in range(256)]
    start=time.perf_counter();regular_blocks(table,bases[0]);first=time.perf_counter()-start
    reference=lambda:[numpy_blocks(table,q) for q in bases]
    compiled=lambda:[regular_blocks(table,q) for q in bases]
    expected=reference();actual=compiled()
    error=float(max(np.max(np.abs(a-b)) for a,b in zip(expected,actual)))
    assert error<1e-12
    compiled()
    numpy_seconds=median_seconds(reference);numba_seconds=median_seconds(compiled)
    generation_seconds=median_seconds(lambda:irreducible_representations(group),repeats=3)
    result={'group':'D12','order':group.order,'block_dimension':degree,'basis_count':len(bases),'seed':59,
            'first_jit_or_cache_seconds':first,'warm_numpy_median_seconds':numpy_seconds,
            'warm_numba_median_seconds':numba_seconds,'speedup':numpy_seconds/numba_seconds,
            'max_absolute_error':error,'kernel_serial':True,'numba_threads_env':os.environ.get('NUMBA_NUM_THREADS'),
            'blas_threads_env':os.environ.get('OPENBLAS_NUM_THREADS'),
            'warm_full_generation_median_seconds':generation_seconds,
            'scope':'prepared Q-dagger L(g) Q blocks versus vectorized NumPy gather/batched matmul; separate full-generation runtime, no end-to-end speedup or MATLAB comparison'}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='work/regular-block-benchmark.json')
    run(parser.parse_args().output)
