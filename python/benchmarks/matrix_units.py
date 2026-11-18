"""Matrix-unit preparation against optimized NumPy contractions, several sizes."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from tbkit import dihedral_group,MatrixRepresentation,symmetry_adapted_basis
from tbkit.group_kernels import matrix_unit_column


def numpy_column(matrices,irrep):
    order,dimension,_=matrices.shape;degree=irrep.shape[1]
    weights=(degree/order)*irrep[:,:,0].conj().T
    return (weights@matrices.reshape(order,-1)).reshape(degree,dimension,dimension)


def median(function,repeats=5):
    times=[]
    for _ in range(repeats):
        start=time.perf_counter();function();times.append(time.perf_counter()-start)
    return float(np.median(times))


def run(output='work/matrix-unit-benchmark.json'):
    group=dihedral_group(3);angle=2*np.pi/3
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    irrep=MatrixRepresentation(group,[np.linalg.matrix_power(rotation,a)@np.diag([1,(-1)**b]) for a,b in group])
    small=irrep.as_arrays();rng=np.random.default_rng(83);loads=[];first=None
    for dimension in (8,16,32):
        canonical=np.array([np.kron(np.eye(dimension//2),m) for m in small]);cases=[]
        for _ in range(128):
            q=np.linalg.qr(rng.normal(size=(dimension,dimension))+1j*rng.normal(size=(dimension,dimension)))[0]
            value=np.ascontiguousarray(q@canonical@q.conj().T);value.setflags(write=False);cases.append(value)
        if first is None:
            start=time.perf_counter();matrix_unit_column(cases[0],small);first=time.perf_counter()-start
        reference=lambda:[numpy_column(big,small) for big in cases]
        compiled=lambda:[matrix_unit_column(big,small) for big in cases]
        expected,actual=reference(),compiled()
        error=float(max(np.max(np.abs(a-b)) for a,b in zip(expected,actual)));assert error<1e-12
        compiled();numpy_seconds=median(reference);numba_seconds=median(compiled)
        target=MatrixRepresentation(group,cases[0]);timings={}
        for backend in ('numpy','numba'):
            timings[backend]=median(lambda:symmetry_adapted_basis(target,[irrep],backend=backend),repeats=3)
        loads.append({'dimension':dimension,'case_count':len(cases),'warm_numpy_median_seconds':numpy_seconds,
                      'warm_numba_median_seconds':numba_seconds,'speedup':numpy_seconds/numba_seconds,
                      'max_absolute_error':error,'warm_complete_decomposition_median_seconds':timings})
    result={'group':'D3','order':group.order,'irrep_dimension':2,'seed':83,'first_jit_or_cache_seconds':first,
            'kernel_serial':True,'numba_threads_env':os.environ.get('NUMBA_NUM_THREADS'),
            'blas_threads_env':os.environ.get('OPENBLAS_NUM_THREADS'),'loads':loads,
            'scope':'prepared matrix-unit columns versus vectorized NumPy matmul; full checked decomposition measured separately, no MATLAB comparison'}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='work/matrix-unit-benchmark.json')
    run(parser.parse_args().output)
