"""Prepared SU(2) arrays against vectorized NumPy, excluding rotation expm."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from tbkit.orbitals import angular_momentum
from tbkit.angular_kernels import angular_momentum_doubled


def numpy_generators(twice_j):
    n=twice_j+1;columns=np.arange(1,n)
    value=np.sqrt(columns.astype(float)*(twice_j-columns+1))/2
    x=np.zeros((n,n),complex);y=np.zeros((n,n),complex)
    x[columns-1,columns]=value;x[columns,columns-1]=value
    y[columns-1,columns]=-1j*value;y[columns,columns-1]=1j*value
    z=np.diag((twice_j/2-np.arange(n)).astype(complex))
    return x,y,z


def median(function,repeats=5):
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter();function();samples.append(time.perf_counter()-start)
    return float(np.median(samples))


def workload(function,argument,count):
    checksum=0.
    for _ in range(count):
        x,y,z=function(argument);checksum+=z[0,0].real+x[0,-1].real
    return checksum


def run(output='work/angular-generator-benchmark.json'):
    start=time.perf_counter();angular_momentum_doubled(1);first=time.perf_counter()-start
    loads=[];count=256
    for doubled in (1,20,100):
        expected=numpy_generators(doubled);actual=angular_momentum_doubled(doubled)
        error=float(max(np.max(abs(a-b)) for a,b in zip(actual,expected)));assert error<1e-13
        reference=lambda:workload(numpy_generators,doubled,count)
        compiled=lambda:workload(angular_momentum_doubled,doubled,count)
        checked=lambda:workload(angular_momentum,doubled/2,count)
        assert reference()==compiled()==checked()
        compiled();np_seconds=median(reference);nb_seconds=median(compiled)
        loads.append({'twice_j':doubled,'dimension':doubled+1,'count':count,'warm_numpy_median_seconds':np_seconds,
                      'warm_numba_median_seconds':nb_seconds,'warm_checked_api_median_seconds':median(checked),
                      'prepared_kernel_speedup':np_seconds/nb_seconds,'max_absolute_error':error})
    result={'first_jit_or_cache_seconds':first,'kernel_serial':True,
            'numba_threads_env':os.environ.get('NUMBA_NUM_THREADS'),'blas_threads_env':os.environ.get('OPENBLAS_NUM_THREADS'),
            'loads':loads,'scope':'prepared generator construction versus vectorized NumPy; checked API separate; not spin_rotation/expm, CG evaluation or MATLAB comparison'}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='work/angular-generator-benchmark.json')
    run(parser.parse_args().output)
