"""Warm one-shot versus prepared response sweeps; identical kernels and data."""
import json
import platform
import time
import numpy as np
import numba
from threadpoolctl import threadpool_limits
from tbkit import HR,response_kernel,prepare_geometry,response_from_geometry


def measure(function,repeats=3):
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter();function();samples.append(time.perf_counter()-start)
    return float(np.median(samples))


def run():
    seed=20261113;rng=np.random.default_rng(seed)
    r=rng.integers(-3,4,(31,3));r[0]=0
    h=.03*(rng.normal(size=(31,8,8))+1j*rng.normal(size=(31,8,8)))
    model=HR(r,h).hermitize().with_hopping([0,0,0],np.diag(np.arange(8)*4.),add=True)
    k=rng.uniform(-np.pi,np.pi,(512,3));mu=np.linspace(-1,29,41)
    requests=[(kind,axes,t) for t in (50.,150.,300.) for kind,axes in [('berry',(0,1)),('soahc',(0,1,2)),('ncte',(0,1,2))]]
    def one_shot():
        return np.array([response_kernel(model,k,mu,kind=kind,indices=axes,temperature=t) for kind,axes,t in requests])
    def prepared():
        geometry=prepare_geometry(model,k)
        return np.array([response_from_geometry(geometry,mu,kind=kind,indices=axes,temperature=t) for kind,axes,t in requests])
    with threadpool_limits(limits=1):
        start=time.perf_counter();geometry=prepare_geometry(model,k);first=time.perf_counter()-start
        # Both timed paths include preparation; only the reusable path avoids
        # the other eight preparations. No precomputed work is timed as free.
        baseline=one_shot();shared=prepared()
        error=float(np.max(abs(baseline-shared)))
        np.testing.assert_allclose(baseline,shared,rtol=1e-12,atol=1e-12)
        one=measure(one_shot);reuse=measure(prepared)
    return {'platform':platform.platform(),'numpy':np.__version__,'numba':numba.__version__,
            'seed':seed,'numba_threads':numba.get_num_threads(),'blas_threads':1,
            'nk':len(k),'norb':model.norb,'nr':len(model.vectors),'requests':len(requests),
            'first_geometry_preparation_s':first,'warm_one_shot_s':one,
            'warm_prepared_s':reuse,'one_shot_over_prepared':one/reuse,
            'max_absolute_error':error,'geometry_bytes':geometry.nbytes,
            'one_shot_batch_eigensystem_preparations':len(requests),
            'reused_batch_eigensystem_preparations':1,
            'scope':'workflow reuse benchmark, not a MATLAB comparison or a universal kernel speedup'}

if __name__=='__main__':print(json.dumps(run(),indent=2))
