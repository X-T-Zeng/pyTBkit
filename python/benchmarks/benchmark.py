"""Run with NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1; writes JSON to stdout.

No MATLAB speedup is inferred from comparing Python implementations.
"""
import json
import platform
import time
import numpy as np
import numba
from threadpoolctl import threadpool_limits
from tbkit import HR, models


def measure(f,repeat=5):
    times=[]
    for _ in range(repeat):
        start=time.perf_counter(); f(); times.append(time.perf_counter()-start)
    return float(np.median(times))


def reference(m,k):
    out=[]
    for point in k:
        h=np.zeros((m.norb,m.norb),complex)
        for r,t in zip(m.cartesian_vectors,m.hopping):
            h+=np.exp(1j*np.dot(point,r))*t
        out.append(h)
    return np.array(out)


def run():
    rng=np.random.default_rng(20261113)
    vectors=rng.integers(-5,6,(121,3))
    h=rng.normal(size=(121,8,8))+1j*rng.normal(size=(121,8,8))
    large=HR(vectors,h).hermitize()
    results=[]
    with threadpool_limits(limits=1):
        for name,m,nk in [('two_band',models.qi_wu_zhang(),20000),('wannier_like',large,3000)]:
            k=rng.uniform(-np.pi,np.pi,(nk,m.dim))
            start=time.perf_counter(); a=m.hamiltonian(k); first=time.perf_counter()-start
            b=m.hamiltonian(k,backend='numpy')
            error=float(np.max(np.abs(a-b)))
            np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
            numpy_s=measure(lambda:m.hamiltonian(k,backend='numpy'))
            numba_s=measure(lambda:m.hamiltonian(k))
            results.append(dict(model=name,nk=nk,norb=m.norb,nr=len(m.vectors),
                                first_numba_call_s=first,numpy_median_s=numpy_s,
                                numba_median_s=numba_s,numpy_over_numba=numpy_s/numba_s,max_error=error,
                                reference_loop_100_s=measure(lambda:reference(m,k[:100]),3),
                                bands_numpy_s=measure(lambda:m.bands(k,backend='numpy'),3),
                                bands_numba_s=measure(lambda:m.bands(k),3)))
    return dict(platform=platform.platform(),processor=platform.processor(),numpy=np.__version__,numba=numba.__version__,numba_threads=numba.get_num_threads(),blas_threads=1,seed=20261113,results=results)

if __name__=='__main__': print(json.dumps(run(),indent=2))
