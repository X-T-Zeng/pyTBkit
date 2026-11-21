"""Array dressing versus full model API; cold compilation is timed separately."""
import argparse
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import numba
from threadpoolctl import threadpool_limits
from tbkit import HR,embed_orbitals
from tbkit.gauge_kernels import embed_hamiltonian,embed_gradient


def measure(function,repeats=5):
    wall=[];cpu=[]
    for _ in range(repeats):
        start=time.perf_counter();cs=time.process_time();function()
        wall.append(time.perf_counter()-start);cpu.append(time.process_time()-cs)
    return {'wall_samples_s':wall,'cpu_samples_s':cpu,'wall_median_s':float(np.median(wall)),
            'cpu_median_s':float(np.median(cpu))}


def numpy_dressing(k,t,h,dh):
    p=np.exp(1j*k@t.T)
    out=p.conj()[:,:,None]*h*p[:,None,:]
    displacement=t.T[:,None,:]-t.T[:,:,None]
    gradient=p.conj()[:,None,:,None]*(dh+1j*displacement[None]*h[:,None])*p[:,None,None,:]
    return out,gradient


def run():
    rng=np.random.default_rng(20261121)
    report={'seed':20261121,'platform':platform.platform(),'numpy':np.__version__,'numba':numba.__version__,
            'numba_threads':numba.get_num_threads(),'blas_threads':1,'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),
            'scope':'point-orbital array dressing and full H+gradient API; no MATLAB or universal speed claim','cases':[]}
    with threadpool_limits(limits=1,user_api='blas'):
        for n,nk in ((8,1024),(32,1024)):
            r=rng.integers(-2,3,(9,3));r[0]=0
            coefficients=rng.normal(size=(9,n,n))+1j*rng.normal(size=(9,n,n))
            model=HR(r,coefficients,orbitals=rng.random((n,3))).hermitize()
            e=embed_orbitals(model);k=rng.uniform(-np.pi,np.pi,(nk,3))
            h=model.hamiltonian(k,backend='numpy');dh=model.gradient(k,backend='numpy');centers=e.centers
            if n==8:
                for name,kernel,args in [('hamiltonian',embed_hamiltonian,(k,centers,h)),('gradient',embed_gradient,(k,centers,h,dh))]:
                    fresh=numba.njit(cache=False,nogil=True,parallel=True)(kernel.py_func)
                    start=time.perf_counter();cs=time.process_time();fresh(*args)
                    report['fresh_compile_and_first_'+name]={'wall_s':time.perf_counter()-start,'cpu_s':time.process_time()-cs}
            reference=numpy_dressing(k,centers,h,dh)
            def arrays():return embed_hamiltonian(k,centers,h),embed_gradient(k,centers,h,dh)
            result=arrays();errors=[]
            for actual,expected in zip(result,reference):
                np.testing.assert_allclose(actual,expected,rtol=5e-13,atol=5e-13)
                errors.append(float(np.max(abs(actual-expected))))
            # Warm every public path before timing, and check against precomputed independent arrays.
            for backend in ('numpy','numba'):
                np.testing.assert_allclose(e.hamiltonian(k,backend=backend),reference[0],rtol=5e-13,atol=5e-13)
                np.testing.assert_allclose(e.gradient(k,backend=backend),reference[1],rtol=5e-13,atol=5e-13)
            case={'nk':nk,'norb':n,'nr':len(model.vectors),'max_abs_H_error':errors[0],'max_abs_gradient_error':errors[1]}
            case['array_numba']=measure(arrays)
            case['array_numpy']=measure(lambda:numpy_dressing(k,centers,h,dh))
            for backend in ('numpy','numba'):
                case['full_api_'+backend]=measure(lambda:(e.hamiltonian(k,backend=backend),e.gradient(k,backend=backend)))
            report['cases'].append(case)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args();report=run()
    text=json.dumps(report,indent=2)+'\n'
    if a.output:Path(a.output).write_text(text)
    print(text)
