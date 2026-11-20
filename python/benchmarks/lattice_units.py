"""Setup/search costs and existing Fourier backends after unit-domain fixes.

No new Fourier algorithm or MATLAB speed comparison is claimed. Compilation,
warm API execution and cell/neighbor preparation are measured separately.
"""
import json
import os
import platform
import time
import numpy as np
import numba
from threadpoolctl import threadpool_limits
from tbkit import HR,Structure,backends


def measure(function):
    wall=[];cpu=[]
    for _ in range(5):
        t=time.perf_counter();p=time.process_time();function()
        wall.append(time.perf_counter()-t);cpu.append(time.process_time()-p)
    return {'median_wall_s':float(np.median(wall)),'median_process_cpu_s':float(np.median(cpu)),
            'wall_samples_s':wall,'process_cpu_samples_s':cpu}


def run():
    seed=1729;rng=np.random.default_rng(seed)
    r=rng.integers(-3,4,(31,3));h=.02*(rng.normal(size=(31,8,8))+1j*rng.normal(size=(31,8,8)))
    model=HR(r,h).hermitize();k=rng.uniform(-np.pi,np.pi,(1024,3))
    positions=np.indices((4,4,4)).reshape(3,-1).T/4
    cases=[]
    with threadpool_limits(limits=1,user_api='blas'):
        fresh=numba.njit(cache=False,nogil=True,parallel=True)(backends.fourier.py_func)
        t=time.perf_counter();p=time.process_time();fresh(k[:2],model.cartesian_vectors,model.hopping)
        first={'wall_s':time.perf_counter()-t,'process_cpu_s':time.process_time()-p}
        reference=model.hamiltonian(k,backend='numpy');gradient=model.gradient(k,backend='numpy')
        reference_bonds=None
        for a in (1.,1e-12):
            scaled=HR(model.vectors,model.hopping,lattice=a*np.eye(3));kk=k/a
            actual=scaled.hamiltonian(kk);numpy_value=scaled.hamiltonian(kk,backend='numpy')
            g=scaled.gradient(kk);normalized_gradient=g.real/a+1j*(g.imag/a)
            np.testing.assert_allclose(actual,reference,rtol=2e-13,atol=2e-14)
            np.testing.assert_allclose(actual,numpy_value,rtol=2e-13,atol=2e-14)
            np.testing.assert_allclose(normalized_gradient,gradient,rtol=3e-13,atol=3e-14)
            structure=Structure(4*a*np.eye(3),positions,['X']*len(positions))
            neighbors=structure.neighbors(1.1*a)
            signature=np.column_stack((neighbors.i,neighbors.j,neighbors.cells))
            assert len(neighbors.i)==6*len(positions)
            np.testing.assert_allclose(neighbors.distance/a,1.,rtol=2e-14,atol=0)
            if reference_bonds is None:reference_bonds=signature
            else:np.testing.assert_array_equal(signature,reference_bonds)
            cases.append({'length_scale':a,'nk':len(k),'norb':8,'nr':len(model.vectors),
                          'max_absolute_h_error':float(np.max(abs(actual-reference))),
                          'max_absolute_scaled_gradient_error':float(np.max(abs(normalized_gradient-gradient))),
                          'constructor':measure(lambda:HR(model.vectors,model.hopping,lattice=a*np.eye(3))),
                          'warm_numba_hamiltonian':measure(lambda:scaled.hamiltonian(kk)),
                          'warm_numpy_hamiltonian':measure(lambda:scaled.hamiltonian(kk,backend='numpy')),
                          'neighbor_search':measure(lambda:structure.neighbors(1.1*a)),
                          'atoms':len(positions),'directed_bonds':len(neighbors.i)})
    return {'platform':platform.platform(),'numpy':np.__version__,'numba':numba.__version__,'seed':seed,
            'numba_threads':numba.get_num_threads(),'blas_threads':1,'omp_environment':os.environ.get('OMP_NUM_THREADS'),
            'fresh_fourier_compile_and_first_call':first,'cases':cases,
            'scope':'existing Fourier kernels unchanged; cell validation is setup work, KD-tree search remains SciPy/Python; no universal speedup claim'}


if __name__=='__main__':print(json.dumps(run(),indent=2))
