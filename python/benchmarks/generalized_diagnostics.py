"""Only diagnostics use the array kernel; eigensolving remains LAPACK."""
import json
import platform
import time
import numpy as np
import numba
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from tbkit import HK, NonOrthogonal
from tbkit.generalized_kernels import generalized_diagnostics
from tbkit.nonorthogonal import generalized_diagnostics_numpy


def measure(function,repeats=5):
    wall=[];cpu=[]
    for _ in range(repeats):
        start=time.perf_counter();process=time.process_time();function()
        wall.append(time.perf_counter()-start);cpu.append(time.process_time()-process)
    return {'median_wall_s':float(np.median(wall)),'median_process_cpu_s':float(np.median(cpu)),
            'wall_samples_s':wall,'process_cpu_samples_s':cpu}


def run():
    seed=20261118;rng=np.random.default_rng(seed);results=[]
    a=np.eye(2,dtype=complex)[None];e=np.ones((1,2));fresh=numba.njit(cache=False,nogil=True)(generalized_diagnostics.py_func)
    with threadpool_limits(limits=1):
        start=time.perf_counter();process=time.process_time();fresh(a,a,e,a)
        first={'wall_s':time.perf_counter()-start,'process_cpu_s':time.process_time()-process}
        for batch,n in ((512,4),(64,32)):
            b=np.eye(n)+.03*(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))
            metric=b.conj().T@b;metric/=max(abs(metric.real).max(),abs(metric.imag).max())
            s=np.tile(metric,(batch,1,1));h=np.empty_like(s);u=np.empty_like(s);energy=np.empty((batch,n))
            for q in range(batch):
                z=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
                h[q]=(z+z.conj().T)/max(1.,abs(z).max())
                energy[q],u[q]=eigh(h[q],s[q])
            reference=generalized_diagnostics_numpy(h,s,energy,u);actual=generalized_diagnostics(h,s,energy,u)
            errors=[float(np.max(abs(x-y))) for x,y in zip(actual,reference)]
            for x,y in zip(actual,reference):np.testing.assert_allclose(x,y,rtol=2e-13,atol=2e-14)
            kernel=measure(lambda:generalized_diagnostics(h,s,energy,u))
            numpy=measure(lambda:generalized_diagnostics_numpy(h,s,energy,u))
            # A separate end-to-end spectrum includes model evaluation, the
            # condition spectrum, LAPACK solve, checks and result ownership.
            model=NonOrthogonal(HK({(0,):h[0]}),HK({(0,):s[0]}));k=np.zeros((min(batch,64),1))
            model.spectrum(k);model.spectrum(k,backend='numpy')
            workflow_numba=measure(lambda:model.spectrum(k),repeats=3)
            workflow_numpy=measure(lambda:model.spectrum(k,backend='numpy'),repeats=3)
            results.append({'batch':batch,'norb':n,'max_residual_ratio_difference':errors[0],
                            'max_metric_error_difference':errors[1],'numba_diagnostics':kernel,'numpy_diagnostics':numpy,
                            'numpy_over_numba_wall':numpy['median_wall_s']/kernel['median_wall_s'],
                            'workflow_batch':len(k),'whole_spectrum_numba':workflow_numba,'whole_spectrum_numpy':workflow_numpy})
    return {'platform':platform.platform(),'numpy':np.__version__,'numba':numba.__version__,'seed':seed,
            'numba_threads':numba.get_num_threads(),'blas_threads':1,'kernel_parallel':False,
            'fresh_compile_and_first_call':first,'cases':results,
            'scope':'diagnostic arrays and separate checked spectrum workflow; no all-toolkit or MATLAB speedup claim'}


if __name__=='__main__':print(json.dumps(run(),indent=2))
