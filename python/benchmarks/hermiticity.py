"""Independent NumPy diagnostics versus Numba; compilation timed separately."""
import json
import platform
import time
import numpy as np
import numba
from threadpoolctl import threadpool_limits
from tbkit.base import hermitian
from tbkit.validation_kernels import hermiticity_metrics


def measure(function):
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter()-start)
    return float(np.median(samples))


def reference(a):
    # Independent LAPACK-style/vectorized norms, ordinary-scale workload only.
    return np.linalg.norm(a-a.conj().swapaxes(-1, -2), axis=(-2, -1))/np.linalg.norm(a, axis=(-2, -1))


def run():
    rng = np.random.default_rng(20261117)
    tiny = np.ascontiguousarray(rng.normal(size=(2, 2, 2))+1j*rng.normal(size=(2, 2, 2)))
    fresh = numba.njit(cache=False, nogil=True)(hermiticity_metrics.py_func)
    start = time.perf_counter()
    fresh(tiny)
    compile_s = time.perf_counter()-start
    results = []
    with threadpool_limits(limits=1):
        for batch, n in ((4096, 8), (32, 128)):
            a = np.ascontiguousarray(rng.normal(size=(batch, n, n))+1j*rng.normal(size=(batch, n, n)))
            scales, norms, defects, exact = hermiticity_metrics(a)
            expected = reference(a)
            error = float(np.max(abs(defects/norms-expected)))
            np.testing.assert_allclose(defects/norms, expected, rtol=2e-14, atol=0)
            h = (a+a.conj().swapaxes(-1, -2))*.5
            hermitian(h)
            compiled = measure(lambda: hermiticity_metrics(a))
            numpy_s = measure(lambda: reference(a))
            results.append({'batch': batch, 'norb': n, 'max_absolute_ratio_error': error,
                            'warm_numba_metrics_s': compiled, 'warm_numpy_metrics_s': numpy_s,
                            'numpy_over_numba': numpy_s/compiled,
                            'warm_validating_wrapper_s': measure(lambda: hermitian(h))})
    return {'platform': platform.platform(), 'numpy': np.__version__, 'numba': numba.__version__,
            'seed': 20261117, 'numba_threads': numba.get_num_threads(), 'blas_threads': 1,
            'kernel_parallel': False, 'fresh_compile_and_first_call_s': compile_s,
            'cases': results, 'scope': 'diagnostic workload; not whole bands or MATLAB speedup'}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
