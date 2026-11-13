"""Compare actual sparse storage and JIT interpolation; no inferred MATLAB timing."""
import json
import time
import numpy as np
import numba
from scipy import sparse
from threadpoolctl import threadpool_limits
from tbkit.sparse import SparseHR


def measure(function,repeats=5):
    samples=[]
    for _ in range(repeats):
        t=time.perf_counter();function();samples.append(time.perf_counter()-t)
    return float(np.median(samples))


def run():
    n=64
    m=SparseHR({(0,):sparse.diags(np.linspace(-1,1,n)),(1,):sparse.diags(np.ones(n-1),1),(-1,):sparse.diags(np.ones(n-1),-1)})
    dense=m.to_dense();k=np.linspace(-np.pi,np.pi,256)[:,None]
    with threadpool_limits(limits=1):
        a=m.hamiltonian(k);b=dense.hamiltonian(k)
        np.testing.assert_allclose(a,b,atol=1e-13)
        sparse_time=measure(lambda:m.hamiltonian(k))
        dense_time=measure(lambda:dense.hamiltonian(k))
    # A separate large model exercises sparse shift-invert without dense blocks.
    large=SparseHR({(0,):sparse.diags(np.arange(800.)+.1),(1,):.2*sparse.eye(800),(-1,):.2*sparse.eye(800)})
    large.sparse_bands([.2],6,sigma=5)
    csr_bytes=sum(h.data.nbytes+h.indices.nbytes+h.indptr.nbytes for h in large.hopping)
    dense_bytes=len(large.vectors)*large.norb**2*16
    return {'numba_threads':numba.get_num_threads(),'blas_threads':1,'interpolation':{'norb':n,'nk':len(k),'sparse_numba_s':sparse_time,'dense_numba_s':dense_time,'dense_over_sparse':dense_time/sparse_time},'large_storage':{'norb':800,'nnz':len(large.values),'csr_bytes':csr_bytes,'coo_jit_bytes':large.values.nbytes+large.rows.nbytes+large.cols.nbytes+large.rindex.nbytes,'equivalent_dense_hopping_bytes':dense_bytes,'sparse_eigsh_median_s':measure(lambda:large.sparse_bands([.2],6,sigma=5),3)}}

if __name__=='__main__':print(json.dumps(run(),indent=2))
