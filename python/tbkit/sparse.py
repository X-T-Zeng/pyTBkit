"""CSR hopping storage with Numba COO kernels and sparse spectral solvers."""
import numpy as np
from scipy import sparse
from .hr import HR,immutable
from .base import TBkit,points
from . import backends


class SparseHR(TBkit):
    def __init__(self,terms,*,lattice=None,orbitals=None):
        if not terms: raise ValueError("nonempty mapping R -> CSR required")
        r=np.asarray(list(terms))
        if r.ndim!=2 or not 1<=r.shape[1]<=3 or not np.all(r==np.rint(r)):
            raise ValueError("integer vectors required")
        matrices=[sparse.csr_matrix(h,dtype=complex,copy=True) for h in terms.values()]
        n=matrices[0].shape[0]; self.dim=r.shape[1]; self.norb=n
        if n<1 or any(h.shape!=(n,n) for h in matrices): raise ValueError("equal square matrices required")
        lattice=np.eye(self.dim) if lattice is None else np.asarray(lattice,float)
        if lattice.shape!=(self.dim,self.dim) or abs(np.linalg.det(lattice))<1e-12: raise ValueError("invertible lattice required")
        self.lattice=immutable(lattice,float); self.vectors=immutable(r,np.int64)
        self.cartesian_vectors=immutable(r@lattice,float)
        self.orbitals=immutable(np.zeros((n,self.dim)) if orbitals is None else orbitals,float)
        if self.orbitals.shape!=(n,self.dim): raise ValueError("orbital coordinate mismatch")
        rows=[]; cols=[]; values=[]; ridx=[]
        for index,h in enumerate(matrices):
            h.sum_duplicates(); h.eliminate_zeros(); coo=h.tocoo()
            rows.extend(coo.row);cols.extend(coo.col);values.extend(coo.data);ridx.extend([index]*h.nnz)
            h.data.setflags(write=False);h.indices.setflags(write=False);h.indptr.setflags(write=False)
        self.hopping=tuple(matrices)
        self.rows=immutable(rows,np.int64);self.cols=immutable(cols,np.int64)
        self.values=immutable(values,np.complex128); self.rindex=immutable(ridx,np.int64)

    @classmethod
    def from_dense(cls,model,*,threshold=0.):
        if threshold<0: raise ValueError("nonnegative threshold required")
        return cls({tuple(r):sparse.csr_matrix(np.where(abs(h)>threshold,h,0)) for r,h in zip(model.vectors,model.hopping)},lattice=model.lattice,orbitals=model.orbitals)

    def sparse_hamiltonian(self,k):
        k,single=points(k,self.dim)
        if not single: raise ValueError("one k point required")
        out=sparse.csr_matrix((self.norb,self.norb),dtype=complex)
        for phase,h in zip(np.exp(1j*(self.cartesian_vectors@k[0])),self.hopping): out=out+phase*h
        return out

    def hamiltonian(self,k,*,backend='numba'):
        k,single=points(k,self.dim)
        if backend=='numba': out=backends.coo_fourier(k,self.cartesian_vectors,self.rindex,self.rows,self.cols,self.values,self.norb)
        elif backend=='numpy': out=np.array([self.sparse_hamiltonian(q).toarray() for q in k])
        else: raise ValueError("backend must be numba or numpy")
        return out[0] if single else out

    def gradient(self,k,*,backend='numba'):
        k,single=points(k,self.dim)
        if backend=='numba': out=backends.coo_gradient(k,self.cartesian_vectors,self.rindex,self.rows,self.cols,self.values,self.norb)
        elif backend=='numpy':
            out=np.zeros((len(k),self.dim,self.norb,self.norb),complex)
            for q,point in enumerate(k):
                for r,h in zip(self.cartesian_vectors,self.hopping):
                    for a in range(self.dim): out[q,a]+=1j*r[a]*np.exp(1j*r@point)*h.toarray()
        else: raise ValueError("backend must be numba or numpy")
        return out[0] if single else out

    def to_dense(self,*,max_bytes=1_000_000_000):
        if len(self.vectors)*self.norb**2*16>max_bytes: raise MemoryError("dense hopping exceeds max_bytes")
        return HR(self.vectors,[h.toarray() for h in self.hopping],lattice=self.lattice,orbitals=self.orbitals)

    def supercell(self,matrix):
        from .construction import sparse_supercell
        return sparse_supercell(self,matrix)

    def cut(self,cells,*,periodic=None):
        from .construction import sparse_cut
        return sparse_cut(self,cells,periodic=periodic)

    def slab(self,layers,axis=0):
        if not isinstance(layers,int) or layers<1 or not 0<=axis<self.dim:raise ValueError("valid layer count/axis required")
        repeats=np.ones(self.dim,int);repeats[axis]=layers;large=self.supercell(repeats)
        terms={tuple(r):h for r,h in zip(large.vectors,large.hopping) if r[axis]==0}
        if not terms:terms={(0,)*self.dim:sparse.csr_matrix((large.norb,large.norb),dtype=complex)}
        return SparseHR(terms,lattice=large.lattice,orbitals=large.orbitals)
