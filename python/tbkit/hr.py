"""Real-space tight binding: H(k) = sum_R H_R exp(+i k.R).

lattice rows are real lattice vectors. k defaults to Cartesian (radians /
length); fractional k uses reciprocal coordinates with an explicit 2*pi.
"""
import itertools
import numpy as np
from scipy import sparse
from .base import TBkit, points
from . import backends


def immutable(a, dtype):
    a = np.array(a, dtype=dtype, order="C", copy=True)
    if not np.isfinite(a).all():
        raise ValueError("nonfinite model data")
    a.setflags(write=False)
    return a


class HR(TBkit):
    def __init__(self, vectors, hopping, *, lattice=None, orbitals=None):
        r = np.asarray(vectors)
        h = np.asarray(hopping)
        if r.ndim != 2 or not 1 <= r.shape[1] <= 3 or not np.allclose(r, np.rint(r), atol=1e-10, rtol=0):
            raise ValueError("vectors must be an integer (nr,dim) array, dim=1..3")
        if h.ndim != 3 or h.shape[0] != len(r) or h.shape[1] != h.shape[2] or h.shape[1] == 0:
            raise ValueError("hopping must have shape (nr,norb,norb)")
        self.dim, self.norb = r.shape[1], h.shape[1]
        if len(r)==0:
            r=np.zeros((1,self.dim),int)
            h=np.zeros((1,self.norb,self.norb),complex)
        lattice = np.eye(self.dim) if lattice is None else lattice
        if np.shape(lattice) != (self.dim, self.dim) or abs(np.linalg.det(lattice)) < 1e-14:
            raise ValueError("lattice must be square and invertible")
        unique, inv = np.unique(np.rint(r).astype(int), axis=0, return_inverse=True)
        combined = np.zeros((len(unique), self.norb, self.norb), complex)
        np.add.at(combined, inv, h)
        self.vectors = immutable(unique, np.int64)
        self.hopping = immutable(combined, np.complex128)
        self.lattice = immutable(lattice, float)
        self.cartesian_vectors = immutable(self.vectors @ self.lattice, float)
        if orbitals is None:
            orbitals = np.zeros((self.norb, self.dim))
        if np.shape(orbitals) != (self.norb, self.dim):
            raise ValueError("orbitals must be fractional coordinates (norb,dim)")
        self.orbitals = immutable(orbitals, float)

    @classmethod
    def from_hoppings(cls, norb, terms, *, dim=3, lattice=None):
        """terms: mapping R -> matrix; an empty mapping creates a zero model."""
        if not terms:
            terms = {(0,)*dim: np.zeros((norb,norb))}
        h = np.asarray(list(terms.values()))
        if h.shape[1:] != (norb,norb):
            raise ValueError("norb and matrix sizes differ")
        return cls(list(terms), h, lattice=lattice)

    def with_hopping(self, vector, matrix, *, add=False, conjugate=False):
        """Return a new model; compiled arrays cannot become stale after edits."""
        r = np.asarray(vector)
        if r.shape != (self.dim,) or not np.allclose(r,np.rint(r),atol=1e-10,rtol=0):
            raise ValueError("integer vector of model dimension required")
        matrix = np.asarray(matrix, complex)
        if matrix.shape != (self.norb,self.norb):
            raise ValueError("wrong hopping matrix shape")
        terms = {tuple(r): h.copy() for r,h in zip(self.vectors,self.hopping)}
        r = tuple(np.rint(r).astype(int))
        terms[r] = terms.get(r,0)+matrix if add else matrix
        if conjugate:
            minus = tuple(-x for x in r)
            if minus == r:
                from .base import hermitian
                hermitian(terms[r])
            else:
                terms[minus] = terms[r].conj().T
        return HR(list(terms),list(terms.values()),lattice=self.lattice,orbitals=self.orbitals)

    def _vectors(self, coordinates):
        if coordinates == "cartesian": return self.cartesian_vectors
        if coordinates == "fractional": return np.ascontiguousarray(2*np.pi*self.vectors)
        raise ValueError("coordinates must be cartesian or fractional")

    def hamiltonian(self, k, *, backend="numba", coordinates="cartesian"):
        k, single = points(k,self.dim)
        r = self._vectors(coordinates)
        if backend == "numba": out = backends.fourier(k,r,self.hopping)
        elif backend == "fortran":
            from .native import evaluate
            out = evaluate('fourier',k,r,self.hopping)
        elif backend == "numpy":
            out = (np.exp(1j*(k@r.T)) @ self.hopping.reshape(len(r),-1)).reshape(-1,self.norb,self.norb)
        else: raise ValueError("HR backend must be numba, numpy or fortran")
        return out[0] if single else out

    def gradient(self, k, *, backend="numba", coordinates="cartesian"):
        k, single = points(k,self.dim)
        r = self._vectors(coordinates)
        if backend == "numba": out = backends.fourier_gradient(k,r,self.hopping)
        elif backend == "fortran":
            from .native import evaluate
            out = evaluate('gradient',k,r,self.hopping)
        elif backend == "numpy":
            out = np.einsum('kr,ra,rij->kaij',np.exp(1j*(k@r.T)),1j*r,self.hopping,optimize=True)
        else: raise ValueError("HR backend must be numba, numpy or fortran")
        return out[0] if single else out

    def sparse_hamiltonian(self, k):
        k, single = points(k,self.dim)
        if not single: raise ValueError("one k point required")
        out = sparse.csr_matrix((self.norb,self.norb),dtype=complex)
        for phase,h in zip(np.exp(1j*(self.cartesian_vectors@k[0])),self.hopping):
            out += sparse.csr_matrix(h)*phase
        return out

    def hermitize(self):
        terms = {tuple(r):h for r,h in zip(self.vectors,self.hopping)}
        keys = set(terms) | {tuple(-x for x in r) for r in terms}
        result = {r:(terms.get(r,0)+np.asarray(terms.get(tuple(-x for x in r),np.zeros((self.norb,self.norb)))).conj().T)/2 for r in keys}
        return HR(list(result),list(result.values()),lattice=self.lattice,orbitals=self.orbitals)

    def supercell(self, repeats):
        """Diagonal supercell. Arbitrary integer transformation matrices are not supported."""
        repeats = np.asarray(repeats)
        if repeats.ndim == 2:
            from .construction import supercell
            return supercell(self,repeats)
        if repeats.shape != (self.dim,) or np.any(repeats<1) or not np.all(repeats==np.rint(repeats)):
            raise ValueError("positive integer repeats of length dim required")
        repeats = repeats.astype(int)
        cells = list(itertools.product(*(range(n) for n in repeats)))
        index = {c:i for i,c in enumerate(cells)}
        n = len(cells)*self.norb
        terms = {}
        for cidx,c in enumerate(cells):
            for r,h in zip(self.vectors,self.hopping):
                dest = np.asarray(c)+r
                shift = tuple(dest//repeats)
                jidx = index[tuple(dest%repeats)]
                if shift not in terms: terms[shift]=np.zeros((n,n),complex)
                terms[shift][cidx*self.norb:(cidx+1)*self.norb,jidx*self.norb:(jidx+1)*self.norb] += h
        orb = np.concatenate([(self.orbitals+c)/repeats for c in cells])
        return HR(list(terms),list(terms.values()),lattice=repeats[:,None]*self.lattice,orbitals=orb)

    def cut(self, cells, *, periodic=None):
        """Finite sparse Hamiltonian; periodic=True wraps the specified axis."""
        cells = np.asarray(cells)
        if cells.shape != (self.dim,) or np.any(cells<1) or not np.all(cells==np.rint(cells)):
            raise ValueError("positive integer cell counts required")
        cells=cells.astype(int)
        periodic = np.zeros(self.dim,bool) if periodic is None else np.asarray(periodic,bool)
        if periodic.shape != (self.dim,): raise ValueError("wrong periodic shape")
        positions=list(itertools.product(*(range(n) for n in cells)))
        index={c:i for i,c in enumerate(positions)}
        rows,cols,data=[],[],[]
        for ci,c in enumerate(positions):
            for r,h in zip(self.vectors,self.hopping):
                dest=np.asarray(c)+r
                if np.any(((dest<0)|(dest>=cells)) & ~periodic): continue
                cj=index[tuple(dest%cells)]
                ii,jj=np.nonzero(h)
                rows.extend(ci*self.norb+ii); cols.extend(cj*self.norb+jj); data.extend(h[ii,jj])
        return sparse.coo_matrix((data,(rows,cols)),shape=(len(positions)*self.norb,)*2).tocsr()

    def to_hk(self, order=2, center=None):
        """Taylor expansion in Cartesian displacement q=k-center."""
        from math import factorial
        from .symbolic import HK
        if not isinstance(order,int) or order<0: raise ValueError("nonnegative integer order required")
        center=np.zeros(self.dim) if center is None else np.asarray(center,float)
        if center.shape != (self.dim,): raise ValueError("wrong expansion center")
        terms={}
        phase=np.exp(1j*(self.cartesian_vectors@center))
        for power in itertools.product(range(order+1),repeat=self.dim):
            if sum(power)>order: continue
            factor=phase.copy()
            for a,p in enumerate(power): factor *= (1j*self.cartesian_vectors[:,a])**p/factorial(p)
            terms[power]=np.einsum('r,rij->ij',factor,self.hopping)
        return HK(terms)

    def to_htrig(self):
        from .symbolic import Htrig
        return Htrig(self.cartesian_vectors,self.hopping,1j*self.hopping,lattice=self.lattice,orbitals=self.orbitals)

    def save(self,path):
        from .io import save_hdf5
        save_hdf5(self,path)

    @classmethod
    def from_wannier90(cls,path,**kwargs):
        from .io import read_wannier90
        return read_wannier90(path,**kwargs)
