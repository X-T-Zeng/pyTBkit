"""Explicit nonorthogonal-basis H c = E S c; no false orthogonal topology API."""
import numpy as np
from scipy.linalg import eigh
from .base import points,hermitian


class NonOrthogonal:
    def __init__(self,hamiltonian,overlap):
        if hamiltonian.dim!=overlap.dim or hamiltonian.norb!=overlap.norb:
            raise ValueError("H/S dimensions differ")
        if hasattr(hamiltonian,'lattice') and hasattr(overlap,'lattice') and not np.allclose(hamiltonian.lattice,overlap.lattice):
            raise ValueError("H/S lattices differ")
        self.h,self.s=hamiltonian,overlap
        self.dim,self.norb=hamiltonian.dim,hamiltonian.norb

    def bands(self,k,*,eigenvectors=False,backend='numba',chunk_size=256):
        k,single=points(k,self.dim)
        if chunk_size<1 or not len(k): raise ValueError("nonempty k and positive chunk size required")
        energies=[]; vectors=[]
        for start in range(0,len(k),chunk_size):
            kk=k[start:start+chunk_size]
            h=self.h.hamiltonian(kk,backend=backend); s=self.s.hamiltonian(kk,backend=backend)
            hermitian(h); hermitian(s)
            for hi,si in zip(h,s):
                if np.linalg.eigvalsh(si).min()<=0: raise ValueError("overlap must be positive definite")
                e,u=eigh(hi,si)
                energies.append(e); vectors.append(u)
        e=np.array(energies); u=np.array(vectors)
        if single: e=e[0];u=u[0]
        return (e,u) if eigenvectors else e

    def band_gradient(self,k,*,backend='numba',gap_tol=1e-9):
        """Hellmann-Feynman c†(dH-E dS)c, eigenvectors S-normalized."""
        e,u=self.bands(k,eigenvectors=True,backend=backend)
        if gap_tol<=0:raise ValueError("positive gap_tol required")
        gaps=abs(e[..., :,None]-e[...,None,:])
        if np.any((gaps<gap_tol)&~np.eye(self.norb,dtype=bool)):
            raise ValueError("band gradient at a degeneracy requires directional degenerate perturbation theory")
        dh=self.h.gradient(k,backend=backend); ds=self.s.gradient(k,backend=backend)
        hermitian(dh);hermitian(ds)
        diagonal_h=np.einsum('...in,...aij,...jn->...an',u.conj(),dh,u)
        diagonal_s=np.einsum('...in,...aij,...jn->...an',u.conj(),ds,u)
        return (diagonal_h-e[...,None,:]*diagonal_s).real
