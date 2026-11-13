"""Explicit orbital Bloch embedding for point-orbital tight-binding models.

Hamiltonian-only Kubo geometry is coefficient-frame geometry. A material's
full Wannier position/dipole connection is additional data, not recoverable
from hr.dat or Wannier centers alone. This wrapper assumes point orbitals.
"""
import numpy as np
from .base import TBkit,points
from .hr import immutable


class OrbitalEmbedding(TBkit):
    """H_emb(k)=D(k)^dagger H_periodic(k) D(k), D_jj=exp(i k.tau_j).

    Local Berry geometry can be computed from this model's analytic gradient.
    Fractional intracell centers make this frame nonperiodic in the BZ;
    closed reciprocal Wilson loops require its boundary_sewing(G).
    """
    requires_boundary_sewing=True

    def __init__(self,model):
        if not hasattr(model,'lattice') or not hasattr(model,'orbitals'):
            raise ValueError('explicit lattice and orbital centers required')
        self.model=model;self.dim,self.norb=model.dim,model.norb
        self.lattice=immutable(model.lattice,float);self.orbitals=immutable(model.orbitals,float)
        self.centers=immutable(self.orbitals@self.lattice,float)

    def hamiltonian(self,k,*,backend='numba'):
        k,single=points(k,self.dim);phase=np.exp(1j*(k@self.centers.T))
        h=self.model.hamiltonian(k,backend=backend)
        out=phase.conj()[:,:,None]*h*phase[:,None,:]
        return out[0] if single else out

    def gradient(self,k,*,backend='numba'):
        k,single=points(k,self.dim);phase=np.exp(1j*(k@self.centers.T))
        h=self.model.hamiltonian(k,backend=backend);dh=self.model.gradient(k,backend=backend)
        displacement=self.centers.T[:,None,:]-self.centers.T[:,:,None]
        covariant=dh+1j*displacement[None]*h[:,None]
        out=phase.conj()[:,None,:,None]*covariant*phase[:,None,None,:]
        return out[0] if single else out

    def boundary_sewing(self,reciprocal_shift):
        """S_G satisfies H_emb(k+G)=S_G H_emb(k) S_G^dagger."""
        g=np.asarray(reciprocal_shift,float)
        if g.shape!=(self.dim,) or not np.allclose(g@self.lattice.T/(2*np.pi),np.rint(g@self.lattice.T/(2*np.pi)),atol=1e-10):
            raise ValueError('a Cartesian reciprocal lattice vector is required')
        return np.diag(np.exp(-1j*(self.centers@g)))


def embed_orbitals(model):return OrbitalEmbedding(model)
