"""Explicit orbital Bloch embedding for point-orbital tight-binding models.

Hamiltonian-only Kubo geometry is coefficient-frame geometry. A material's
full Wannier position/dipole connection is additional data, not recoverable
from hr.dat or Wannier centers alone. This wrapper assumes point orbitals.
"""
import numpy as np
from .base import TBkit,points
from .hr import immutable


def model_frame(model):
    """Explicit representation tag, preserved by built-in model algebra."""
    frame=getattr(model,'orbital_frame','model')
    if frame not in ('model','point-orbital'):
        raise ValueError('unknown orbital frame')
    return frame


def checked_centers(orbitals,norb,dim):
    """Finite real FRACTIONAL centers; do not wrap or identify integer shifts."""
    if np.iscomplexobj(orbitals):raise ValueError('orbital centers must be real')
    centers=np.asarray(orbitals,float)
    if centers.shape!=(norb,dim) or not np.isfinite(centers).all():
        raise ValueError('finite orbital centers (norb,dim) required')
    return immutable(centers,float)


class OrbitalEmbedding(TBkit):
    """H_emb(k)=D(k)^dagger H_periodic(k) D(k), D_jj=exp(i k.tau_j).

    Local Berry geometry can be computed from this model's analytic gradient.
    Fractional intracell centers make this frame nonperiodic in the BZ;
    closed reciprocal Wilson loops require its boundary_sewing(G).
    """
    requires_boundary_sewing=True
    orbital_frame='point-orbital'

    def __init__(self,model):
        if model_frame(model)=='point-orbital':
            raise ValueError('model is already in the point-orbital frame; use embed_orbitals')
        if getattr(model,'requires_boundary_sewing',False):
            raise ValueError('embedding requires a coefficient frame with no preexisting boundary sewing')
        if not hasattr(model,'lattice') or not hasattr(model,'orbitals'):
            raise ValueError('explicit lattice and orbital centers required')
        self.model=model;self.dim,self.norb=model.dim,model.norb
        if np.iscomplexobj(model.lattice):raise ValueError('real lattice required')
        self.lattice=immutable(model.lattice,float)
        if self.lattice.shape!=(self.dim,self.dim) or not np.isfinite(self.lattice).all():
            raise ValueError('finite square lattice required')
        self.orbitals=checked_centers(model.orbitals,self.norb,self.dim)
        self.centers=immutable(self.orbitals@self.lattice,float)
        if not np.isfinite(self.centers).all():raise ValueError('Cartesian orbital centers not representable')

    def hamiltonian(self,k,*,backend='numba'):
        k,single=points(k,self.dim)
        h=self.model.hamiltonian(k,backend=backend)
        if backend=='numba':
            from .gauge_kernels import embed_hamiltonian
            out=embed_hamiltonian(k,self.centers,np.ascontiguousarray(h))
        else:
            phase=np.exp(1j*(k@self.centers.T))
            out=phase.conj()[:,:,None]*h*phase[:,None,:]
        return out[0] if single else out

    def gradient(self,k,*,backend='numba'):
        k,single=points(k,self.dim)
        h=self.model.hamiltonian(k,backend=backend);dh=self.model.gradient(k,backend=backend)
        if backend=='numba':
            from .gauge_kernels import embed_gradient
            out=embed_gradient(k,self.centers,np.ascontiguousarray(h),np.ascontiguousarray(dh))
        else:
            phase=np.exp(1j*(k@self.centers.T))
            displacement=self.centers.T[:,None,:]-self.centers.T[:,:,None]
            covariant=dh+1j*displacement[None]*h[:,None]
            out=phase.conj()[:,None,:,None]*covariant*phase[:,None,None,:]
        return out[0] if single else out

    def boundary_sewing(self,reciprocal_shift):
        """S_G satisfies H_emb(k+G)=S_G H_emb(k) S_G^dagger."""
        if np.iscomplexobj(reciprocal_shift):raise ValueError('real reciprocal lattice vector required')
        g=np.asarray(reciprocal_shift,float)
        if g.shape!=(self.dim,) or not np.isfinite(g).all():
            raise ValueError('a finite Cartesian reciprocal lattice vector is required')
        with np.errstate(over='ignore',invalid='ignore'):
            fractional=g@self.lattice.T/(2*np.pi)
        if not np.isfinite(fractional).all() or not np.allclose(fractional,np.rint(fractional),atol=1e-10,rtol=0):
            raise ValueError('a Cartesian reciprocal lattice vector is required')
        return np.diag(np.exp(-1j*(self.centers@g)))


def embed_orbitals(model):
    """Apply the point-orbital frame exactly once, including algebra wrappers."""
    return model if model_frame(model)=='point-orbital' else OrbitalEmbedding(model)
