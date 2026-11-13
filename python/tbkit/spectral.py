"""Caller-owned, immutable orthogonal-basis spectral snapshots.

No model object or global model cache is retained. A snapshot intentionally
represents the model at preparation time, even if the caller later edits it.
Store/reuse one manageable k batch; these arrays are not an out-of-core cache.
"""
from dataclasses import dataclass
import numpy as np
from .base import TBkit,points,hermitian
from .hr import immutable
from .gauge import OrbitalEmbedding,embed_orbitals


@dataclass(frozen=True)
class SpectralBatch:
    """Batched k, E, orthonormal eigenvectors and band-basis dH/dk matrices.

    Shapes: (nk,dim), (nk,n), (nk,n,n), (nk,dim,n,n). Eigenvectors are
    columns; velocity_matrices have gradient units, not gradient/hbar.
    """
    k: np.ndarray
    energy: np.ndarray
    eigenvectors: np.ndarray
    velocity_matrices: np.ndarray
    single: bool=False
    frame: str='model'

    def __post_init__(self):
        if np.iscomplexobj(self.k) or np.iscomplexobj(self.energy):
            raise ValueError('real k points and energy eigenvalues required')
        k=immutable(self.k,float);e=immutable(self.energy,float)
        u=immutable(self.eigenvectors,np.complex128);v=immutable(self.velocity_matrices,np.complex128)
        if k.ndim!=2 or not len(k) or k.shape[1]<1 or e.ndim!=2 or len(e)!=len(k) or e.shape[1]<1:
            raise ValueError('nonempty batched k and energy arrays required')
        nk,n=e.shape
        if u.shape!=(nk,n,n) or v.shape!=(nk,k.shape[1],n,n):
            raise ValueError('spectral matrix shapes do not match k and energies')
        if self.single and nk!=1:raise ValueError('single-point snapshot must contain one k point')
        if self.frame not in ('model','point-orbital'):raise ValueError('unknown spectral frame')
        hermitian(v)
        if not np.allclose(u.conj().swapaxes(-1,-2)@u,np.eye(n),atol=1e-9,rtol=0):
            raise ValueError('orthonormal eigenvectors required; generalized bases need their metric/connection')
        for name,value in (('k',k),('energy',e),('eigenvectors',u),('velocity_matrices',v)):
            object.__setattr__(self,name,value)

    @property
    def dim(self):return self.k.shape[1]

    @property
    def norb(self):return self.energy.shape[1]

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.k,self.energy,self.eigenvectors,self.velocity_matrices))


def prepare_spectral(model,k,*,backend='numba',frame='model',chunk_size=2048):
    """Diagonalize once and retain band-basis gradients for response reuse.

    frame='model' uses the supplied model's derivative convention. For HR this
    is periodic coefficient geometry. frame='point-orbital' explicitly applies
    the point-orbital embedding, including intracell position commutators.
    Neither option reconstructs missing Wannier dipole/position matrices.
    """
    if not isinstance(model,TBkit):
        raise TypeError('orthogonal TBkit model required; nonorthogonal response needs a metric/connection formulation')
    if frame not in ('model','point-orbital'):raise ValueError('frame must be model or point-orbital')
    if frame=='point-orbital' and not isinstance(model,OrbitalEmbedding):model=embed_orbitals(model)
    k,single=points(k,model.dim)
    if len(k)==0:raise ValueError('nonempty k batch required')
    e,u=model.bands(k,eigenvectors=True,backend=backend,chunk_size=chunk_size)
    gradient=model.gradient(k,backend=backend)
    hermitian(gradient)
    if backend=='numba':
        from .backends import band_velocities
        v=band_velocities(u,gradient)
    else:
        v=u.conj().swapaxes(-1,-2)[:,None]@gradient@u[:,None]
    return SpectralBatch(k,e,u,v,single=single,frame='point-orbital' if isinstance(model,OrbitalEmbedding) else 'model')
