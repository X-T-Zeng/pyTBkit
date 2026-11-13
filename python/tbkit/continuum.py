"""Controlled finite basis representations for continuum Hamiltonians."""
from functools import lru_cache
import numpy as np
from .base import hermitian


def landau_hamiltonian(model,levels,*,magnetic_length=1.,field_sign=1,axes=(0,1),k_parallel=None):
    """Weyl-ordered HK in a truncated harmonic oscillator basis.

    [k_a,k_b]=i*field_sign/l_B^2 away from the truncation boundary.
    Order: oscillator index first, orbital second. Boundary-induced spurious
    levels must be checked by increasing levels; no SI conversion is assumed.
    """
    if not hasattr(model,'powers') or not isinstance(levels,int) or levels<2 or magnetic_length<=0 or field_sign not in (-1,1):
        raise ValueError("HK, levels>=2, l_B>0, field_sign=+/-1 required")
    a,b=axes
    if a==b or min(axes)<0 or max(axes)>=model.dim: raise ValueError("invalid magnetic axes")
    k=np.zeros(model.dim) if k_parallel is None else np.asarray(k_parallel,float)
    if k.shape!=(model.dim,): raise ValueError("k_parallel dimension mismatch")
    lower=np.diag(np.sqrt(np.arange(1,levels)),1)
    x=(lower+lower.T)/(np.sqrt(2)*magnetic_length)
    y=-1j*field_sign*(lower-lower.T)/(np.sqrt(2)*magnetic_length)
    eye=np.eye(levels)
    @lru_cache(None)
    def weyl(p,q):
        if p==q==0: return eye
        out=np.zeros((levels,levels),complex)
        if p: out+=p*x@weyl(p-1,q)
        if q: out+=q*y@weyl(p,q-1)
        return out/(p+q)
    out=np.zeros((levels*model.norb,)*2,complex)
    for powers,coefficient in zip(model.powers,model.coefficients):
        scalar=np.prod([k[d]**int(p) for d,p in enumerate(powers) if d not in axes])
        out+=np.kron(weyl(int(powers[a]),int(powers[b]))*scalar,coefficient)
    hermitian(out)
    return out


def plane_wave_hamiltonian(kinetic,k,reciprocal_vectors,couplings):
    """Plane-wave expansion with supplied momentum-space harmonics.

    reciprocal_vectors: list of g integer multi-indices and Cartesian G vectors
    as a mapping tuple(g)->G. couplings maps integer harmonic difference g-h
    to norb-square potential matrices. H_gh=delta_gh H0(k+G_g)+V[g-h].
    Requires both conjugate harmonics; invalid Hermitian data raise.
    """
    k=np.asarray(k,float); labels=list(reciprocal_vectors)
    if not labels: raise ValueError("nonempty plane-wave basis required")
    if len(set(map(len,labels)))!=1: raise ValueError("harmonic index dimensions differ")
    momenta=np.asarray([reciprocal_vectors[g] for g in labels],float)
    if momenta.shape!=(len(labels),kinetic.dim) or k.shape!=(kinetic.dim,): raise ValueError("momentum dimension mismatch")
    n=kinetic.norb;out=np.zeros((len(labels)*n,)*2,complex)
    diagonal=kinetic.hamiltonian(momenta+k)
    for i,g in enumerate(labels):
        for j,h in enumerate(labels):
            block=np.array(couplings.get(tuple(np.asarray(g)-h),np.zeros((n,n))),complex,copy=True)
            if block.shape!=(n,n): raise ValueError("harmonic potential dimension mismatch")
            if i==j: block+=diagonal[i]
            out[i*n:(i+1)*n,j*n:(j+1)*n]=block
    hermitian(out)
    return out
