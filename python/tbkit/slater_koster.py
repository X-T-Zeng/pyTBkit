"""Two-center s/p/d Slater-Koster hopping via bond-axis orbital rotations."""
import numpy as np
from .orbitals import ORBITALS,orbital_rotation
from .hr import HR

PARAMETERS=('ss_sigma','sp_sigma','sd_sigma','pp_sigma','pp_pi','pd_sigma','pd_pi','dd_sigma','dd_pi','dd_delta')


def slater_koster(displacement,parameters):
    """Full 9x9 directed hopping in ORBITALS order.

    Real two-center integrals; odd/even l cross terms are antisymmetric on
    reversing the bond: T(-d)=T(d)^T. All missing integrals are zero.
    """
    unknown=set(parameters)-set(PARAMETERS)
    if unknown: raise ValueError(f'unknown SK parameters: {sorted(unknown)}')
    d=np.asarray(displacement,float)
    if d.shape!=(3,) or not np.isfinite(d).all() or np.linalg.norm(d)==0: raise ValueError("nonzero bond required")
    z=d/np.linalg.norm(d)
    seed=np.eye(3)[np.argmin(np.abs(z))]
    x=np.cross(seed,z); x/=np.linalg.norm(x); y=np.cross(z,x)
    rotation=orbital_rotation(np.column_stack([x,y,z]))
    p={key:float(parameters.get(key,0.)) for key in PARAMETERS}
    t=np.zeros((9,9))
    t[0,0]=p['ss_sigma']
    t[0,3]=p['sp_sigma']; t[3,0]=-p['sp_sigma']
    t[0,8]=t[8,0]=p['sd_sigma']
    t[1,1]=t[2,2]=p['pp_pi']; t[3,3]=p['pp_sigma']
    t[3,8]=p['pd_sigma']; t[8,3]=-p['pd_sigma']
    t[1,6]=t[2,5]=p['pd_pi']; t[6,1]=t[5,2]=-p['pd_pi']
    t[8,8]=p['dd_sigma']; t[5,5]=t[6,6]=p['dd_pi']; t[4,4]=t[7,7]=p['dd_delta']
    return rotation@t@rotation.T


def from_structure(structure,orbitals,cutoff,parameters,*,onsite=None,periodic=(True,True,True)):
    """Build HR from atom-wise orbital names and all directed neighbor bonds.

    parameters(distance, species_i, species_j) -> integral mapping; or one
    mapping for a homogeneous system. For heteroatomic directed parameters,
    the caller must ensure reverse-bond Hermiticity. onsite may be one vector
    of diagonal energies per orbital or a complete on-site matrix.
    """
    if len(orbitals)!=len(structure.positions): raise ValueError("one orbital list per atom required")
    indices=[]
    for names in orbitals:
        if len(set(names))!=len(names) or not names: raise ValueError("unique nonempty orbitals required per atom")
        try: indices.append([ORBITALS.index(name) for name in names])
        except ValueError as exc: raise ValueError("only real s/p/d orbitals are supported") from exc
    offsets=np.r_[0,np.cumsum([len(o) for o in indices])]; n=int(offsets[-1])
    terms={(0,0,0):np.zeros((n,n),complex)}
    if onsite is not None:
        a=np.asarray(onsite,complex)
        if a.shape==(n,): a=np.diag(a)
        if a.shape!=(n,n): raise ValueError("onsite dimension mismatch")
        terms[(0,0,0)]+=a
    bonds=structure.neighbors(cutoff,periodic=periodic)
    for i,j,r,d,distance in zip(bonds.i,bonds.j,bonds.cells,bonds.displacement,bonds.distance):
        p=parameters(distance,structure.species[i],structure.species[j]) if callable(parameters) else parameters
        block=slater_koster(d,p)[np.ix_(indices[i],indices[j])]
        key=tuple(r)
        if key not in terms: terms[key]=np.zeros((n,n),complex)
        terms[key][offsets[i]:offsets[i+1],offsets[j]:offsets[j+1]]+=block
    centers=np.concatenate([np.repeat(pos[None,:],len(names),axis=0) for pos,names in zip(structure.positions,indices)])
    model=HR(list(terms),list(terms.values()),lattice=structure.lattice,orbitals=centers)
    from .base import hermitian
    # Check coefficients individually, not only accidental equality at Gamma.
    blocks={tuple(r):h for r,h in zip(model.vectors,model.hopping)}
    for r,h in blocks.items():
        if not np.allclose(h,np.asarray(blocks.get(tuple(-x for x in r),np.zeros_like(h))).conj().T):
            raise ValueError("SK callback generated non-Hermitian reverse bonds")
    hermitian(model.hamiltonian([0,0,0],backend='numpy'))
    return model
