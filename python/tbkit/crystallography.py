"""Optional spglib-backed space groups, standard cells and atom mappings."""
from dataclasses import dataclass
import numpy as np
from .structure import Structure
from .hr import immutable


def _cell(structure):
    labels=list(dict.fromkeys(structure.species))
    numbers=[labels.index(s)+1 for s in structure.species]
    return (structure.lattice,structure.positions,numbers),labels


def _spglib():
    try:import spglib
    except ImportError as exc:raise ImportError('install tbkit-python[crystal] for space-group operations') from exc
    return spglib


@dataclass(frozen=True)
class CrystalOperation:
    rotation: np.ndarray
    translation: np.ndarray
    cartesian_rotation: np.ndarray
    permutation: np.ndarray
    cell_shifts: np.ndarray


def spacegroup(structure,*,symprec=1e-5,angle_tolerance=-1.):
    """Return international symbol/number, Hall symbol and dataset atom labels."""
    if symprec<=0:raise ValueError("positive symprec required")
    cell,_=_cell(structure)
    dataset=_spglib().get_symmetry_dataset(cell,symprec=symprec,angle_tolerance=angle_tolerance)
    if dataset is None:raise ValueError('spglib could not identify this structure')
    return {'number':int(dataset.number),'international':dataset.international,'hall':dataset.hall,
            'wyckoffs':tuple(dataset.wyckoffs),'equivalent_atoms':np.array(dataset.equivalent_atoms),
            'rotations':np.array(dataset.rotations),'translations':np.array(dataset.translations)}


def crystal_operations(structure,*,symprec=1e-5):
    data=spacegroup(structure,symprec=symprec);operations=[]
    lattice=structure.lattice
    for r,t in zip(data['rotations'],data['translations']):
        transformed=structure.positions@r.T+t;permutation=[];shifts=[]
        for i,p in enumerate(transformed):
            matches=[]
            for j,q in enumerate(structure.positions):
                if structure.species[i]!=structure.species[j]:continue
                cell=np.rint(p-q).astype(int)
                if np.linalg.norm((p-q-cell)@lattice)<symprec*2:matches.append((j,cell))
            if len(matches)!=1:raise ValueError('ambiguous atom mapping; duplicate/near-coincident sites')
            j,cell=matches[0];permutation.append(j);shifts.append(cell)
        if len(set(permutation))!=len(permutation):raise ValueError('non-bijective symmetry mapping')
        cart=lattice.T@r@np.linalg.inv(lattice.T)
        operations.append(CrystalOperation(immutable(r,np.int64),immutable(t,float),immutable(cart,float),immutable(permutation,np.int64),immutable(shifts,np.int64)))
    return operations


def standardize(structure,*,primitive=False,idealize=True,symprec=1e-5):
    cell,labels=_cell(structure)
    result=_spglib().standardize_cell(cell,to_primitive=primitive,no_idealize=not idealize,symprec=symprec)
    if result is None:raise ValueError('spglib could not standardize this structure')
    lattice,positions,numbers=result
    return Structure(lattice,positions,[labels[int(n)-1] for n in numbers])


def _orbital_blocks(structure,operation,orbitals,spinful):
    from .orbitals import ORBITALS,orbital_rotation,spin_rotation
    from scipy.spatial.transform import Rotation
    if len(orbitals)!=len(structure.positions):raise ValueError("one orbital list per atom required")
    if sorted(operation.permutation.tolist())!=list(range(len(orbitals))):raise ValueError("operation atom mapping invalid")
    spin=2 if spinful else 1
    counts=np.array([len(o)*spin for o in orbitals]);offsets=np.r_[0,np.cumsum(counts)]
    full=orbital_rotation(operation.cartesian_rotation)
    if spinful:
        axial=np.linalg.det(operation.cartesian_rotation)*operation.cartesian_rotation
        vector=Rotation.from_matrix(axial).as_rotvec();angle=np.linalg.norm(vector)
        su2=np.eye(2) if angle<1e-12 else spin_rotation(vector,angle)
    else:su2=np.ones((1,1))
    blocks=[]
    for i,j in enumerate(operation.permutation):
        try:rows=[ORBITALS.index(o) for o in orbitals[j]];cols=[ORBITALS.index(o) for o in orbitals[i]]
        except ValueError as exc:raise ValueError("only real s/p/d orbital sewing is supported") from exc
        block=np.kron(full[np.ix_(rows,cols)],su2)
        if block.shape[0]!=block.shape[1] or not np.allclose(block.conj().T@block,np.eye(len(block)),atol=1e-9):
            raise ValueError("orbital set is not closed under this crystal operation")
        blocks.append(block)
    return offsets,blocks


def sewing_matrix(structure,operation,orbitals,k_fractional,*,spinful=False):
    """Periodic-gauge U_g(k), H(gk)=U_g(k) H(k) U_g(k)^dagger.

    gk_frac = k_frac @ inverse(rotation_fractional). Cell shifts generate
    exp(-2pi i gk.n_atom), including nonsymmorphic translation phases.
    Basis order is atom, real orbital, optional spin up/down.
    """
    from .base import points
    k,single=points(k_fractional,3)
    offsets,blocks=_orbital_blocks(structure,operation,orbitals,spinful)
    target=k@np.linalg.inv(operation.rotation)
    n=int(offsets[-1]);out=np.zeros((len(k),n,n),complex)
    for i,j in enumerate(operation.permutation):
        phase=np.exp(-2j*np.pi*(target@operation.cell_shifts[i]))
        out[:,offsets[j]:offsets[j+1],offsets[i]:offsets[i+1]]=phase[:,None,None]*blocks[i]
    return out[0] if single else out


def crystal_residual(model,structure,operation,orbitals,k_fractional,*,spinful=False,backend='numba'):
    if model.dim!=3 or not np.allclose(model.lattice,structure.lattice):raise ValueError("model and structure lattices differ")
    reciprocal=2*np.pi*np.linalg.inv(structure.lattice).T
    k=np.asarray(k_fractional,float)
    target=k@np.linalg.inv(operation.rotation)
    u=sewing_matrix(structure,operation,orbitals,k,spinful=spinful)
    if u.shape[-1]!=model.norb:raise ValueError("orbital/model sizes differ")
    left=model.hamiltonian(target@reciprocal,backend=backend)
    right=u@model.hamiltonian(k@reciprocal,backend=backend)@u.conj().swapaxes(-1,-2)
    return float(np.max(abs(left-right)))


def symmetrize_hr(model,structure,operations,orbitals,*,spinful=False,drop_tol=0.):
    """Average real-space HR under supplied complete crystal operations.

    Maps R -> R rotation^T + n_j - n_i and each atom-orbital block by its
    local representation. Pass a full space-group coset list, not generators.
    """
    from .hr import HR
    if model.dim!=3 or not np.allclose(model.lattice,structure.lattice) or not operations:raise ValueError("3D matching model and nonempty operations required")
    if drop_tol<0 or not np.isfinite(drop_tol):raise ValueError('finite nonnegative drop_tol required')
    terms={}
    for operation in operations:
        offsets,blocks=_orbital_blocks(structure,operation,orbitals,spinful)
        if offsets[-1]!=model.norb:raise ValueError("orbital/model size mismatch")
        for r,h in zip(model.vectors,model.hopping):
            transformed=r@operation.rotation.T
            for i,ii in enumerate(operation.permutation):
                for j,jj in enumerate(operation.permutation):
                    block=blocks[i]@h[offsets[i]:offsets[i+1],offsets[j]:offsets[j+1]]@blocks[j].conj().T
                    if not np.any(abs(block)>drop_tol):continue
                    key=tuple(transformed+operation.cell_shifts[j]-operation.cell_shifts[i])
                    if key not in terms:terms[key]=np.zeros((model.norb,model.norb),complex)
                    terms[key][offsets[ii]:offsets[ii+1],offsets[jj]:offsets[jj+1]]+=block/len(operations)
    if not terms:terms={(0,0,0):np.zeros((model.norb,model.norb),complex)}
    return HR(list(terms),list(terms.values()),lattice=model.lattice,orbitals=model.orbitals)
