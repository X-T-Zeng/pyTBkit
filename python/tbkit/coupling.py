"""Ordinary finite-group coupling and aligned multiplicity spaces.

Columns are coupled kets in the original basis. This is distinct from the
angular-momentum coupled_basis convention (coupled rows, uncoupled columns).
"""
from dataclasses import dataclass,replace
import numpy as np
import sympy as sp
from .groups import _integer
from .representations import MatrixRepresentation,_simplify,_zero,_array
from .group_kernels import matrix_unit_column


@dataclass(frozen=True)
class RepresentationDecomposition:
    representation: MatrixRepresentation
    irreps: tuple
    multiplicities: tuple
    basis: object
    tolerance: float
    tensor_shape: object=None

    @property
    def exact(self):return isinstance(self.basis,sp.MatrixBase)

    @property
    def slices(self):
        offset=0;result=[]
        for irrep,multiplicity in zip(self.irreps,self.multiplicities):
            stop=offset+irrep.dimension*multiplicity;result.append(slice(offset,stop));offset=stop
        return tuple(result)

    def copies(self,index):
        index=_integer(index,'irrep index')
        if index>=len(self.irreps):raise ValueError('irrep index out of range')
        start=self.slices[index].start;degree=self.irreps[index].dimension
        return tuple(self.basis[:,start+i*degree:start+(i+1)*degree] for i in range(self.multiplicities[index]))

    @property
    def coefficients(self):
        """C[a,b,coupled_column], for clebsch_gordan results only."""
        if self.tensor_shape is None:raise ValueError('this is not a tensor-product decomposition')
        shape=(*self.tensor_shape,self.representation.dimension)
        if self.exact:return sp.ImmutableDenseNDimArray(tuple(self.basis),shape)
        return self.basis.reshape(shape)

    def transform(self,operator):
        """U† A U; no Hermitization or claim that A is group invariant."""
        if self.exact and isinstance(operator,sp.MatrixBase):
            if operator.shape!=(self.representation.dimension,)*2:raise ValueError('operator shape mismatch')
            return sp.ImmutableMatrix((self.basis.adjoint()*operator*self.basis).applyfunc(_simplify))
        value=_array(operator)
        if value.shape!=(self.representation.dimension,)*2:raise ValueError('operator shape mismatch')
        basis=_array(self.basis.tolist()) if self.exact else self.basis
        return basis.conj().T@value@basis


def _exact_seeds(projector,multiplicity):
    vectors=[]
    for column in projector.columnspace():
        value=column
        for vector in vectors:value-=vector*(vector.adjoint()*value)[0]
        value=value.applyfunc(_simplify);norm=_simplify((value.adjoint()*value)[0])
        if norm==0:continue
        if norm.is_positive is not True:raise ValueError('positive exact seed norm not proved')
        vectors.append(sp.ImmutableMatrix((value/sp.sqrt(norm)).applyfunc(_simplify)))
    if len(vectors)!=multiplicity:raise ValueError('matrix-unit rank disagrees with character multiplicity')
    return vectors


def _exact_copies(target,irrep,multiplicity):
    degree=irrep.dimension;units=[]
    for i in range(degree):
        value=sp.zeros(target.dimension)
        for full,small in zip(target.matrices,irrep.matrices):value+=sp.conjugate(small[i,0])*full
        units.append(sp.ImmutableMatrix((value*sp.Rational(degree,target.group.order)).applyfunc(_simplify)))
    first=units[0]
    if not _zero(first.adjoint()-first) or not _zero(first*first-first):
        raise ValueError('exact matrix unit is not an orthogonal projector')
    return tuple(sp.ImmutableMatrix(sp.Matrix.hstack(*[unit*seed for unit in units]).applyfunc(_simplify))
                 for seed in _exact_seeds(first,multiplicity))


def _numeric_copies(target,irrep,multiplicity,tolerance,backend):
    full,small=target.as_arrays(),irrep.as_arrays();degree=irrep.dimension
    if backend=='numba':units=matrix_unit_column(full,small)
    else:
        weights=(degree/target.group.order)*small[:,:,0].conj().T
        units=(weights@full.reshape(target.group.order,-1)).reshape(degree,target.dimension,target.dimension)
    first=units[0]
    if not np.allclose(first,first.conj().T,atol=tolerance,rtol=0):raise ValueError('matrix unit is not Hermitian within tolerance')
    values,vectors=np.linalg.eigh(first)
    if np.max(np.minimum(abs(values),abs(values-1)))>tolerance:
        raise ValueError('matrix unit spectrum is not a projector within tolerance')
    seeds=vectors[:,values>.5]
    if seeds.shape[1]!=multiplicity:raise ValueError('matrix-unit rank disagrees with character multiplicity')
    return tuple(np.column_stack([unit@seeds[:,j] for unit in units]) for j in range(multiplicity))


def symmetry_adapted_basis(target,irreps,*,backend=None,tolerance=None,max_dimension=64):
    """Resolve a complete representation into aligned irrep copies.

    E_ij=(d/|G|) sum_g conjugate(D_alpha(g)[i,j]) D_target(g).
    Orthonormal seeds in range(E_00) produce Q_mu[:,i]=E_i0 seed_mu.
    Hence D_target(g) Q_mu=Q_mu D_alpha(g), not merely equal characters.

    Columns are ordered irrep, copy, internal irrep index. Arbitrary unitary
    rotations between equivalent copies remain a valid multiplicity gauge.
    Default exact mode requires constant exact SymPy representation matrices;
    symbolic operators transformed afterwards may retain free parameters.
    Numeric backends are numpy or numba. All results validate full isometry,
    intertwiners and completeness; no partial basis is returned.
    """
    if not isinstance(target,MatrixRepresentation):raise TypeError('MatrixRepresentation target required')
    irreps=tuple(irreps)
    if not irreps or any(not isinstance(r,MatrixRepresentation) for r in irreps):raise ValueError('nonempty irrep list required')
    if target.dimension>_integer(max_dimension,'max_dimension',1):raise ValueError('decomposition exceeds max_dimension')
    if tolerance is None:tolerance=max(target.tolerance,*(r.tolerance for r in irreps))
    if (isinstance(tolerance,(bool,np.bool_)) or not isinstance(tolerance,(int,float,np.integer,np.floating))
            or not np.isfinite(tolerance) or not 0<tolerance<.5):
        raise ValueError('finite tolerance between 0 and 0.5 required for disjoint projector clusters')
    exact=target.exact and all(r.exact for r in irreps)
    backend=('sympy' if exact else 'numba') if backend is None else backend
    if backend not in ('sympy','numpy','numba'):raise ValueError('backend must be sympy, numpy or numba')
    if backend=='sympy':
        if not exact:raise ValueError('SymPy coupling requires exact representations')
        if any(m.free_symbols for r in (target,*irreps) for m in r.matrices):
            raise ValueError('bind representation parameters before exact coupling; constant exact matrices required')
    multiplicities=target.decompose(irreps)
    copies=[]
    for irrep,multiplicity in zip(irreps,multiplicities):
        if not multiplicity:continue
        pieces=(_exact_copies(target,irrep,multiplicity) if backend=='sympy'
                else _numeric_copies(target,irrep,multiplicity,float(tolerance),backend))
        for piece in pieces:
            if backend=='sympy':
                if any(not _zero(full*piece-piece*small) for full,small in zip(target.matrices,irrep.matrices)):
                    raise ValueError('exact intertwiner identity not proved')
            elif not np.allclose(target.as_arrays()@piece,piece@irrep.as_arrays(),atol=tolerance,rtol=0):
                raise ValueError('intertwiner identity exceeds tolerance')
            copies.append(piece)
    if backend=='sympy':
        basis=sp.ImmutableMatrix(sp.Matrix.hstack(*copies))
        if basis.shape!=(target.dimension,)*2 or not _zero(basis.adjoint()*basis-sp.eye(target.dimension)):
            raise ValueError('complete exact coupling basis is not unitary')
    else:
        basis=np.ascontiguousarray(np.column_stack(copies))
        if basis.shape!=(target.dimension,)*2 or not np.allclose(basis.conj().T@basis,np.eye(target.dimension),atol=tolerance,rtol=0):
            raise ValueError('complete coupling basis is not unitary within tolerance')
        basis.setflags(write=False)
    return RepresentationDecomposition(target,irreps,multiplicities,basis,float(tolerance))


def clebsch_gordan(left,right,irreps,**options):
    """Finite-group tensor coefficients C[a,b,column]=<a,b|coupled column>.

    Row a*dim(right)+b follows numpy/SymPy Kronecker ordering. Ordinary complex
    irreps only; this does not impose an SU(2) Condon-Shortley phase convention.
    """
    if not isinstance(left,MatrixRepresentation):raise TypeError('MatrixRepresentation factors required')
    left._compatible(right)
    limit=_integer(options.get('max_dimension',64),'max_dimension',1)
    if left.dimension*right.dimension>limit:raise ValueError('tensor product exceeds max_dimension')
    result=symmetry_adapted_basis(left.tensor(right),irreps,**options)
    return replace(result,tensor_shape=(left.dimension,right.dimension))
