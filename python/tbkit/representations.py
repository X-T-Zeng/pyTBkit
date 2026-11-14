"""Ordinary unitary finite-group representations and exact character algebra.

Antiunitary corepresentations and projective factor systems are deliberately
excluded from ordinary character inner products and projectors.
"""
from dataclasses import dataclass
import numpy as np
import sympy as sp
from .groups import FiniteGroup,_integer
from .group_kernels import character_projector


def _simplify(expression):
    expression=sp.sympify(expression)
    # Exact roots of unity need their real/imaginary form for sums such as
    # 1+exp(2*pi*i/3)+exp(4*pi*i/3). No Float recovery or rounding is used.
    return sp.simplify(sp.expand_complex(expression) if not expression.free_symbols else expression)


def _zero(matrix):return all(_simplify(x)==0 for x in matrix)


def _array(value):
    try:a=np.asarray(value,dtype=complex)
    except (TypeError,ValueError) as exc:raise ValueError('bind symbolic representation parameters before numeric evaluation') from exc
    if not np.isfinite(a).all():raise ValueError('finite representation matrices required')
    return a


@dataclass(frozen=True,init=False,eq=False)
class MatrixRepresentation:
    """D(a*b)=D(a)D(b), with columns transforming as ket coefficients.

    Every group pair and unitarity relation is checked. SymPy matrices infer
    exact=True; exact mode rejects Float atoms and requires provable identities.
    Numeric mode checks finite matrices against an explicit tolerance.
    """
    group: FiniteGroup
    matrices: object
    dimension: int
    exact: bool
    tolerance: float
    name: str

    def __init__(self,group,matrices,*,exact=None,tolerance=1e-10,name='D'):
        if not isinstance(group,FiniteGroup):raise TypeError('validated FiniteGroup required')
        data=tuple(matrices)
        if len(data)!=group.order:raise ValueError('one matrix per indexed group element required')
        if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('positive finite tolerance required')
        exact=all(isinstance(m,sp.MatrixBase) for m in data) if exact is None else bool(exact)
        if exact:
            values=tuple(sp.ImmutableMatrix(m) for m in data)
            if any(m.atoms(sp.Float) for m in values):raise ValueError('exact representation contains floats; use exact SymPy input or exact=False')
            n=values[0].rows
            if n<1 or any(m.shape!=(n,n) for m in values):raise ValueError('equal nonempty square matrices required')
            if any(not _zero(m.adjoint()*m-sp.eye(n)) for m in values):raise ValueError('unitarity not proved with supplied symbolic assumptions')
            if any(not _zero(values[a]*values[b]-values[group.table[a,b]]) for a in range(group.order) for b in range(group.order)):
                raise ValueError('ordinary representation multiplication law failed; preserve central phases instead of using an action quotient')
        else:
            values=_array(data)
            if values.ndim!=3 or values.shape[1]<1 or values.shape[1]!=values.shape[2]:raise ValueError('equal nonempty square matrices required')
            n=values.shape[1]
            if not np.allclose(values.conj().swapaxes(-1,-2)@values,np.eye(n),atol=tolerance,rtol=0):raise ValueError('unitary matrices required')
            if any(not np.allclose(values[a]@values[b],values[group.table[a,b]],atol=tolerance,rtol=0) for a in range(group.order) for b in range(group.order)):
                raise ValueError('ordinary representation multiplication law failed; preserve central phases instead of using an action quotient')
            values=np.array(values,dtype=complex,order='C',copy=True);values.setflags(write=False)
        for key,value in dict(group=group,matrices=values,dimension=n,exact=exact,tolerance=float(tolerance),name=str(name)).items():
            object.__setattr__(self,key,value)

    @classmethod
    def from_operators(cls,group,operators,**kwargs):
        from .operators import Oper
        operators=tuple(operators)
        if any(not isinstance(op,Oper) for op in operators):raise TypeError('Oper elements required')
        if any(op.antiunitary for op in operators):
            raise ValueError('antiunitary operations require corepresentations, not ordinary characters')
        matrices=[op._symbolic_unitary if op._symbolic_unitary is not None else op.unitary for op in operators]
        return cls(group,matrices,**kwargs)

    def __repr__(self):return f'MatrixRepresentation({self.name!r}, dim={self.dimension}, group={self.group.name!r}, exact={self.exact})'
    def matrix(self,element):return self.matrices[self.group.index(element)]

    def as_arrays(self,parameters=None):
        """Read-only contiguous complex128 matrices for NumPy/Numba kernels."""
        if not self.exact:
            if parameters:raise ValueError('numeric representation has no symbolic parameters')
            return self.matrices
        values=_array([m.subs(parameters or {},simultaneous=True).tolist() for m in self.matrices])
        if parameters:
            # A substitution can violate the assumptions used in an exact
            # proof (e.g. assigning an imaginary value to a real angle).
            return MatrixRepresentation(self.group,values,exact=False,tolerance=self.tolerance,name=self.name).matrices
        values=np.ascontiguousarray(values);values.setflags(write=False)
        return values

    def characters(self,*,by_class=False):
        result=tuple(_simplify(sp.trace(m)) for m in self.matrices) if self.exact else np.trace(self.matrices,axis1=1,axis2=2)
        if by_class:result=tuple(result[self.group.index(c[0])] for c in self.group.conjugacy_classes())
        if self.exact:return tuple(result)
        result=np.array(result,dtype=complex,copy=True);result.setflags(write=False);return result

    def _compatible(self,other):
        if not isinstance(other,MatrixRepresentation) or self.group is not other.group:
            raise ValueError('representations must share the same indexed FiniteGroup object')

    def inner(self,other):
        """Character inner product <self,other>; not a matrix-vector overlap."""
        self._compatible(other)
        a,b=self.characters(),other.characters()
        if self.exact and other.exact:
            return _simplify(sum(sp.conjugate(x)*y for x,y in zip(a,b))/self.group.order)
        return complex(np.vdot(_array(a),_array(b))/self.group.order)

    def is_irreducible(self):
        value=self.inner(self)
        return value==1 if self.exact else bool(abs(value-1)<=self.tolerance)

    def frobenius_schur(self):
        """Irrep type: +1 real, 0 complex, -1 quaternionic (ordinary groups)."""
        if not self.is_irreducible():raise ValueError('Frobenius-Schur type requires an irrep')
        chars=self.characters()
        values=[chars[self.group.table[g,g]] for g in range(self.group.order)]
        if self.exact:
            value=_simplify(sum(values)/self.group.order)
            if value not in (-1,0,1):raise ValueError('Frobenius-Schur indicator not proved')
            return int(value)
        value=complex(sum(values)/self.group.order)
        nearest=min((-1,0,1),key=lambda i:abs(value-i))
        if abs(value-nearest)>self.tolerance:raise ValueError('Frobenius-Schur indicator exceeds tolerance')
        return nearest

    def multiplicity(self,irrep):
        self._compatible(irrep)
        if not irrep.is_irreducible():raise ValueError('irreducible target representation required')
        value=irrep.inner(self)
        if self.exact and irrep.exact:
            if value.is_integer is not True or value<0:raise ValueError('nonnegative integer multiplicity not proved')
            return int(value)
        number=complex(value);integer=int(round(number.real))
        if integer<0 or abs(number-integer)>max(self.tolerance,irrep.tolerance):raise ValueError('character multiplicity is not an integer within tolerance')
        return integer

    def decompose(self,irreps,*,require_complete=True):
        irreps=tuple(irreps)
        if not irreps:raise ValueError('nonempty irrep list required')
        for i,a in enumerate(irreps):
            self._compatible(a)
            if any(abs(complex(a.inner(b)))>max(a.tolerance,b.tolerance) for b in irreps[:i]):
                raise ValueError('duplicate/equivalent irreps in decomposition')
        multiplicities=tuple(self.multiplicity(irrep) for irrep in irreps)
        if require_complete and sum(m*r.dimension for m,r in zip(multiplicities,irreps))!=self.dimension:
            raise ValueError('irrep list does not exhaust this representation')
        return multiplicities

    def projector(self,irrep,*,backend=None):
        """Central isotypic projector, retaining every copy of the irrep.

        This does not choose a basis within an irrep or split repeated copies.
        """
        self.multiplicity(irrep)  # validates same group and irreducibility
        backend=('sympy' if self.exact and irrep.exact else 'numba') if backend is None else backend
        characters=irrep.characters()
        if backend=='sympy':
            if not self.exact or not irrep.exact:raise ValueError('SymPy projector requires exact representations')
            out=sp.zeros(self.dimension)
            for c,m in zip(characters,self.matrices):out+=sp.conjugate(c)*m
            return sp.ImmutableMatrix((out*sp.Rational(irrep.dimension,self.group.order)).applyfunc(_simplify))
        matrices=self.as_arrays();characters=np.ascontiguousarray(_array(characters))
        if backend=='numba':return character_projector(matrices,characters,irrep.dimension)
        if backend=='numpy':return irrep.dimension/self.group.order*np.einsum('g,gij->ij',characters.conj(),matrices)
        raise ValueError('backend must be sympy, numpy or numba')

    def tensor(self,other):
        self._compatible(other)
        if self.exact and other.exact:
            matrices=[sp.kronecker_product(a,b) for a,b in zip(self.matrices,other.matrices)]
        else:matrices=[np.kron(a,b) for a,b in zip(self.as_arrays(),other.as_arrays())]
        return MatrixRepresentation(self.group,matrices,tolerance=max(self.tolerance,other.tolerance),name=f'{self.name} tensor {other.name}')

    def direct_sum(self,other):
        self._compatible(other)
        if self.exact and other.exact:matrices=[sp.diag(a,b) for a,b in zip(self.matrices,other.matrices)]
        else:
            from scipy.linalg import block_diag
            matrices=[block_diag(a,b) for a,b in zip(self.as_arrays(),other.as_arrays())]
        return MatrixRepresentation(self.group,matrices,tolerance=max(self.tolerance,other.tolerance),name=f'{self.name} + {other.name}')

    def restrict(self,subgroup):
        if isinstance(subgroup,FiniteGroup):self.group.subgroup(subgroup)  # validates embedding
        else:subgroup=self.group.subgroup(subgroup)
        return MatrixRepresentation(subgroup,[self.matrix(g) for g in subgroup],exact=self.exact,tolerance=self.tolerance,name=f'{self.name} restricted')

    def on_subspace(self,basis):
        """Restrict to an orthonormal invariant column space, e.g. a multiplet.

        Characters are traces of the resulting block, not individual diagonal
        entries in an arbitrary degenerate eigenbasis. Noninvariant spaces fail.
        This does not determine a Bloch little group or supply missing sewing.
        """
        if self.exact and isinstance(basis,sp.MatrixBase):
            q=sp.ImmutableMatrix(basis)
            if (q.rows!=self.dimension or q.cols<1 or q.cols>q.rows or q.atoms(sp.Float)
                    or not _zero(q.adjoint()*q-sp.eye(q.cols))):
                raise ValueError('exact orthonormal subspace columns required')
            matrices=[sp.ImmutableMatrix((q.adjoint()*m*q).applyfunc(_simplify)) for m in self.matrices]
            if any(not _zero(m*q-q*small) for m,small in zip(self.matrices,matrices)):
                raise ValueError('subspace is not invariant under the group')
        else:
            q=_array(basis)
            if (q.ndim!=2 or q.shape[0]!=self.dimension or q.shape[1]<1 or q.shape[1]>q.shape[0]
                    or not np.allclose(q.conj().T@q,np.eye(q.shape[1]),atol=self.tolerance,rtol=0)):
                raise ValueError('orthonormal subspace columns required')
            full=self.as_arrays();matrices=q.conj().T@full@q
            if not np.allclose(full@q,q@matrices,atol=self.tolerance,rtol=0):
                raise ValueError('subspace is not invariant under the group')
        return MatrixRepresentation(self.group,matrices,tolerance=self.tolerance,name=f'{self.name} subspace')

    def _repr_latex_(self):
        return '$'+r'\chi_{'+sp.latex(sp.Symbol(self.name))+r'}='+sp.latex(sp.ImmutableMatrix([self.characters(by_class=True)]))+'$'


def matrix_group(generators,*,exact=None,tolerance=1e-10,max_order=128,name='matrix group'):
    """Close unitary matrices under exact equality or numeric tolerance, never phase.

    Returns the faithful defining representation and its validated finite group.
    Exact closure accepts constant SymPy matrices; infinite groups hit max_order.
    """
    limit=_integer(max_order,'max_order',1);generators=tuple(generators)
    if not generators:raise ValueError('nonempty matrix generators required')
    exact=all(isinstance(m,sp.MatrixBase) for m in generators) if exact is None else bool(exact)
    if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('positive finite tolerance required')
    if exact:
        gens=tuple(sp.ImmutableMatrix(m) for m in generators);n=gens[0].rows;identity=sp.ImmutableMatrix(sp.eye(n))
        if n<1 or any(m.shape!=(n,n) or m.free_symbols or m.atoms(sp.Float) for m in gens):
            raise ValueError('constant exact square matrices without floats required')
        if any(not _zero(m.adjoint()*m-sp.eye(n)) for m in gens):raise ValueError('unitary generators required')
        multiply=lambda a,b:sp.ImmutableMatrix((a*b).applyfunc(_simplify))
        equal=lambda a,b:_zero(a-b)
    else:
        gens=_array(generators)
        if gens.ndim!=3 or gens.shape[1]<1 or gens.shape[1]!=gens.shape[2]:raise ValueError('equal square generators required')
        n=gens.shape[1];identity=np.eye(n,dtype=complex)
        if not np.allclose(gens.conj().swapaxes(-1,-2)@gens,identity,atol=tolerance,rtol=0):raise ValueError('unitary generators required')
        multiply=lambda a,b:a@b
        equal=lambda a,b:np.allclose(a,b,atol=tolerance,rtol=0)
    elements=[identity];cursor=0
    def locate(m):return next((i for i,a in enumerate(elements) if equal(m,a)),None)
    while cursor<len(elements):
        a=elements[cursor];cursor+=1
        for b in gens:
            c=multiply(a,b)
            if locate(c) is None:
                if len(elements)>=limit:raise ValueError('matrix group does not close within max_order')
                elements.append(c)
    table=[]
    for a in elements:
        row=[locate(multiply(a,b)) for b in elements]
        if None in row:raise ValueError('matrix products are not closed within tolerance')
        table.append(row)
    group=FiniteGroup(table,name=name,max_order=limit)
    return MatrixRepresentation(group,elements,exact=exact,tolerance=tolerance,name='defining')


@dataclass(frozen=True,init=False)
class CharacterTable:
    """Validated ordinary irrep table; supply irreps or explicitly generate numerically."""
    group: FiniteGroup
    irreps: tuple
    classes: tuple
    values: sp.ImmutableMatrix

    @classmethod
    def from_group(cls,group,**options):
        """Numerically generate ordinary irreps; the returned table is not exact."""
        from .irreps import irreducible_representations
        return cls(irreducible_representations(group,**options))

    def __init__(self,irreps):
        irreps=tuple(irreps)
        if not irreps or any(not isinstance(r,MatrixRepresentation) for r in irreps):raise ValueError('nonempty irreducible representation list required')
        group=irreps[0].group;classes=group.conjugacy_classes()
        if len(irreps)!=len(classes):raise ValueError('one inequivalent irrep per conjugacy class required for a complete table')
        if len({r.name for r in irreps})!=len(irreps):raise ValueError('distinct irrep names required for display')
        for i,r in enumerate(irreps):
            irreps[0]._compatible(r)
            if not r.is_irreducible():raise ValueError('table row is not irreducible')
            for other in irreps[:i]:
                if abs(complex(r.inner(other)))>max(r.tolerance,other.tolerance):raise ValueError('table rows are not orthogonal')
        if sum(r.dimension**2 for r in irreps)!=group.order:raise ValueError('irrep dimensions do not satisfy sum d**2=|G|')
        values=sp.ImmutableMatrix([list(r.characters(by_class=True)) for r in irreps])
        for key,value in dict(group=group,irreps=irreps,classes=classes,values=values).items():object.__setattr__(self,key,value)

    def latex(self):
        headers=['']+[r'C_{'+str(i)+r'}\ (|C|='+str(len(c))+')' for i,c in enumerate(self.classes)]
        rows=[' & '.join(headers)]
        rows+=[' & '.join([sp.latex(sp.Symbol(r.name))]+[sp.latex(x) for x in self.values.row(i)]) for i,r in enumerate(self.irreps)]
        return r'\begin{array}{c|'+('c'*len(self.classes))+'}'+rows[0]+r'\\\hline '+r'\\'.join(rows[1:])+r'\end{array}'

    def _repr_latex_(self):return '$'+self.latex()+'$'
