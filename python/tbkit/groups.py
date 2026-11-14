"""Validated finite groups, distinct from individual symmetry operators.

Elements are hashable labels; the immutable integer Cayley table owns the
operation. No equivalence up to phase or automatic projective quotient occurs.
"""
from abc import ABC,abstractmethod
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
from .group_kernels import associative


def _integer(value,name,minimum=0):
    if not isinstance(value,(int,np.integer)) or isinstance(value,(bool,np.bool_)) or value<minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


class AbstractFiniteGroup(ABC):
    """Extension contract: finite elements, identity, multiplication and inverse.

    An Oper is one transformation, not a finite collection. It therefore does
    not inherit this container contract. Custom implementations can be checked
    and materialized with FiniteGroup.from_group.
    """
    @property
    @abstractmethod
    def elements(self):...

    @property
    @abstractmethod
    def identity(self):...

    @abstractmethod
    def multiply(self,a,b):...

    @abstractmethod
    def inverse(self,a):...

    @property
    def order(self):return len(self.elements)
    def __len__(self):return self.order
    def __iter__(self):return iter(self.elements)


@dataclass(frozen=True,init=False,eq=False)
class FiniteGroup(AbstractFiniteGroup):
    """Finite group with fully checked closure, identity, Latin law and associativity.

    table[i,j] indexes the element elements[i]*elements[j]. Labels and their
    ordering are preserved, including identities not at index zero.
    """
    _elements: tuple
    _table: np.ndarray
    _identity: object
    _inverse: np.ndarray
    _lookup: object
    name: str

    def __init__(self,table,elements=None,*,name='G',max_order=256):
        limit=_integer(max_order,'max_order',1)
        a=np.asarray(table)
        if a.ndim!=2 or a.shape[0]!=a.shape[1] or not len(a) or len(a)>limit:
            raise ValueError('nonempty square Cayley table within max_order required')
        n=len(a)
        if (a.dtype.kind not in 'iuf' or not np.isfinite(a).all() or np.any(a<0)
                or np.any(a>=n) or not np.all(a==np.rint(a))):
            raise ValueError('Cayley entries must be integer element indices in range')
        a=np.array(a,dtype=np.int64,order='C',copy=True);a.setflags(write=False)
        expected=np.arange(n)
        if not np.all(np.sort(a,axis=0)==expected[:,None]) or not np.all(np.sort(a,axis=1)==expected):
            raise ValueError('every Cayley row and column must be a permutation')
        identities=np.flatnonzero(np.all(a==expected,axis=1)&np.all(a==expected[:,None],axis=0))
        if len(identities)!=1:raise ValueError('unique two-sided identity required')
        e=int(identities[0])
        if not associative(a):raise ValueError('Cayley operation is not associative')
        labels=tuple(range(n)) if elements is None else tuple(elements)
        try:lookup={label:i for i,label in enumerate(labels)}
        except TypeError as exc:raise ValueError('hashable element labels required') from exc
        if len(labels)!=n or len(lookup)!=n:raise ValueError('one distinct label per table element required')
        inverse=np.array([np.flatnonzero((a[i]==e)&(a[:,i]==e))[0] for i in range(n)],dtype=np.int64)
        inverse.setflags(write=False)
        for key,value in {'_elements':labels,'_table':a,'_identity':labels[e],
                          '_inverse':inverse,'_lookup':MappingProxyType(lookup),'name':str(name)}.items():
            object.__setattr__(self,key,value)

    @property
    def elements(self):return self._elements
    @property
    def identity(self):return self._identity
    @property
    def table(self):return self._table

    def __repr__(self):return f'FiniteGroup({self.name!r}, order={self.order})'

    def index(self,element):
        try:return self._lookup[element]
        except (KeyError,TypeError) as exc:raise ValueError('element does not belong to this group') from exc

    def multiply(self,a,b):return self.elements[self.table[self.index(a),self.index(b)]]
    def inverse(self,a):return self.elements[self._inverse[self.index(a)]]

    @classmethod
    def from_group(cls,group,*,max_order=256,name=None):
        if not isinstance(group,AbstractFiniteGroup):raise TypeError('AbstractFiniteGroup required')
        elements=tuple(group.elements)
        if not elements or len(elements)>_integer(max_order,'max_order',1):raise ValueError('group exceeds max_order or is empty')
        try:lookup={g:i for i,g in enumerate(elements)}
        except TypeError as exc:raise ValueError('hashable labels required') from exc
        try:table=[[lookup[group.multiply(a,b)] for b in elements] for a in elements]
        except (KeyError,TypeError) as exc:raise ValueError('group multiplication is not closed') from exc
        result=cls(table,elements,name=name or getattr(group,'name','G'),max_order=max_order)
        if result.identity!=group.identity or any(result.inverse(g)!=group.inverse(g) for g in elements):
            raise ValueError('declared identity/inverse disagrees with multiplication')
        return result

    def power(self,a,exponent):
        if not isinstance(exponent,(int,np.integer)) or isinstance(exponent,(bool,np.bool_)):
            raise ValueError('integer power required')
        self.index(a);n=int(exponent)
        if n<0:a=self.inverse(a);n=-n
        out=self.identity
        while n:
            if n&1:out=self.multiply(out,a)
            a=self.multiply(a,a);n//=2
        return out

    def element_order(self,a):
        i=self.index(a);e=self.index(self.identity);p=e
        for order in range(1,self.order+1):
            p=self.table[p,i]
            if p==e:return order
        raise RuntimeError('validated finite group has no element order')

    @property
    def is_abelian(self):return bool(np.array_equal(self.table,self.table.T))

    def conjugacy_classes(self):
        remaining=set(range(self.order));classes=[]
        while remaining:
            a=min(remaining)
            members=tuple(sorted({int(self.table[self.table[g,a],self._inverse[g]]) for g in range(self.order)}))
            classes.append(tuple(self.elements[i] for i in members));remaining.difference_update(members)
        return tuple(classes)

    def center(self):
        return tuple(self.elements[i] for i in range(self.order) if np.array_equal(self.table[i],self.table[:,i]))

    def _generated_indices(self,generators):
        gens=set(generators);gens.update(int(self._inverse[i]) for i in tuple(gens))
        known={self.index(self.identity)};pending=list(known)
        while pending:
            a=pending.pop()
            for b in sorted(gens):
                c=int(self.table[a,b])
                if c not in known:known.add(c);pending.append(c)
        return tuple(sorted(known))

    def _subset(self,elements):
        labels=tuple(elements)
        indices=tuple(sorted({self.index(g) for g in labels}))
        if len(indices)!=len(labels):raise ValueError('subgroup elements must be distinct')
        if self.index(self.identity) not in indices:raise ValueError('subgroup must contain the identity')
        pool=set(indices)
        if any(int(self.table[a,b]) not in pool for a in indices for b in indices):
            raise ValueError('subset is not closed')
        if isinstance(elements,AbstractFiniteGroup):
            if any(elements.multiply(self.elements[a],self.elements[b])!=self.elements[self.table[a,b]] for a in indices for b in indices):
                raise ValueError('subgroup operation differs from the parent')
        return indices

    def _subgroup(self,indices):
        indices=tuple(indices);lookup={g:i for i,g in enumerate(indices)}
        table=[[lookup[int(self.table[a,b])] for b in indices] for a in indices]
        return FiniteGroup(table,[self.elements[i] for i in indices],name=f'subgroup of {self.name}',max_order=self.order)

    def subgroup(self,elements):return self._subgroup(self._subset(elements))

    def generated_subgroup(self,generators):
        return self._subgroup(self._generated_indices([self.index(g) for g in generators]))

    def subgroups(self,*,max_subgroups=1024):
        """Enumerate by adjoining generators; raise rather than return a partial list."""
        limit=_integer(max_subgroups,'max_subgroups',1)
        first=(self.index(self.identity),);known={first};pending=[first]
        while pending:
            h=pending.pop()
            for g in range(self.order):
                if g in h:continue
                generated=self._generated_indices((*h,g))
                if generated not in known:
                    if len(known)>=limit:raise ValueError('subgroup enumeration exceeds max_subgroups')
                    known.add(generated);pending.append(generated)
        return tuple(self._subgroup(h) for h in sorted(known,key=lambda h:(len(h),h)))

    def is_normal(self,subgroup):
        h=self._subset(subgroup);pool=set(h)
        return all(int(self.table[self.table[g,a],self._inverse[g]]) in pool
                   for g in range(self.order) for a in h)

    def cosets(self,subgroup,*,side='left'):
        if side not in ('left','right'):raise ValueError('side must be left or right')
        h=self._subset(subgroup);remaining=set(range(self.order))-set(h);out=[h]
        while remaining:
            a=min(remaining)
            members=tuple(sorted(int(self.table[a,b] if side=='left' else self.table[b,a]) for b in h))
            out.append(members);remaining.difference_update(members)
        return tuple(tuple(self.elements[i] for i in members) for members in out)

    def quotient(self,subgroup):
        subgroup=self.subgroup(subgroup)
        if not self.is_normal(subgroup):raise ValueError('quotient requires a normal subgroup')
        cosets=self.cosets(subgroup);which={g:i for i,c in enumerate(cosets) for g in c}
        table=[[which[self.multiply(a[0],b[0])] for b in cosets] for a in cosets]
        return FiniteGroup(table,[frozenset(c) for c in cosets],name=f'{self.name}/H',max_order=self.order)

    def commutator_subgroup(self):
        generators=[self.multiply(self.multiply(a,b),self.multiply(self.inverse(a),self.inverse(b)))
                    for a in self.elements for b in self.elements]
        return self.generated_subgroup(generators)

    def regular_representation(self,*,exact=False,max_dimension=32):
        if self.order>_integer(max_dimension,'max_dimension',1):raise ValueError('regular representation exceeds max_dimension')
        from .representations import MatrixRepresentation
        matrices=np.zeros((self.order,self.order,self.order),dtype=int)
        for g in range(self.order):matrices[g,self.table[g],np.arange(self.order)]=1
        return MatrixRepresentation(self,matrices,exact=exact,name='regular')

    def latex(self,*,max_order=16):
        import sympy as sp
        if self.order>max_order:raise ValueError('Cayley display exceeds max_order')
        label=lambda g:sp.latex(sp.Symbol(str(g)))
        rows=[' & '.join(['\\cdot']+[label(g) for g in self.elements])]
        rows+=[' & '.join([label(g)]+[label(self.elements[i]) for i in self.table[a]]) for a,g in enumerate(self.elements)]
        return r'\begin{array}{c|'+('c'*self.order)+'}'+r'\\\hline '.join([rows[0],r'\\'.join(rows[1:])])+r'\end{array}'

    def _repr_latex_(self):
        return '$'+(self.latex() if self.order<=16 else r'|G|='+str(self.order))+'$'


def cyclic_group(n):
    n=_integer(n,'cyclic order',1)
    return FiniteGroup((np.arange(n)[:,None]+np.arange(n))%n,name=f'C{n}',max_order=n)


def dihedral_group(n):
    """D_n has order 2n; labels (k,b)=r**k s**b, s r s=r**-1."""
    n=_integer(n,'polygon order',2)
    elements=tuple((k,b) for b in (0,1) for k in range(n))
    table=[[(b+c)%2*n+(k+(-1)**b*l)%n for l,c in elements] for k,b in elements]
    return FiniteGroup(table,elements,name=f'D{n}',max_order=2*n)


def permutation_group(generators,*,max_order=256):
    """Generate exact permutations; a*b applies b first, then a."""
    limit=_integer(max_order,'max_order',1);gens=[tuple(g) for g in generators]
    if not gens or not gens[0]:raise ValueError('nonempty permutation generators required')
    degree=len(gens[0]);identity=tuple(range(degree))
    for g in gens:
        if (len(g)!=degree or any(not isinstance(i,(int,np.integer)) or isinstance(i,(bool,np.bool_)) for i in g)
                or tuple(sorted(g))!=identity):raise ValueError('generators must permute the same 0..degree-1 set')
    multiply=lambda a,b:tuple(a[b[i]] for i in range(degree))
    elements=[identity];lookup={identity:0};cursor=0
    while cursor<len(elements):
        a=elements[cursor];cursor+=1
        for b in gens:
            c=multiply(a,b)
            if c not in lookup:
                if len(elements)>=limit:raise ValueError('group exceeds max_order')
                lookup[c]=len(elements);elements.append(c)
    table=[[lookup[multiply(a,b)] for b in elements] for a in elements]
    return FiniteGroup(table,elements,name='permutation group',max_order=limit)
