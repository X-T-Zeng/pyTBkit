"""Finite symmetry closure and symmetry-constrained k.p bases."""
import itertools
import numpy as np
from .base import TBkit
from .operators import Oper


def _equivalent(a,b,tol):
    if a.antiunitary!=b.antiunitary or a.antisymmetry!=b.antisymmetry or not np.allclose(a.rotation,b.rotation,atol=tol): return False
    # Global phases have no effect on conjugation of Hamiltonians.
    flat=b.unitary.ravel(); index=np.argmax(abs(flat))
    phase=a.unitary.ravel()[index]/flat[index]
    return np.isclose(abs(phase),1,atol=tol) and np.allclose(a.unitary,phase*b.unitary,atol=tol)


def finite_group(generators,*,max_order=256,tol=1e-9):
    if not generators or max_order<1: raise ValueError("nonempty generators and positive max_order required")
    n=len(generators[0].unitary); dim=len(generators[0].rotation)
    if any(len(g.unitary)!=n or len(g.rotation)!=dim for g in generators): raise ValueError("generator dimensions differ")
    elements=[Oper(np.eye(n),np.eye(dim))]; cursor=0
    while cursor<len(elements):
        a=elements[cursor]; cursor+=1
        for b in generators:
            c=Oper(a.unitary@(b.unitary.conj() if a.antiunitary else b.unitary),b.rotation@a.rotation,antiunitary=a.antiunitary!=b.antiunitary,antisymmetry=a.antisymmetry!=b.antisymmetry)
            if not any(_equivalent(c,e,tol) for e in elements):
                if len(elements)>=max_order: raise ValueError("group does not close within max_order")
                elements.append(c)
    return elements


class _Symmetrized(TBkit):
    def __init__(self,model,group):
        self.norb,self.dim=model.norb,model.dim
        self.models=[g.apply(model) for g in group]
        if hasattr(model,'lattice'): self.lattice=model.lattice

    def hamiltonian(self,k,*,backend='numba'):
        return sum(m.hamiltonian(k,backend=backend) for m in self.models)/len(self.models)

    def gradient(self,k,*,backend='numba'):
        return sum(m.gradient(k,backend=backend) for m in self.models)/len(self.models)


def symmetrize(model,generators,*,max_order=256):
    return _Symmetrized(model,finite_group(generators,max_order=max_order))


def invariant_kp(norb,dim,order,generators,*,max_unknowns=300,method="symbolic",tolerance=1e-10):
    """Exact SymPy nullspace of Hermitian polynomial symmetry constraints.

    Returns independent HK models, each multiplied by a free real parameter.
    Supports constant basis representations and orthogonal momentum actions;
    nonsymmorphic k-dependent representation matrices are not covered.
    method='symbolic' uses exact polynomial nullspaces after checked algebraic
    recovery. method='numeric' uses coefficient-space real SVD with an explicit
    relative rank tolerance, avoiding symbolic expressions and sampled k fits.
    """
    import sympy as sp
    from .symbolic import HK
    if norb<1 or dim<1 or order<0 or any(int(x)!=x for x in (norb,dim,order)): raise ValueError("integer model sizes/order required")
    norb,dim,order=int(norb),int(dim),int(order)
    generators=tuple(generators)
    if method not in ('symbolic','numeric') or tolerance<=0:raise ValueError('invalid invariant generation method/tolerance')
    powers=[p for p in itertools.product(range(order+1),repeat=dim) if sum(p)<=order]
    if len(powers)*norb**2>max_unknowns: raise ValueError("symbolic problem exceeds max_unknowns")
    if any(g.unitary.shape!=(norb,norb) or g.rotation.shape!=(dim,dim) for g in generators): raise ValueError("symmetry/model dimensions differ")
    if method=='numeric':return _numeric_invariants(norb,dim,powers,generators,tolerance)
    basis=[]
    for i in range(norb):
        m=sp.zeros(norb); m[i,i]=1; basis.append(m)
    for i in range(norb):
        for j in range(i+1,norb):
            m=sp.zeros(norb);m[i,j]=m[j,i]=1;basis.append(m)
            m=sp.zeros(norb);m[i,j]=-sp.I;m[j,i]=sp.I;basis.append(m)
    x=sp.symbols(f'k0:{dim}',real=True); coeff=sp.symbols(f'c0:{len(powers)*len(basis)}',real=True)
    h=sp.zeros(norb); column=0
    for p in powers:
        monomial=sp.prod(k**degree for k,degree in zip(x,p))
        for b in basis:
            h+=coeff[column]*b*monomial; column+=1
    equations=[]
    for g in generators:
        u,r=g.exact_matrices()
        rotated=r*sp.Matrix(x)
        transformed=h.subs(dict(zip(x,rotated)),simultaneous=True)
        if g.antiunitary: transformed=transformed.conjugate()
        if g.antisymmetry: transformed=-transformed
        delta=h-u*transformed*u.conjugate().T
        for expr in delta:
            for part in (sp.re(expr).expand(complex=True),sp.im(expr).expand(complex=True)):
                equations.extend(sp.Poly(sp.expand(part),*x).coeffs())
    matrix=sp.linear_eq_to_matrix(equations,coeff)[0] if equations else sp.zeros(0,len(coeff))
    result=[]
    for vector in matrix.nullspace():
        terms={};column=0
        for p in powers:
            block=sp.zeros(norb)
            for b in basis:
                block+=vector[column]*b;column+=1
            if block!=sp.zeros(norb): terms[p]=np.array(block).astype(complex)
        if terms: result.append(HK(terms))
    return result


def _numeric_invariants(norb,dim,powers,generators,tolerance):
    from scipy.linalg import svd
    from .symbolic import HK
    # Orthonormal real coordinates of Hermitian matrices: Tr(B_i B_j)=delta_ij.
    matrices=[]
    for i in range(norb):
        b=np.zeros((norb,norb),complex);b[i,i]=1;matrices.append(b)
    for i in range(norb):
        for j in range(i+1,norb):
            b=np.zeros((norb,norb),complex);b[i,j]=b[j,i]=1/np.sqrt(2);matrices.append(b)
            b=np.zeros((norb,norb),complex);b[i,j]=-1j/np.sqrt(2);b[j,i]=1j/np.sqrt(2);matrices.append(b)
    matrices=np.array(matrices);nb=norb**2;size=len(powers)*nb
    lookup={p:i for i,p in enumerate(powers)};constraints=[]
    for g in generators:
        transformed=g.unitary@(matrices.conj() if g.antiunitary else matrices)@g.unitary.conj().T
        if g.antisymmetry:transformed=-transformed
        internal=np.einsum('aij,bji->ab',matrices,transformed).real
        action=np.zeros((size,size))
        for source,power in enumerate(powers):
            expansion={(0,)*dim:1.}
            for a,degree in enumerate(power):
                for _ in range(degree):
                    new={}
                    for p,value in expansion.items():
                        for b in range(dim):
                            q=list(p);q[b]+=1;q=tuple(q)
                            new[q]=new.get(q,0.)+value*g.rotation[a,b]
                    expansion=new
            for p,value in expansion.items():
                target=lookup[p]
                action[target*nb:(target+1)*nb,source*nb:(source+1)*nb]+=value*internal
        constraints.append(action-np.eye(size))
    if constraints:
        _,singular,right=svd(np.concatenate(constraints),full_matrices=False)
        # An identity action may leave only roundoff. A purely relative cutoff
        # would classify that roundoff as physical constraints and remove terms.
        cutoff=tolerance*max(1.,float(singular[0]))
        rank=int(np.count_nonzero(singular>cutoff))
        coefficients=right[rank:].T
    else:coefficients=np.eye(size)
    result=[]
    for vector in coefficients.T:
        blocks=np.einsum('pa,aij->pij',vector.reshape(len(powers),nb),matrices)
        result.append(HK(dict(zip(powers,blocks))))
    return result
