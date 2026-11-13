"""Explicit numerical model algebra, keeping each model's k convention."""
import numbers
import numpy as np
from .base import TBkit


class LinearCombination(TBkit):
    def __init__(self,models,weights):
        self.models=tuple(models);self.weights=tuple(weights)
        self.dim,self.norb=models[0].dim,models[0].norb
        if any(m.dim!=self.dim or m.norb!=self.norb for m in models):raise ValueError("model dimensions differ")
        lattices=[m.lattice for m in models if hasattr(m,'lattice')]
        if lattices:
            if any(not np.allclose(x,lattices[0]) for x in lattices):raise ValueError("model lattices differ")
            self.lattice=lattices[0]
        centers=[m.orbitals for m in models if hasattr(m,'orbitals')]
        if centers and any(not np.allclose(x,centers[0]) for x in centers):raise ValueError("orbital gauges/centers differ")
        if centers:self.orbitals=centers[0]

    def hamiltonian(self,k,*,backend='numba'):
        return sum(w*m.hamiltonian(k,backend=backend) for m,w in zip(self.models,self.weights))

    def gradient(self,k,*,backend='numba'):
        return sum(w*m.gradient(k,backend=backend) for m,w in zip(self.models,self.weights))


def combine(a,b,sign=1):
    if not isinstance(b,TBkit):return NotImplemented
    from .hr import HR
    from .symbolic import HK
    check=LinearCombination([a,b],[1,sign])
    if isinstance(a,HR) and isinstance(b,HR):
        terms={tuple(r):h.copy() for r,h in zip(a.vectors,a.hopping)}
        for r,h in zip(b.vectors,b.hopping):terms[tuple(r)]=terms.get(tuple(r),0)+sign*h
        return HR(list(terms),list(terms.values()),lattice=a.lattice,orbitals=a.orbitals)
    if isinstance(a,HK) and isinstance(b,HK):
        terms={tuple(p):c.copy() for p,c in zip(a.powers,a.coefficients)}
        for p,c in zip(b.powers,b.coefficients):terms[tuple(p)]=terms.get(tuple(p),0)+sign*c
        return HK(terms)
    return check


def scale(model,value):
    if not isinstance(value,numbers.Number) or not np.isfinite(value):return NotImplemented
    from .hr import HR
    from .symbolic import HK,Htrig
    if isinstance(model,HR):return HR(model.vectors,model.hopping*value,lattice=model.lattice,orbitals=model.orbitals)
    if isinstance(model,HK):return HK({tuple(p):value*c for p,c in zip(model.powers,model.coefficients)})
    if isinstance(model,Htrig):return Htrig(model.frequencies,model.cosine*value,model.sine*value,lattice=model.lattice,orbitals=model.orbitals)
    return LinearCombination([model],[value])
