"""Polynomial HK and trigonometric Htrig, compiled through shared kernels."""
import numpy as np
import sympy as sp
from .base import TBkit, points
from .hr import immutable
from . import backends


class HK(TBkit):
    def __init__(self, terms):
        if not terms: raise ValueError("at least one polynomial term required")
        powers=np.asarray(list(terms))
        coeff=np.asarray(list(terms.values()),complex)
        if powers.ndim!=2 or np.any(powers<0) or not np.all(powers==np.rint(powers)):
            raise ValueError("powers must be nonnegative integer tuples")
        if coeff.ndim!=3 or coeff.shape[1]!=coeff.shape[2] or coeff.shape[1]==0:
            raise ValueError("coefficients must be square matrices of equal size")
        self.dim,self.norb=powers.shape[1],coeff.shape[1]
        self.powers=immutable(powers,np.int64)
        self.coefficients=immutable(coeff,np.complex128)

    def hamiltonian(self,k,*,backend="numba"):
        k,single=points(k,self.dim)
        if backend=="numba": out=backends.polynomial(k,self.powers,self.coefficients)
        elif backend=="numpy":
            factors=np.prod(k[:,None,:]**self.powers[None,:,:],axis=-1)
            out=np.einsum('kt,tij->kij',factors,self.coefficients,optimize=True)
        else: raise ValueError("backend must be numba or numpy")
        return out[0] if single else out

    def derivative(self,axis):
        if not 0<=axis<self.dim: raise ValueError("invalid derivative axis")
        terms={}
        for p,c in zip(self.powers,self.coefficients):
            if p[axis]:
                q=p.copy(); q[axis]-=1
                terms[tuple(q)]=c*p[axis]
        if not terms: terms={(0,)*self.dim:np.zeros((self.norb,self.norb))}
        return HK(terms)

    def gradient(self,k,*,backend="numba"):
        if backend == "numba":
            k,single=points(k,self.dim)
            out=backends.polynomial_gradient(k,self.powers,self.coefficients)
            return out[0] if single else out
        if backend != "numpy": raise ValueError("backend must be numba or numpy")
        return np.stack([self.derivative(a).hamiltonian(k,backend=backend) for a in range(self.dim)],axis=-3)

    @classmethod
    def from_sympy(cls,matrix,symbols,parameters=None):
        """Extract polynomial coefficients after explicit parameter substitution."""
        matrix=sp.Matrix(matrix).subs(parameters or {})
        if matrix.rows!=matrix.cols: raise ValueError("square symbolic matrix required")
        symbols=tuple(symbols)
        if matrix.free_symbols-set(symbols): raise ValueError("unbound symbolic parameters")
        terms={}
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                for p,c in sp.Poly(sp.expand(matrix[i,j]),*symbols).terms():
                    if p not in terms: terms[p]=np.zeros(matrix.shape,complex)
                    terms[p][i,j]=complex(c)
        return cls(terms)

    def to_sympy(self,symbols=None):
        symbols=sp.symbols(f'k0:{self.dim}',real=True) if symbols is None else symbols
        if len(symbols)!=self.dim: raise ValueError("wrong symbol count")
        out=sp.zeros(self.norb)
        for p,c in zip(self.powers,self.coefficients):
            out+=sp.Matrix(c)*sp.prod(x**int(n) for x,n in zip(symbols,p))
        return out

    def to_symbolic(self,symbols=None):
        """Editable symbolic view; numerical coefficients remain floats."""
        from .symbolic_model import SymbolicHamiltonian
        symbols=sp.symbols(f'k0:{self.dim}',real=True) if symbols is None else tuple(symbols)
        return SymbolicHamiltonian(self.to_sympy(symbols),symbols)

    def kp2tb(self,spacing=1.0):
        """Central-difference lattice regularization through quadratic order.

        k_a -> sin(a*k_a)/a, k_a^2 -> 2*(1-cos(a*k_a))/a^2.
        Mixed quadratic terms use products of central differences.
        """
        from .hr import HR
        if spacing<=0: raise ValueError("positive spacing required")
        if np.any(self.powers.sum(axis=1)>2): raise ValueError("kp2tb supports total degree <= 2")
        zero=(0,)*self.dim
        terms={}
        for p,c in zip(self.powers,self.coefficients):
            expansion={zero:1.0+0j}
            for a,degree in enumerate(p):
                axis=np.zeros(self.dim,int); axis[a]=1
                plus,minus=tuple(axis),tuple(-axis)
                if degree==0: factors={zero:1.0}
                elif degree==1: factors={plus:1/(2j*spacing),minus:-1/(2j*spacing)}
                else: factors={zero:2/spacing**2,plus:-1/spacing**2,minus:-1/spacing**2}
                new={}
                for r,v in expansion.items():
                    for s,w in factors.items():
                        key=tuple(np.asarray(r)+s); new[key]=new.get(key,0)+v*w
                expansion=new
            for r,v in expansion.items(): terms[r]=terms.get(r,0)+v*c
        return HR(list(terms),list(terms.values()),lattice=np.eye(self.dim)*spacing)


class Htrig(TBkit):
    def __init__(self,frequencies,cosine,sine=None,*,lattice=None,orbitals=None):
        f=np.asarray(frequencies,float); c=np.asarray(cosine,complex)
        s=np.zeros_like(c) if sine is None else np.asarray(sine,complex)
        if f.ndim!=2 or c.ndim!=3 or len(f)!=len(c) or c.shape[1]!=c.shape[2] or s.shape!=c.shape:
            raise ValueError("frequencies (nt,dim), cosine/sine (nt,norb,norb) required")
        self.dim,self.norb=f.shape[1],c.shape[1]
        lattice=np.eye(self.dim) if lattice is None else np.asarray(lattice,float)
        if lattice.shape!=(self.dim,self.dim) or abs(np.linalg.det(lattice))<1e-14:raise ValueError('invertible lattice required')
        orbitals=np.zeros((self.norb,self.dim)) if orbitals is None else orbitals
        if np.shape(orbitals)!=(self.norb,self.dim):raise ValueError('orbital center shape mismatch')
        self.lattice=immutable(lattice,float);self.orbitals=immutable(orbitals,float)
        self.frequencies=immutable(f,float)
        self.cosine=immutable(c,np.complex128); self.sine=immutable(s,np.complex128)

    def hamiltonian(self,k,*,backend="numba"):
        k,single=points(k,self.dim)
        if backend=="numba": out=backends.trig(k,self.frequencies,self.cosine,self.sine)
        elif backend=="numpy":
            phase=k@self.frequencies.T
            out=np.einsum('kt,tij->kij',np.cos(phase),self.cosine)+np.einsum('kt,tij->kij',np.sin(phase),self.sine)
        else: raise ValueError("backend must be numba or numpy")
        return out[0] if single else out

    def gradient(self,k,*,backend="numba"):
        if backend == "numba":
            k,single=points(k,self.dim)
            out=backends.trig_gradient(k,self.frequencies,self.cosine,self.sine)
            return out[0] if single else out
        if backend != "numpy": raise ValueError("backend must be numba or numpy")
        return np.stack([Htrig(self.frequencies,self.sine*self.frequencies[:,a,None,None],-self.cosine*self.frequencies[:,a,None,None]).hamiltonian(k,backend=backend) for a in range(self.dim)],axis=-3)

    def to_hr(self,lattice=None):
        from .hr import HR
        lattice=self.lattice if lattice is None else np.asarray(lattice,float)
        r=self.frequencies@np.linalg.inv(lattice)
        if not np.allclose(r,np.rint(r),atol=1e-10,rtol=0): raise ValueError("frequencies not commensurate with lattice")
        return HR(np.rint(np.concatenate([r,-r])).astype(int),np.concatenate([(self.cosine-1j*self.sine)/2,(self.cosine+1j*self.sine)/2]),lattice=lattice,orbitals=self.orbitals)

    def to_symbolic(self,symbols=None):
        """Symbolic sin/cos view; existing floats are not made artificially exact.

        The returned matrix view does not carry lattice/orbital metadata.
        """
        from .symbolic_model import SymbolicHamiltonian
        symbols=sp.symbols(f'k0:{self.dim}',real=True) if symbols is None else tuple(symbols)
        if len(symbols)!=self.dim:raise ValueError('wrong symbol count')
        out=sp.zeros(self.norb)
        for f,c,s in zip(self.frequencies,self.cosine,self.sine):
            phase=sum(sp.Float(v)*k for v,k in zip(f,symbols))
            out+=sp.Matrix(c)*sp.cos(phase)+sp.Matrix(s)*sp.sin(phase)
        return SymbolicHamiltonian(out,symbols)
