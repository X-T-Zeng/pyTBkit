"""Exact angular quantum numbers, spherical harmonics and linear basis states."""
from dataclasses import dataclass
import numpy as np
import sympy as sp
from sympy.physics.wigner import clebsch_gordan
from .orbitals import angular_momentum,spin_rotation


def _half(value):
    value=sp.Rational(str(value))
    if (2*value).q!=1:raise ValueError('integer or half-integer quantum number required')
    return value


@dataclass(frozen=True)
class Qnum:
    l: int
    m: int
    s: object=0
    sz: object=0
    n: object=None

    def __post_init__(self):
        if not isinstance(self.l,int) or self.l<0 or not isinstance(self.m,int) or abs(self.m)>self.l:raise ValueError('l>=0, integer |m|<=l required')
        s,z=_half(self.s),_half(self.sz)
        if s<0 or abs(z)>s or (s-z).q!=1:raise ValueError('invalid spin projection')
        if self.n is not None and (not isinstance(self.n,int) or self.n<=self.l):raise ValueError('radial principal n must exceed l')
        object.__setattr__(self,'s',s);object.__setattr__(self,'sz',z)

    def latex(self):
        labels=[] if self.n is None else ['n='+sp.latex(self.n)]
        labels+=['l='+sp.latex(self.l),'m='+sp.latex(self.m),
                 's='+sp.latex(self.s),'s_z='+sp.latex(self.sz)]
        return r'\left|'+r',\,'.join(labels)+r'\right\rangle'

    def _repr_latex_(self):
        return '$'+self.latex()+'$'


def spherical_harmonic(l,m,theta=None,phi=None,*,real=False):
    """Normalized Condon-Shortley harmonics; real m>0 cosine, m<0 sine.

    For l=1: m=-1,0,1 are py,pz,px. For l=2: -2,-1,0,1,2
    are dxy,dyz,dz2,dzx,dx2-y2. No radial wavefunction is assumed.
    """
    Qnum(l,m)
    theta=sp.Symbol('theta',real=True) if theta is None else theta
    phi=sp.Symbol('phi',real=True) if phi is None else phi
    y=sp.Ynm(l,m,theta,phi)
    if real and m:
        if m>0:y=((-1)**m*y+sp.Ynm(l,-m,theta,phi))/sp.sqrt(2)
        else:y=sp.I*(y-(-1)**m*sp.Ynm(l,-m,theta,phi))/sp.sqrt(2)
    return sp.simplify(sp.expand_func(y).expand(complex=True)) if real else sp.expand_func(y)


class BasisFunc:
    """Symbolic linear combinations of orthonormal |n,l,m,s,sz> labels.

    Radial labels n are treated as orthogonal; this is not a radial integrator.
    Spin/orbital rotations act on the expansion coefficients.
    """
    def __init__(self,terms):
        if any(not isinstance(q,Qnum) for q in terms):raise ValueError('Qnum basis keys required')
        self.terms={q:sp.sympify(c) for q,c in terms.items() if c!=0}

    def __add__(self,other):
        if not isinstance(other,BasisFunc):return NotImplemented
        out=self.terms.copy()
        for q,c in other.terms.items():out[q]=sp.simplify(out.get(q,0)+c)
        return BasisFunc(out)

    def __mul__(self,value):return BasisFunc({q:c*sp.sympify(value) for q,c in self.terms.items()})
    def __rmul__(self,value):return self*value

    def inner(self,other):
        return sp.simplify(sum(sp.conjugate(c)*other.terms.get(q,0) for q,c in self.terms.items()))

    def subs(self,substitutions,*,simultaneous=True):
        """Substitute expansion coefficients, preserving the quantum labels."""
        return BasisFunc({q:c.subs(substitutions,simultaneous=simultaneous) for q,c in self.terms.items()})

    def latex(self):
        return ' + '.join(r'\left('+sp.latex(c)+r'\right)\,'+q.latex()
                          for q,c in self.terms.items()) or '0'

    def _repr_latex_(self):
        return '$\\displaystyle '+self.latex()+'$'

    def rotate(self,axis,angle):
        """Numerical active orbital+spin rotation, returns symbolic coefficients."""
        from scipy.linalg import expm
        axis=np.asarray(axis,float)
        if axis.shape!=(3,) or np.linalg.norm(axis)==0:raise ValueError('nonzero rotation axis required')
        axis=axis/np.linalg.norm(axis);out={}
        groups=set((q.n,q.l,q.s) for q in self.terms)
        for n,l,s in groups:
            orbital=expm(-1j*angle*sum(a*j for a,j in zip(axis,angular_momentum(l))))
            spin=spin_rotation(axis,angle,spin=float(s))
            ml=list(range(l,-l-1,-1));ms=[s-i for i in range(int(2*s)+1)]
            for q,c in self.terms.items():
                if (q.n,q.l,q.s)!=(n,l,s):continue
                for i,m in enumerate(ml):
                    for j,z in enumerate(ms):
                        factor=orbital[i,ml.index(q.m)]*spin[j,ms.index(q.sz)]
                        if abs(factor)<1e-15:continue
                        label=Qnum(l,m,s,z,n);out[label]=out.get(label,0)+c*factor
        return BasisFunc(out)


def coupled_basis(l,s=.5):
    """Exact Clebsch-Gordan matrix (coupled rows, uncoupled columns).

    Coupled labels are (j,mj), descending j and mj. Columns use m=l..-l
    outer and sz=s..-s inner. Matrix rows contain <l,m;s,sz|j,mj>.
    """
    s=_half(s)
    if not isinstance(l,int) or l<0 or s<0:raise ValueError('nonnegative l and spin required')
    uncoupled=[(m,s-z) for m in range(l,-l-1,-1) for z in range(int(2*s)+1)]
    coupled=[];j=sp.Rational(l)+s
    while j>=abs(sp.Rational(l)-s):
        coupled.extend((j,j-z) for z in range(int(2*j)+1));j-=1
    matrix=sp.Matrix([[clebsch_gordan(l,s,j,m,z,mj) for m,z in uncoupled] for j,mj in coupled])
    return matrix,tuple(coupled),tuple(uncoupled)
