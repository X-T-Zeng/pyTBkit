"""Constant unitary/antiunitary symmetry actions with explicit k conventions."""
import numpy as np
from .base import TBkit
from .hr import immutable


class Oper:
    """T[H](k) = sign U H(R k)^(*) U^dagger.

    R is the *input momentum action*, sign=-1 for antisymmetry and * for
    antiunitary. Use from_spatial for a physical spatial action Q: then
    R=Q^-1 (unitary) or -Q^-1 (antiunitary), as in MATLAB HK.applyRU.
    Numeric arrays and exact, constant SymPy matrices are accepted.
    """
    def __init__(self,unitary,rotation,*,antiunitary=False,antisymmetry=False):
        import sympy as sp
        self._symbolic_unitary=sp.ImmutableMatrix(unitary) if isinstance(unitary,sp.MatrixBase) else None
        self._symbolic_rotation=sp.ImmutableMatrix(rotation) if isinstance(rotation,sp.MatrixBase) else None
        try:
            u=np.asarray(unitary,complex);r=np.asarray(rotation,float)
        except (TypeError,ValueError) as exc:raise ValueError('constant numerical or exact algebraic matrices required') from exc
        if u.ndim!=2 or u.shape[0]<1 or u.shape[0]!=u.shape[1] or not np.isfinite(u).all() or not np.allclose(u.conj().T@u,np.eye(len(u)),atol=1e-10,rtol=0):
            raise ValueError('unitary matrix required')
        if r.ndim!=2 or r.shape[0]<1 or r.shape[0]!=r.shape[1] or not np.isfinite(r).all() or not np.allclose(r.T@r,np.eye(len(r)),atol=1e-10,rtol=0):
            raise ValueError('orthogonal momentum rotation required')
        self.unitary=immutable(u,complex);self.rotation=immutable(r,float)
        self.antiunitary=bool(antiunitary);self.antisymmetry=bool(antisymmetry)

    @classmethod
    def from_spatial(cls,unitary,spatial_rotation,*,antiunitary=False,antisymmetry=False):
        import sympy as sp
        inverse=spatial_rotation.inv() if isinstance(spatial_rotation,sp.MatrixBase) else np.linalg.inv(np.asarray(spatial_rotation,float))
        return cls(unitary,(-1 if antiunitary else 1)*inverse,antiunitary=antiunitary,antisymmetry=antisymmetry)

    def exact_matrices(self,tolerance=1e-12):
        """Recover common algebraic entries without treating roundoff as exact.

        Recovered matrices must be *exactly* unitary/orthogonal. If arbitrary
        numeric input cannot be recovered, use exact SymPy input or numeric
        invariant generation. Explicit SymPy input is never thresholded.
        """
        import sympy as sp
        if tolerance<=0:raise ValueError('positive symbolic recovery tolerance required')
        def scalar(v):
            if abs(v)<=tolerance:return sp.S.Zero
            return sp.nsimplify(float(v),tolerance=tolerance,full=False)
        def recover(a):
            return sp.ImmutableMatrix(a.shape[0],a.shape[1],lambda i,j:scalar(a[i,j].real)+sp.I*scalar(a[i,j].imag))
        u=self._symbolic_unitary if self._symbolic_unitary is not None else recover(self.unitary)
        r=self._symbolic_rotation if self._symbolic_rotation is not None else recover(self.rotation)
        u=sp.ImmutableMatrix(u.applyfunc(sp.simplify));r=sp.ImmutableMatrix(r.applyfunc(sp.simplify))
        if (u.conjugate().T*u-sp.eye(len(self.unitary))).applyfunc(sp.simplify)!=sp.zeros(len(self.unitary)) or (r.T*r-sp.eye(len(self.rotation))).applyfunc(sp.simplify)!=sp.zeros(len(self.rotation)):
            raise ValueError("exact symmetry recovery failed; pass exact SymPy matrices or use method='numeric'")
        if not np.allclose(np.array(u).astype(complex),self.unitary,atol=10*tolerance,rtol=0) or not np.allclose(np.array(r).astype(float),self.rotation,atol=10*tolerance,rtol=0):
            raise ValueError('exact symmetry recovery exceeded tolerance')
        return u,r

    def apply(self,model):
        if model.norb!=len(self.unitary) or model.dim!=len(self.rotation):raise ValueError('operator/model dimensions differ')
        return _Transformed(model,self)

    def residual(self,model,k,*,backend='numba'):
        return float(np.max(np.abs(model.hamiltonian(k,backend=backend)-self.apply(model).hamiltonian(k,backend=backend))))


class _Transformed(TBkit):
    def __init__(self,model,oper):
        self.model,self.oper=model,oper;self.norb,self.dim=model.norb,model.dim
        if hasattr(model,'lattice'):self.lattice=immutable(model.lattice@oper.rotation,float)

    def hamiltonian(self,k,*,backend='numba'):
        h=self.model.hamiltonian(np.asarray(k)@self.oper.rotation.T,backend=backend)
        if self.oper.antiunitary:h=h.conj()
        return (-1 if self.oper.antisymmetry else 1)*self.oper.unitary@h@self.oper.unitary.conj().T

    def gradient(self,k,*,backend='numba'):
        d=self.model.gradient(np.asarray(k)@self.oper.rotation.T,backend=backend)
        if self.oper.antiunitary:d=d.conj()
        d=(-1 if self.oper.antisymmetry else 1)*self.oper.unitary@d@self.oper.unitary.conj().T
        return np.einsum('...aij,ab->...bij',d,self.oper.rotation)
