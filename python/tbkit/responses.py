"""Band geometric response kernels and streaming BZ integration.

Derivatives are dH/dk (not velocity divided by hbar). Return mathematical
kernels; material-specific SI prefactors are not silently applied.
"""
from dataclasses import dataclass
import itertools
import numpy as np
from scipy.special import expit
from .spectral import SpectralBatch,prepare_spectral
from .hr import immutable

from .constants import KB_EV_PER_K as KB


def fermi_derivative(energy,mu=0.,temperature=50.,order=1):
    if np.ndim(temperature)!=0 or not np.isfinite(temperature) or temperature<=0 or order not in (0,1,2): raise ValueError("finite scalar T>0 and derivative order 0,1,2 required")
    beta=1/(KB*temperature);x=(np.asarray(energy)-mu)*beta
    if order==0:return expit(-x)
    # f*(1-f) loses the negative-energy tail when f rounds to one.
    # This equivalent expression preserves symmetry and avoids overflow.
    z=np.exp(-abs(x));variance=z/(1+z)**2
    if order==1:return -beta*variance
    return beta**2*variance*np.tanh(x/2)


@dataclass(frozen=True)
class GeometryBatch:
    """Reusable, clean-limit real band geometry with an explicit gap policy.

    Arrays always retain their batch axis. Degenerate single-band quantities
    in omit mode are basis dependent; only justified subspace sums are physical.
    """
    energy: np.ndarray
    velocity: np.ndarray
    berry: np.ndarray
    metric: np.ndarray
    bcp: np.ndarray
    degenerate: np.ndarray
    single: bool=False
    frame: str='model'
    gap_tol: float=1e-8
    degeneracy: str='raise'

    def __post_init__(self):
        if not np.isfinite(self.gap_tol) or self.gap_tol<=0 or self.degeneracy not in ('raise','omit'):
            raise ValueError('positive finite gap_tol and explicit raise/omit policy required')
        values={}
        for name in ('energy','velocity','berry','metric','bcp'):
            a=getattr(self,name)
            if np.iscomplexobj(a):raise ValueError('clean band geometry must be real; finite complex pole response is a different prescription')
            values[name]=immutable(a,float)
        e,v=values['energy'],values['velocity']
        if e.ndim!=2 or not len(e) or e.shape[1]<1 or v.ndim!=3 or v.shape[0]!=len(e) or v.shape[2]!=e.shape[1] or v.shape[1]<1:
            raise ValueError('invalid batched energy/velocity shape')
        expected=(len(e),v.shape[1],v.shape[1],e.shape[1])
        if any(values[name].shape!=expected for name in ('berry','metric','bcp')):
            raise ValueError('geometry tensor shape mismatch')
        d=immutable(self.degenerate,bool)
        if d.shape!=(len(e),) or (self.single and len(e)!=1):raise ValueError('invalid batch metadata')
        if self.frame not in ('model','point-orbital'):raise ValueError('unknown geometry frame')
        if self.degeneracy=='raise' and np.any(d):raise ValueError('band degeneracy; use occupied-subspace geometry or a justified explicit omit policy')
        for name,value in values.items():object.__setattr__(self,name,value)
        object.__setattr__(self,'degenerate',d)

    @property
    def dim(self):return self.velocity.shape[1]

    @property
    def nbytes(self):
        return sum(getattr(self,name).nbytes for name in ('energy','velocity','berry','metric','bcp','degenerate'))

    def as_dict(self,*,copy=False):
        result={name:(getattr(self,name)[0] if self.single else getattr(self,name)) for name in ('energy','velocity','berry','metric','bcp')}
        return {name:value.copy() for name,value in result.items()} if copy else result


def prepare_geometry(model_or_spectral,k=None,*,backend='numba',gap_tol=1e-8,degeneracy='raise',frame='model'):
    """Compute clean band geometry once, then reuse it for tensors/mu/T scans.

    A SpectralBatch needs no k argument. Its frame is already fixed and cannot
    be changed retroactively. Finite-eta/lifetime models are not inferred.
    """
    if not np.isfinite(gap_tol) or gap_tol<=0 or degeneracy not in ('raise','omit'):
        raise ValueError('finite positive gap_tol and degeneracy=raise/omit required')
    if isinstance(model_or_spectral,SpectralBatch):
        if k is not None or frame!='model':raise ValueError('a prepared snapshot already fixes k points and frame')
        batch=model_or_spectral
    else:
        if k is None:raise ValueError('k points required for a model')
        batch=prepare_spectral(model_or_spectral,k,backend=backend,frame=frame)
    e=batch.energy;v=batch.velocity_matrices
    if backend=='numba':
        from .backends import geometry_from_velocities
        velocity,berry,metric,bcp,degenerate=geometry_from_velocities(e,v,gap_tol)
    else:
        if backend not in ('numpy','fortran'):raise ValueError('backend must be numba, numpy or fortran')
        delta=e[:,:,None]-e[:,None,:];eye=np.eye(e.shape[1],dtype=bool)
        valid=(abs(delta)>gap_tol)&~eye
        degenerate=np.any((~valid)&~eye,axis=(-2,-1))
        inv=np.zeros_like(delta);np.divide(1.,delta,out=inv,where=valid)
        shape=(len(e),batch.dim,batch.dim,e.shape[1])
        berry=np.zeros(shape);metric=np.zeros(shape);bcp=np.zeros(shape)
        for a,b in itertools.product(range(batch.dim),repeat=2):
            product=v[:,a]*v[:,b].swapaxes(-1,-2)
            berry[:,a,b]=(-2*product.imag*inv**2).sum(axis=-1)
            metric[:,a,b]=(product.real*inv**2).sum(axis=-1)
            bcp[:,a,b]=(2*product.real*inv**3).sum(axis=-1)
        velocity=v.diagonal(axis1=-2,axis2=-1).real
    return GeometryBatch(e,velocity,berry,metric,bcp,degenerate,single=batch.single,frame=batch.frame,gap_tol=gap_tol,degeneracy=degeneracy)


def geometric_tensors(model,k,*,backend='numba',gap_tol=1e-8,degeneracy='raise',frame='model'):
    """Dictionary compatibility API; use prepare_geometry for explicit reuse."""
    return prepare_geometry(model,k,backend=backend,gap_tol=gap_tol,degeneracy=degeneracy,frame=frame).as_dict(copy=True)


def response_from_geometry(geometry,mu,*,kind='berry',indices=(0,1),temperature=50.,coefficients=(1.,1.,-1.)):
    """Apply occupations to caller-owned GeometryBatch without diagonalization.

    Returns (nk,nmu), or (nmu,) for a single-point preparation. Derivatives
    retain dH/dk units; material/SI prefactors are never applied implicitly.
    """
    from .topology import fermi
    if not isinstance(geometry,GeometryBatch):raise TypeError('validated GeometryBatch required')
    if np.iscomplexobj(mu):raise ValueError('chemical potential must be real')
    mu=np.atleast_1d(np.asarray(mu,float))
    if mu.ndim!=1 or not len(mu) or not np.isfinite(mu).all():raise ValueError('nonempty finite scalar or 1D mu required')
    count=2 if kind=='berry' else 3
    if kind not in ('berry','soahc','ncte') or len(indices)!=count or any(not isinstance(x,(int,np.integer)) or not 0<=x<geometry.dim for x in indices):
        raise ValueError('invalid response kind/indices')
    if np.iscomplexobj(coefficients) or np.shape(coefficients)!=(3,) or not np.isfinite(coefficients).all():raise ValueError('three finite real NCTE coefficients required')
    e=geometry.energy;vel=geometry.velocity;omega=geometry.berry;g=geometry.bcp
    if kind=='berry':
        a,b=indices;band=omega[:,a,b]
        weight=fermi(e[:,:,None],mu[None,None,:],temperature)
    else:
        a,b,c=indices
        if kind=='soahc':band=g[:,b,c]*vel[:,a]-g[:,a,c]*vel[:,b]
        else:
            c1,c2,c3=coefficients
            band=c1*vel[:,a]*omega[:,b,c]+c2*vel[:,b]*omega[:,c,a]+c3*vel[:,c]*omega[:,a,b]
        weight=fermi_derivative(e[:,:,None],mu[None,None,:],temperature)
        if kind=='ncte':weight=weight*(e[:,:,None]-mu[None,None,:])
    out=np.einsum('kn,knm->km',band,weight)
    return out[0] if geometry.single else out


def response_kernel(model,k,mu,*,kind='berry',indices=(0,1),temperature=50.,coefficients=(1.,1.,-1.),backend='numba',gap_tol=1e-8,degeneracy='raise',frame='model'):
    """One-shot compatibility API for clean-limit Berry/SOAHC/NCTE kernels.

    For parameter/temperature/tensor sweeps prepare_geometry once and call
    response_from_geometry repeatedly. This API does not implement the official
    MATLAB finite-eta complex-pole prescription.
    """
    geometry=prepare_geometry(model,k,backend=backend,gap_tol=gap_tol,degeneracy=degeneracy,frame=frame)
    return response_from_geometry(geometry,mu,kind=kind,indices=indices,temperature=temperature,coefficients=coefficients)


def integrate_bz(function,shape,*,lattice=None,chunk_size=512,normalization='average'):
    """Streaming endpoint-free reciprocal mesh; function(batch_k)->(nk,...).

    average: BZ mean; integral: integral d^d k; states: integral d^d k/(2pi)^d.
    No full k mesh or response tensor is retained in memory.
    """
    shape=np.asarray(shape)
    if shape.ndim!=1 or not len(shape) or np.any(shape<1) or not np.all(shape==np.rint(shape)) or chunk_size<1:
        raise ValueError("positive integer grid sizes/chunk_size required")
    shape=shape.astype(int);dim=len(shape)
    lattice=np.eye(dim) if lattice is None else np.asarray(lattice,float)
    if lattice.shape!=(dim,dim) or abs(np.linalg.det(lattice))<1e-12: raise ValueError("invertible lattice required")
    if normalization not in ('average','integral','states'): raise ValueError("invalid normalization")
    reciprocal=2*np.pi*np.linalg.inv(lattice).T;n=int(np.prod(shape));total=None;compensation=None
    for start in range(0,n,chunk_size):
        fractional=np.array(np.unravel_index(np.arange(start,min(start+chunk_size,n)),tuple(shape))).T/shape
        values=np.asarray(function(fractional@reciprocal))
        if values.ndim<1 or len(values)!=len(fractional) or not np.isfinite(values).all(): raise ValueError("integrand must return finite (nk,...) values")
        part=values.sum(axis=0)
        if total is None: total=np.zeros_like(part,dtype=np.result_type(part,float));compensation=np.zeros_like(total)
        # Compensated sum of batch totals, deterministic for fixed chunk size.
        y=part-compensation;t=total+y;compensation=(t-total)-y;total=t
    factor={'average':1.,'integral':abs(np.linalg.det(reciprocal)),'states':1/abs(np.linalg.det(lattice))}[normalization]
    return total*(factor/n)
