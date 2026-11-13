"""Band geometric response kernels and streaming BZ integration.

Derivatives are dH/dk (not velocity divided by hbar). Return mathematical
kernels; material-specific SI prefactors are not silently applied.
"""
import itertools
import numpy as np
from scipy.special import expit
from .base import points,hermitian

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


def _spectral(model,k,backend,gap_tol,degeneracy):
    if gap_tol<=0 or degeneracy not in ('raise','omit'): raise ValueError("positive gap_tol and degeneracy='raise'/'omit' required")
    k,single=points(k,model.dim)
    e,u=model.bands(k,eigenvectors=True,backend=backend)
    dh=model.gradient(k,backend=backend)
    hermitian(dh)
    v=u.conj().swapaxes(-1,-2)[:,None]@dh@u[:,None]
    delta=e[:,:,None]-e[:,None,:];eye=np.eye(model.norb,dtype=bool)
    valid=(abs(delta)>gap_tol)&~eye
    if degeneracy=='raise' and np.any(~valid&~eye): raise ValueError("degenerate bands in band response kernel")
    inv=np.zeros_like(delta); np.divide(1.,delta,out=inv,where=valid)
    return e,v,inv,single


def geometric_tensors(model,k,*,backend='numba',gap_tol=1e-8,degeneracy='raise'):
    """Return energies, band gradients, Berry tensor, quantum metric and BCP.

    Layout: energy (...,n), velocity (...,dim,n), all tensors (...,dim,dim,n).
    BCP=2 Re sum(v_a[n,m]v_b[m,n])/(E_n-E_m)^3.
    omit reproduces MATLAB's gap cutoff; raise is safer near degeneracies.
    """
    if backend=='numba':
        from . import backends
        if gap_tol<=0 or degeneracy not in ('raise','omit'): raise ValueError("invalid gap_tol/degeneracy policy")
        k,single=points(k,model.dim)
        e,u=model.bands(k,eigenvectors=True,backend=backend)
        dh=model.gradient(k,backend=backend)
        hermitian(dh)
        velocity,berry,metric,bcp,degenerate=backends.spectral_geometry(e,u,dh,gap_tol)
        if degeneracy=='raise' and np.any(degenerate): raise ValueError("degenerate bands in band response kernel")
        result={'energy':e,'velocity':velocity,'berry':berry,'metric':metric,'bcp':bcp}
        return {key:value[0] for key,value in result.items()} if single else result
    e,v,inv,single=_spectral(model,k,backend,gap_tol,degeneracy)
    shape=(len(e),model.dim,model.dim,model.norb)
    berry=np.zeros(shape);metric=np.zeros(shape);bcp=np.zeros(shape)
    for a,b in itertools.product(range(model.dim),repeat=2):
        product=v[:,a]*v[:,b].swapaxes(-1,-2)
        berry[:,a,b]=(-2*product.imag*inv**2).sum(axis=-1)
        metric[:,a,b]=(product.real*inv**2).sum(axis=-1)
        bcp[:,a,b]=(2*product.real*inv**3).sum(axis=-1)
    result={'energy':e,'velocity':v.diagonal(axis1=-2,axis2=-1).real,'berry':berry,'metric':metric,'bcp':bcp}
    return {key:value[0] for key,value in result.items()} if single else result


def response_kernel(model,k,mu,*,kind='berry',indices=(0,1),temperature=50.,coefficients=(1.,1.,-1.),backend='numba',gap_tol=1e-8,degeneracy='raise'):
    """Occupation-weighted Berry, SOAHC or NCTE kernels per k and mu.

    SOAHC=(G_bc v_a-G_ac v_b) f'(E-mu).
    NCTE=(c1 v_a Omega_bc+c2 v_b Omega_ca+c3 v_c Omega_ab)
    * (E-mu) f'(E-mu), identical to repository single-index formulas.
    """
    from .topology import fermi
    k,single=points(k,model.dim);mu=np.atleast_1d(np.asarray(mu,float))
    if mu.ndim!=1 or not np.isfinite(mu).all(): raise ValueError("finite scalar or 1D mu required")
    count=2 if kind=='berry' else 3
    if kind not in ('berry','soahc','ncte') or len(indices)!=count or any(not isinstance(x,(int,np.integer)) or not 0<=x<model.dim for x in indices):
        raise ValueError("invalid response kind/indices")
    if len(coefficients)!=3: raise ValueError("three NCTE coefficients required")
    data=geometric_tensors(model,k,backend=backend,gap_tol=gap_tol,degeneracy=degeneracy)
    e=data['energy'];vel=data['velocity'];omega=data['berry'];g=data['bcp']
    if kind=='berry':
        a,b=indices;band=omega[:,a,b]
        weight=fermi(e[:,:,None],mu[None,None,:],temperature)
    else:
        a,b,c=indices
        if kind=='soahc': band=g[:,b,c]*vel[:,a]-g[:,a,c]*vel[:,b]
        else:
            c1,c2,c3=coefficients
            band=c1*vel[:,a]*omega[:,b,c]+c2*vel[:,b]*omega[:,c,a]+c3*vel[:,c]*omega[:,a,b]
        weight=fermi_derivative(e[:,:,None],mu[None,None,:],temperature)
        if kind=='ncte': weight=weight*(e[:,:,None]-mu[None,None,:])
    out=np.einsum('kn,knm->km',band,weight)
    return out[0] if single else out


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
