"""Gauge invariant occupied-subspace topology and Kubo Berry curvature."""
import numpy as np
from scipy.special import expit
from .constants import KB_EV_PER_K


def fermi(energy,mu=0.0,temperature=0.0):
    """Fermi occupation for energies in eV and temperature in Kelvin."""
    if np.ndim(temperature)!=0 or not np.isfinite(temperature) or temperature<0:raise ValueError("finite nonnegative scalar temperature required")
    x=np.asarray(energy)-mu
    if temperature==0: return (x<0).astype(float)+0.5*(x==0)
    return expit(-x/(KB_EV_PER_K*temperature))


def berry_curvature(model,k,axes=(0,1),*,gap_tol=1e-9,backend="numba"):
    """Band curvature -2 Im[v_a(n,m)v_b(m,n)]/(E_n-E_m)^2.

    Raises at degeneracies: single-band curvature is not defined there.
    Derivatives and output use the model's default k coordinates.
    """
    if gap_tol<=0:raise ValueError("positive gap_tol required")
    a,b=axes
    if a==b or min(axes)<0 or max(axes)>=model.dim: raise ValueError("invalid curvature axes")
    e,u=model.bands(k,eigenvectors=True,backend=backend)
    d=model.gradient(k,backend=backend)
    from .base import hermitian
    hermitian(d)
    delta=e[..., :,None]-e[...,None,:]
    eye=np.eye(model.norb,dtype=bool)
    if np.any((np.abs(delta)<gap_tol)&~eye): raise ValueError("band degeneracy; use occupied-subspace topology")
    denominator=np.where(eye,np.inf,delta**2)
    va=u.conj().swapaxes(-1,-2)@d[...,a,:,:]@u
    vb=u.conj().swapaxes(-1,-2)@d[...,b,:,:]@u
    return (-2*np.imag(va*vb.swapaxes(-1,-2))/denominator).sum(axis=-1)



def _periodic_bz(model,backend,tol=1e-9):
    """Check the periodic-gauge BZ assumption at deterministic generic points.

    This catches continuum models and missing lattice metadata. It is not a
    symbolic proof; nonperiodic orbital gauges require explicit boundary sewing.
    """
    lattice=getattr(model,'lattice',np.eye(model.dim))
    reciprocal=2*np.pi*np.linalg.inv(lattice).T
    fractional=np.array([[.137+.07*a for a in range(model.dim)],
                         [.319-.03*a for a in range(model.dim)]])
    k=fractional@reciprocal;h=model.hamiltonian(k,backend=backend)
    scale=max(1.,float(np.max(abs(h))))
    for g in reciprocal:
        if np.max(abs(model.hamiltonian(k+g,backend=backend)-h))>tol*scale:
            raise ValueError('periodic-gauge BZ required; check lattice metadata or supply boundary sewing for a nonperiodic gauge')
    return reciprocal

def _link(u,v,tol):
    overlap=u.conj().swapaxes(-1,-2)@v
    # Reject singular occupied-subspace links instead of returning arbitrary phases.
    if np.any(np.linalg.svd(overlap,compute_uv=False)[...,-1]<tol):
        raise ValueError("singular overlap; refine the k mesh")
    det=np.linalg.det(overlap)
    return det/np.abs(det)


def chern_number(model,shape=(31,31),*,occupied=1,gap_tol=1e-8,backend="numba"):
    """Fukui lattice Chern number on the full 2D periodic BZ.

    Sign agrees with berry_curvature (connection A=i<u|du>).
    """
    from .geometry import kmesh
    if model.dim!=2 or not 0<occupied<model.norb: raise ValueError("gapped 2D occupied subspace required")
    if len(shape)!=2 or min(shape)<3: raise ValueError("at least 3 points per axis")
    _periodic_bz(model,backend)
    k=kmesh(shape,lattice=getattr(model,"lattice",None))
    e,u=model.bands(k,eigenvectors=True,backend=backend)
    if np.min(e[:,occupied]-e[:,occupied-1])<=gap_tol: raise ValueError("occupied subspace not gapped")
    u=u[:,:,:occupied].reshape(*shape,model.norb,occupied)
    ux=_link(u,np.roll(u,-1,axis=0),gap_tol)
    uy=_link(u,np.roll(u,-1,axis=1),gap_tol)
    flux=-np.angle(ux*np.roll(uy,-1,axis=0)*np.conj(np.roll(ux,-1,axis=1))*np.conj(uy))
    return float(flux.sum()/(2*np.pi))


def wilson_loop(model,k,*,occupied=1,backend="numba",closed=True,boundary_sewing=None):
    """Eigenphases of product of unitary polar overlap links (radians).

    k omits a repeated endpoint. A closed loop joins last point to first in
    the periodic orbital gauge. Returns phases with A=i<u|du> sign. An open
    path has no gauge-invariant eigenphases; use wilson_transport instead.
    """
    if not closed:raise ValueError("open paths have no gauge-invariant Wilson eigenphases; use wilson_transport")
    if getattr(model,'requires_boundary_sewing',False) and boundary_sewing is None:
        raise ValueError('embedded orbital frame requires explicit boundary_sewing for loop closure')
    if boundary_sewing is not None:
        boundary_sewing=np.asarray(boundary_sewing,complex)
        if boundary_sewing.shape!=(model.norb,model.norb) or not np.allclose(boundary_sewing.conj().T@boundary_sewing,np.eye(model.norb),atol=1e-10):
            raise ValueError('boundary_sewing must be unitary in the orbital basis')
    if not 0<occupied<=model.norb: raise ValueError("invalid occupied count")
    e,u=model.bands(k,eigenvectors=True,backend=backend)
    if e.ndim!=2 or len(e)<2: raise ValueError("at least two k points required")
    if occupied<model.norb and np.min(e[:,occupied]-e[:,occupied-1])<1e-9:
        raise ValueError("occupied subspace not gapped")
    u=u[:,:,:occupied]
    product=np.eye(occupied,dtype=complex)
    for i in range(len(u) if closed else len(u)-1):
        target=u[(i+1)%len(u)]
        if i==len(u)-1 and boundary_sewing is not None:target=boundary_sewing@target
        link=u[i].conj().T@target
        left,s,right=np.linalg.svd(link)
        if s.min()<1e-10: raise ValueError("singular overlap; refine loop")
        product=product@(left@right)
    return np.sort(-np.angle(np.linalg.eigvals(product)))


def _wilson_matrix(u,tol=1e-10):
    product=np.eye(u.shape[-1],dtype=complex)
    for i in range(len(u)):
        left,s,right=np.linalg.svd(u[i].conj().T@u[(i+1)%len(u)])
        if s.min()<tol: raise ValueError("singular Wilson link; refine mesh")
        product=product@(left@right)
    return product


def wilson_flow(model,shape=(31,31),*,occupied=1,loop_axis=0,backend='numba'):
    """Sorted Wilson eigenphases for a family of 2D BZ loops (no Z2 inference).

    Returns evolution fractional coordinates and (nevolution,noccupied) phases.
    No band labels are assigned across crossings.
    """
    from .geometry import kmesh
    if model.dim!=2 or loop_axis not in (0,1) or len(shape)!=2 or min(shape)<3 or not 0<occupied<model.norb:
        raise ValueError("gapped 2D mesh with valid occupied count required")
    _periodic_bz(model,backend)
    e,u=model.bands(kmesh(shape,lattice=getattr(model,'lattice',None)),eigenvectors=True,backend=backend)
    if np.min(e[:,occupied]-e[:,occupied-1])<1e-9: raise ValueError("occupied subspace not gapped")
    u=u[:,:,:occupied].reshape(*shape,model.norb,occupied)
    if loop_axis==1: u=u.swapaxes(0,1)
    phases=np.array([np.sort(-np.angle(np.linalg.eigvals(_wilson_matrix(u[:,j])))) for j in range(u.shape[1])])
    return np.arange(u.shape[1])/u.shape[1],phases


def nested_polarization(model,shape=(31,31),*,occupied=2,phase_window=(-np.pi,0.),loop_axis=0,backend='numba',gap_tol=1e-7):
    """Nested determinant Wilson phase of an isolated Wannier phase sector.

    Uses physical Wannier-sector states at the base point of each inner loop.
    Returns polarization modulo 1. The phase window is in (-pi,pi); the
    Wannier gap must remain open and selected sector rank constant.
    """
    from scipy.linalg import schur
    from .geometry import kmesh
    if model.dim!=2 or loop_axis not in (0,1) or len(shape)!=2 or min(shape)<3 or not 0<occupied<model.norb:
        raise ValueError("gapped 2D occupied subspace required")
    lo,hi=phase_window
    if not -np.pi<=lo<hi<=np.pi: raise ValueError("phase window must lie in [-pi,pi]")
    _periodic_bz(model,backend)
    e,u=model.bands(kmesh(shape,lattice=getattr(model,'lattice',None)),eigenvectors=True,backend=backend)
    if np.min(e[:,occupied]-e[:,occupied-1])<gap_tol: raise ValueError("energy gap closed")
    u=u[:,:,:occupied].reshape(*shape,model.norb,occupied)
    if loop_axis==1:u=u.swapaxes(0,1)
    states=[];rank=None
    for j in range(u.shape[1]):
        w=_wilson_matrix(u[:,j]);triangular,v=schur(w,output='complex')
        phase=-np.angle(np.diag(triangular))
        if np.any(abs(np.angle(np.exp(1j*(phase-lo))))<gap_tol) or np.any(abs(np.angle(np.exp(1j*(phase-hi))))<gap_tol):
            raise ValueError("Wannier gap intersects phase-window boundary")
        select=(phase>lo)&(phase<hi);count=int(select.sum())
        if not 0<count<occupied or (rank is not None and count!=rank): raise ValueError("empty/full or changing Wannier sector")
        rank=count;states.append(u[0,j]@v[:,select])
    nested=_wilson_matrix(np.array(states))
    return float((-np.angle(np.linalg.det(nested))/(2*np.pi))%1)


def mirror_chern(model,mirror,*,occupied=(1,1),eigenvalues=(1j,-1j),shape=(31,31),backend='numba',tol=1e-8):
    """Chern numbers in two constant mirror sectors on a mirror-invariant 2D BZ.

    Returns (C_plus,C_minus,(C_plus-C_minus)/2). This assumes a k-independent
    unitary mirror representation commuting with H throughout the plane.
    """
    from scipy.linalg import schur
    from .base import TBkit
    from .geometry import kmesh
    matrix=np.asarray(mirror,complex)
    if matrix.shape!=(model.norb,model.norb) or not np.allclose(matrix.conj().T@matrix,np.eye(model.norb)):
        raise ValueError("unitary mirror of model dimension required")
    if len(occupied)!=2 or len(eigenvalues)!=2 or np.isclose(*eigenvalues): raise ValueError("two distinct mirror sectors required")
    tri,basis=schur(matrix,output='complex');vals=np.diag(tri)
    k=kmesh(shape,lattice=getattr(model,'lattice',None))
    for start in range(0,len(k),512):
        h=model.hamiltonian(k[start:start+512],backend=backend)
        if np.max(abs(matrix@h-h@matrix))>tol: raise ValueError("mirror does not commute on the integration plane")
    class Sector(TBkit):
        def __init__(self,v):
            self.v=v;self.dim=model.dim;self.norb=v.shape[1]
            if hasattr(model,'lattice'):self.lattice=model.lattice
        def hamiltonian(self,k,*,backend='numba'):return self.v.conj().T@model.hamiltonian(k,backend=backend)@self.v
        def gradient(self,k,*,backend='numba'):return self.v.conj().T@model.gradient(k,backend=backend)@self.v
    numbers=[];covered=np.zeros(model.norb,bool)
    for value,n in zip(eigenvalues,occupied):
        select=abs(vals-value)<tol;covered|=select
        if not select.any():raise ValueError("mirror sector missing")
        numbers.append(chern_number(Sector(basis[:,select]),shape,occupied=n,backend=backend))
    if not covered.all():raise ValueError("additional mirror sectors were not specified")
    return numbers[0],numbers[1],(numbers[0]-numbers[1])/2


def inversion_indicators(model,inversion,time_reversal,*,occupied,backend='numba',tol=1e-7):
    """Spinful inversion Z2 (2D/3D), weak indices and 3D Z4 indicator.

    Assumes a periodic orbital gauge and k-independent inversion P and
    time reversal U_T K with T^2=-1, [P,T]=0. Counts inversion-odd occupied
    Kramers pairs at TRIM, avoiding arbitrary pairing within degeneracies.
    """
    import itertools
    p=np.asarray(inversion,complex);t=np.asarray(time_reversal,complex);n=model.norb
    if model.dim not in (2,3) or occupied%2 or not 0<occupied<n: raise ValueError("even occupied count in a 2D/3D insulator required")
    if p.shape!=(n,n) or t.shape!=(n,n): raise ValueError("symmetry matrix dimension mismatch")
    if not np.allclose(p,p.conj().T,atol=tol) or not np.allclose(p@p,np.eye(n),atol=tol): raise ValueError("P must be a Hermitian involution")
    if not np.allclose(t@t.conj(),-np.eye(n),atol=tol) or not np.allclose(t.conj().T@t,np.eye(n),atol=tol): raise ValueError("spinful antiunitary T^2=-1 required")
    if not np.allclose(p@t,t@p.conj(),atol=tol): raise ValueError("inversion and time reversal must commute")
    reciprocal=_periodic_bz(model,backend)
    bits=np.array(list(itertools.product((0,1),repeat=model.dim)));k=(bits/2)@reciprocal
    h=model.hamiltonian(k,backend=backend);minus=model.hamiltonian(-k,backend=backend)
    # Also inspect generic k, since TRIM alone cannot establish symmetry.
    probe=np.array([[.137]*model.dim,[.217]*model.dim])@reciprocal
    for a,b in [(h,minus),(model.hamiltonian(probe,backend=backend),model.hamiltonian(-probe,backend=backend))]:
        if np.max(abs(a-p@b@p))>tol or np.max(abs(a-t@b.conj()@t.conj().T))>tol:
            raise ValueError("Hamiltonian violates inversion/time-reversal constraints")
    e,u=model.bands(k,eigenvectors=True,backend=backend)
    if np.min(e[:,occupied]-e[:,occupied-1])<tol: raise ValueError("occupied gap closed at TRIM")
    odd=[]
    for hh,v in zip(h,u[:,:,:occupied]):
        if np.max(abs(p@hh-hh@p))>tol: raise ValueError("P does not commute at TRIM; check orbital gauge")
        parity=np.linalg.eigvalsh(v.conj().T@p@v)
        if not np.allclose(abs(parity),1,atol=tol): raise ValueError("occupied subspace not invariant under inversion")
        count=int((parity<0).sum())
        if count%2: raise ValueError("inversion-odd states do not form Kramers pairs")
        odd.append(count//2)
    odd=np.array(odd);strong=int(odd.sum()%2)
    weak=tuple(int(odd[bits[:,a]==1].sum()%2) for a in range(3)) if model.dim==3 else ()
    return {'strong_z2':strong,'weak_z2':weak,'z4':int(odd.sum()%4) if model.dim==3 else None,'trim_bits':bits,'odd_pairs':odd}


def z2_wilson(model,time_reversal,*,occupied=2,shape=(41,41),backend='numba',tol=1e-7,diagnostics=False):
    """2D spinful Z2 by Wilson spectral-flow crossing parity, no inversion needed.

    shape=(points per closed kx loop, points on ky in [0,pi] inclusive).
    Reference phase is in the largest joint endpoint gap. Consecutive spectra
    are matched by minimal circular distance; excessive steps raise. Converge
    both mesh sizes for research use. Constant periodic-gauge T representation.
    """
    from scipy.optimize import linear_sum_assignment
    if model.dim!=2 or len(shape)!=2 or min(shape)<5 or occupied%2 or not 0<occupied<model.norb:
        raise ValueError('2D, even occupied count and mesh >=5 required')
    t=np.asarray(time_reversal,complex);n=model.norb
    if t.shape!=(n,n) or not np.allclose(t@t.conj(),-np.eye(n),atol=tol) or not np.allclose(t.conj().T@t,np.eye(n),atol=tol):
        raise ValueError('unitary spinful time reversal T^2=-1 required')
    reciprocal=_periodic_bz(model,backend)
    ny=shape[1];spectra=[]
    for y in np.linspace(0,.5,ny):
        fractional=np.column_stack([np.arange(shape[0])/shape[0],np.full(shape[0],y)])
        k=fractional@reciprocal
        h=model.hamiltonian(k,backend=backend);minus=model.hamiltonian(-k,backend=backend)
        if np.max(abs(h-t@minus.conj()@t.conj().T))>tol:raise ValueError('time reversal violated on Wilson mesh')
        e,u=model.bands(k,eigenvectors=True,backend=backend)
        if np.min(e[:,occupied]-e[:,occupied-1])<tol:raise ValueError('occupied gap closes on Wilson mesh')
        phases=(-np.angle(np.linalg.eigvals(_wilson_matrix(u[:,:,:occupied])))/(2*np.pi))%1
        spectra.append(np.sort(phases))
    spectra=np.array(spectra)
    # Kramers endpoint spectra must have even degeneracy at every phase.
    for endpoint in (spectra[0],spectra[-1]):
        unused=list(range(occupied))
        while unused:
            i=unused.pop(0)
            distances=[abs((endpoint[j]-endpoint[i]+.5)%1-.5) for j in unused]
            if not distances or min(distances)>tol*10:raise ValueError('Wilson endpoint Kramers pairing failed; check periodic gauge')
            unused.pop(int(np.argmin(distances)))
    endpoints=np.sort(np.r_[spectra[0],spectra[-1]])
    gaps=np.diff(np.r_[endpoints,endpoints[0]+1]);index=int(np.argmax(gaps))
    reference=(endpoints[index]+gaps[index]/2)%1
    if gaps[index]<tol:raise ValueError('no endpoint gap for reference phase')
    crossings=0;previous=spectra[0].copy()
    for current in spectra[1:]:
        delta=(current[None,:]-previous[:,None]+.5)%1-.5
        rows,cols=linear_sum_assignment(abs(delta))
        steps=delta[rows,cols]
        if np.max(abs(steps))>.25:raise ValueError('Wilson flow step too large; refine mesh')
        following=previous+steps
        crossings+=int(np.abs(np.floor(following-reference)-np.floor(previous-reference)).sum())
        previous=following%1
    value=crossings%2
    if diagnostics:return {'z2':value,'reference_phase':float(reference),'crossings':crossings,'phases':spectra,'ky_fractional':np.linspace(0,.5,ny)}
    return value


def wilson_transport(model,k,*,occupied=1,backend='numba'):
    """Open-path unitary transport matrix, covariant under endpoint gauges.

    Under frame changes u_i -> u_i G_i, W -> G_0^dagger W G_last.
    Its eigenphases are not a physical closed-loop Berry phase.
    """
    if not 0<occupied<=model.norb:raise ValueError('invalid occupied count')
    e,u=model.bands(k,eigenvectors=True,backend=backend)
    if e.ndim!=2 or len(e)<2:raise ValueError('at least two k points required')
    if occupied<model.norb and np.min(e[:,occupied]-e[:,occupied-1])<1e-9:raise ValueError('occupied subspace not gapped')
    u=u[:,:,:occupied];product=np.eye(occupied,dtype=complex)
    for i in range(len(u)-1):
        left,s,right=np.linalg.svd(u[i].conj().T@u[i+1])
        if s.min()<1e-10:raise ValueError('singular Wilson link; refine mesh')
        product=product@(left@right)
    return product
