"""Principal-layer surface Green functions and coherent Landauer transport."""
import numpy as np
from scipy.linalg import solve
from .base import hermitian


def _real_scalar(value,name):
    if np.ndim(value)!=0 or np.iscomplexobj(value):
        raise ValueError(f'{name} must be a finite real scalar')
    try:value=float(value)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError(f'{name} must be a finite real scalar') from exc
    if not np.isfinite(value):raise ValueError(f'{name} must be a finite real scalar')
    return value


def _energy_scale(inverse_bare,*couplings,eta):
    # Use E I-H, not the absolute energy origin. A simultaneous E/H shift
    # must not change convergence criteria. Max-entry norms avoid squaring tiny
    # or huge energies before scaling.
    scale=max(eta,float(np.max(abs(inverse_bare))),
              *(float(np.max(abs(c))) for c in couplings))
    if not np.isfinite(scale):raise ValueError('energy scale overflow; change numerical units')
    return scale if scale else 1.


def _check_hermitian(matrix):
    """Check a block relative to its nontrivial part, preserving energy shifts."""
    if (matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1] or not len(matrix)
            or not np.isfinite(matrix).all()):
        raise ValueError('finite nonempty square Hamiltonian required')
    shifted=matrix.copy()
    diagonal=np.diag_indices(len(matrix))
    shifted[diagonal]-=matrix[0,0].real
    magnitude=float(np.max(abs(shifted)))
    hermitian(shifted/(magnitude if magnitude else 1.),atol=1e-12)


def _check_positive(matrix,description):
    """PSD check relative to this matrix, independent of its physical unit."""
    magnitude=float(np.max(abs(matrix)))
    if magnitude:
        eigenvalues=np.linalg.eigvalsh(matrix/magnitude)
        rounding=64*len(matrix)*np.finfo(float).eps*float(np.max(abs(eigenvalues)))
        if eigenvalues[0]<-rounding:
            raise ValueError(description)


def surface_green(energy,onsite,hopping,*,eta=1e-6,tol=1e-12,maxiter=200):
    """Lopez-Sancho decimation for a semi-infinite nearest-layer lead.

    hopping couples layer i to i+1; reversing it reverses the exposed surface.
    Energies and eta use the Hamiltonian energy unit. tol is dimensionless:
    the iteration and Riccati checks use E I-H and hopping scaled together.
    A finite eta is required; results should be converged as eta decreases.
    """
    h=np.asarray(onsite,complex); v=np.asarray(hopping,complex)
    energy=_real_scalar(energy,'energy');eta=_real_scalar(eta,'eta');tol=_real_scalar(tol,'tol')
    if (h.ndim!=2 or h.shape[0]!=h.shape[1] or v.shape!=h.shape or eta<=0 or tol<=0
            or not isinstance(maxiter,(int,np.integer)) or maxiter<1 or not np.isfinite(v).all()):
        raise ValueError("square equal matrices, eta>0 and maxiter>0 required")
    _check_hermitian(h)
    eye=np.eye(len(h),dtype=complex)
    bare=energy*eye-h
    scale=_energy_scale(bare,v,eta=eta)
    a=bare/scale+1j*(eta/scale)*eye;v=v/scale
    # Store inverse Green blocks rather than large absolute on-site energies.
    bulk=a.copy();surface=a.copy();alpha=v.copy();beta=v.conj().T.copy()
    for _ in range(maxiter):
        g=solve(bulk,eye)
        agb=alpha@g@beta; bga=beta@g@alpha
        surface-=agb;bulk-=agb+bga
        alpha,beta=alpha@g@alpha,beta@g@beta
        if max(np.linalg.norm(alpha),np.linalg.norm(beta))<tol:
            result=solve(surface,eye)
            # Decimation can lose digits near a symmetric band center with
            # very small eta. Check the defining Riccati equation, not just
            # the size of the renormalized hopping.
            def residual(g):
                return (a-v@g@v.conj().T)@g-eye
            error=np.linalg.norm(residual(result))
            if error > max(100*tol,1e-10):
                if len(h)>16:
                    raise RuntimeError("surface residual too large; increase eta or use smaller principal layers")
                # Newton correction in column-major vectorization. Only used
                # for small layers, since the Jacobian requires O(n^4) memory.
                for _ in range(8):
                    f=residual(result)
                    jac=np.kron(eye,a-v@result@v.conj().T)-np.kron((v.conj().T@result).T,v)
                    delta=solve(jac,-f.ravel(order="F")).reshape(h.shape,order="F")
                    result+=delta
                    if np.linalg.norm(residual(result))<max(100*tol,1e-10): break
                else: raise RuntimeError("surface Riccati refinement failed")
            try:_check_positive(1j*(result-result.conj().T),'nonretarded surface solution')
            except ValueError as exc:raise RuntimeError(str(exc)) from exc
            return result/scale
    raise RuntimeError("surface Green function failed to converge")


def transmission(energy,device,sigma_left,sigma_right,*,eta=0.):
    """Caroli transmission Tr[Gamma_L G^r Gamma_R G^a].

    Retarded self energies supply broadening. Extra device eta defaults to zero;
    a positive explicit value adds absorption and may suppress transmission.
    Self energies must already be embedded in the full device basis.
    """
    h=np.asarray(device,complex); sl=np.asarray(sigma_left,complex); sr=np.asarray(sigma_right,complex)
    energy=_real_scalar(energy,'energy');eta=_real_scalar(eta,'eta')
    if (h.ndim!=2 or h.shape[0]!=h.shape[1] or sl.shape!=h.shape or sr.shape!=h.shape or eta<0
            or not np.isfinite(sl).all() or not np.isfinite(sr).all()):
        raise ValueError("equal square device/self-energy matrices required")
    _check_hermitian(h)
    eye=np.eye(len(h),dtype=complex);bare=energy*eye-h
    scale=_energy_scale(bare,sl,sr,eta=eta)
    sl=sl/scale;sr=sr/scale
    gl=1j*(sl-sl.conj().T);gr=1j*(sr-sr.conj().T)
    for gamma in (gl,gr):
        _check_positive(gamma,'self energies must be retarded (positive broadening)')
    g=solve(bare/scale+1j*(eta/scale)*eye-sl-sr,eye)
    return float(np.trace(gl@g@gr@g.conj().T).real)


class PrincipalLayer:
    """Nearest-layer lead assembled from a possibly long-range HR model."""
    def __init__(self,onsite,hopping,*,width=1,axis=0):
        self.onsite=np.asarray(onsite,complex);self.hopping=np.asarray(hopping,complex)
        if self.onsite.ndim!=2 or self.onsite.shape[0]!=self.onsite.shape[1] or self.hopping.shape!=self.onsite.shape:
            raise ValueError("equal square principal-layer blocks required")
        _check_hermitian(self.onsite)
        if not np.isfinite(self.hopping).all():raise ValueError('finite lead hopping required')
        self.width=width;self.axis=axis

    def surface(self,energy,*,side='right',eta=1e-6):
        if side not in ('left','right'):raise ValueError("side must be left/right")
        v=self.hopping if side=='right' else self.hopping.conj().T
        return surface_green(energy,self.onsite,v,eta=eta)

    def self_energy(self,energy,*,side='right',eta=1e-6):
        coupling=self.hopping if side=='right' else self.hopping.conj().T
        return coupling@self.surface(energy,side=side,eta=eta)@coupling.conj().T


def principal_layer(model,*,axis=0,transverse_fractional=None):
    """Group max|R_axis| primitive cells into nearest-neighbor lead layers.

    transverse_fractional has model dimension with its axis component zero.
    This convention remains unambiguous for skew real-space lattices.
    """
    if not 0<=axis<model.dim:raise ValueError("invalid transport axis")
    k=np.zeros(model.dim) if transverse_fractional is None else np.asarray(transverse_fractional,float)
    if k.shape!=(model.dim,) or not np.isclose(k[axis],0) or not np.isfinite(k).all():raise ValueError("transverse fractional k must have zero transport component")
    width=max(1,int(np.max(abs(model.vectors[:,axis]))))
    repeats=np.ones(model.dim,int);repeats[axis]=width;large=model.supercell(repeats)
    onsite=np.zeros((large.norb,large.norb),complex);forward=np.zeros_like(onsite);backward=np.zeros_like(onsite)
    for r,h in zip(large.vectors,large.hopping):
        if abs(r[axis])>1:raise RuntimeError("principal-layer construction left long-range hopping")
        phase=np.exp(2j*np.pi*np.dot(k,r))
        if r[axis]==0:onsite+=phase*h
        elif r[axis]==1:forward+=phase*h
        else:backward+=phase*h
    _check_hermitian(onsite)
    magnitude=max(float(np.max(abs(backward))),float(np.max(abs(forward))))
    if magnitude and not np.allclose(backward/magnitude,forward.conj().T/magnitude,atol=1e-12,rtol=1e-10):
        raise ValueError("non-Hermitian lead coupling")
    return PrincipalLayer(onsite,forward,width=width,axis=axis)


def chain_transmission(lead,energy,layers=1,*,eta=1e-6,device_eta=0.):
    """Perfect finite principal-layer chain between identical semi-infinite leads.

    A diagnostic/construction helper, dense in device size. For arbitrary
    devices pass assembled retarded self energies to transmission directly.
    eta broadens the leads; device_eta is an explicit, separate device absorber.
    Their numerical values must be rescaled with the Hamiltonian energy unit.
    """
    if not isinstance(layers,int) or layers<1:raise ValueError("positive device layer count required")
    n=len(lead.onsite);device=np.kron(np.eye(layers),lead.onsite)
    if layers>1:
        device+=np.kron(np.diag(np.ones(layers-1),1),lead.hopping)
        device+=np.kron(np.diag(np.ones(layers-1),-1),lead.hopping.conj().T)
    left=np.zeros_like(device);right=np.zeros_like(device)
    left[:n,:n]=lead.self_energy(energy,side='left',eta=eta)
    right[-n:,-n:]=lead.self_energy(energy,side='right',eta=eta)
    return transmission(energy,device,left,right,eta=device_eta)
