"""Principal-layer surface Green functions and coherent Landauer transport."""
import numpy as np
from scipy.linalg import solve
from .base import hermitian


def surface_green(energy,onsite,hopping,*,eta=1e-6,tol=1e-12,maxiter=200):
    """Lopez-Sancho decimation for a semi-infinite nearest-layer lead.

    hopping couples layer i to i+1; reversing it reverses the exposed surface.
    Energies and eta use the Hamiltonian energy unit.
    """
    h=np.asarray(onsite,complex); v=np.asarray(hopping,complex)
    if h.ndim!=2 or h.shape[0]!=h.shape[1] or v.shape!=h.shape or eta<=0 or maxiter<1:
        raise ValueError("square equal matrices, eta>0 and maxiter>0 required")
    hermitian(h)
    eye=np.eye(len(h),dtype=complex); z=(energy+1j*eta)*eye
    bulk=h.copy(); surface=h.copy(); alpha=v.copy(); beta=v.conj().T.copy()
    for _ in range(maxiter):
        g=solve(z-bulk,eye)
        agb=alpha@g@beta; bga=beta@g@alpha
        surface+=agb; bulk+=agb+bga
        alpha,beta=alpha@g@alpha,beta@g@beta
        if max(np.linalg.norm(alpha),np.linalg.norm(beta))<tol:
            result=solve(z-surface,eye)
            # Decimation can lose digits near a symmetric band center with
            # very small eta. Check the defining Riccati equation, not just
            # the size of the renormalized hopping.
            a=z-h
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
            if np.linalg.eigvalsh(1j*(result-result.conj().T)).min() < -1e-8:
                raise RuntimeError("nonretarded surface solution")
            return result
    raise RuntimeError("surface Green function failed to converge")


def transmission(energy,device,sigma_left,sigma_right,*,eta=1e-12):
    """Caroli transmission Tr[Gamma_L G^r Gamma_R G^a]."""
    h=np.asarray(device,complex); sl=np.asarray(sigma_left,complex); sr=np.asarray(sigma_right,complex)
    if h.ndim!=2 or h.shape[0]!=h.shape[1] or sl.shape!=h.shape or sr.shape!=h.shape or eta<0:
        raise ValueError("equal square device/self-energy matrices required")
    hermitian(h)
    gl=1j*(sl-sl.conj().T); gr=1j*(sr-sr.conj().T)
    if min(np.linalg.eigvalsh(gl).min(),np.linalg.eigvalsh(gr).min()) < -1e-9:
        raise ValueError("self energies must be retarded (positive broadening)")
    g=solve((energy+1j*eta)*np.eye(len(h))-h-sl-sr,np.eye(len(h)))
    return float(np.trace(gl@g@gr@g.conj().T).real)


class PrincipalLayer:
    """Nearest-layer lead assembled from a possibly long-range HR model."""
    def __init__(self,onsite,hopping,*,width=1,axis=0):
        self.onsite=np.asarray(onsite,complex);self.hopping=np.asarray(hopping,complex)
        if self.onsite.ndim!=2 or self.onsite.shape[0]!=self.onsite.shape[1] or self.hopping.shape!=self.onsite.shape:
            raise ValueError("equal square principal-layer blocks required")
        hermitian(self.onsite);self.width=width;self.axis=axis

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
    hermitian(onsite)
    if not np.allclose(backward,forward.conj().T,atol=1e-10):raise ValueError("non-Hermitian lead coupling")
    return PrincipalLayer(onsite,forward,width=width,axis=axis)


def chain_transmission(lead,energy,layers=1,*,eta=1e-6):
    """Perfect finite principal-layer chain between identical semi-infinite leads.

    A diagnostic/construction helper, dense in device size. For arbitrary
    devices pass assembled retarded self energies to transmission directly.
    """
    if not isinstance(layers,int) or layers<1:raise ValueError("positive device layer count required")
    n=len(lead.onsite);device=np.kron(np.eye(layers),lead.onsite)
    if layers>1:
        device+=np.kron(np.diag(np.ones(layers-1),1),lead.hopping)
        device+=np.kron(np.diag(np.ones(layers-1),-1),lead.hopping.conj().T)
    left=np.zeros_like(device);right=np.zeros_like(device)
    left[:n,:n]=lead.self_energy(energy,side='left',eta=eta)
    right[-n:,-n:]=lead.self_energy(energy,side='right',eta=eta)
    return transmission(energy,device,left,right,eta=1e-12)
