"""Classical tensor exchange and small-cluster quantum Heisenberg models."""
import numpy as np
from numba import njit
from scipy import sparse
from .hr import immutable
from .orbitals import angular_momentum


@njit(cache=True,nogil=True)
def exchange_energy_field(spins,edges,couplings,anisotropy,field):
    # Each edge is included once: E=-sum_ij S_i.J_ij.S_j - sum S_i.K_i.S_i - B.S.
    energy=0.;effective=np.zeros_like(spins)
    for e in range(len(edges)):
        i,j=edges[e]
        for a in range(3):
            for b in range(3):
                energy-=spins[i,a]*couplings[e,a,b]*spins[j,b]
                effective[i,a]+=couplings[e,a,b]*spins[j,b]
                effective[j,b]+=couplings[e,a,b]*spins[i,a]
    for i in range(len(spins)):
        for a in range(3):
            energy-=field[i,a]*spins[i,a]
            effective[i,a]+=field[i,a]
            for b in range(3):
                energy-=spins[i,a]*anisotropy[i,a,b]*spins[i,b]
                effective[i,a]+=(anisotropy[i,a,b]+anisotropy[i,b,a])*spins[i,b]
    return energy,effective


class SpinModel:
    """Finite classical spin system. Spins are Cartesian dimensionless vectors."""
    def __init__(self,nsites,edges,couplings,*,anisotropy=None,field=None):
        if not isinstance(nsites,int) or nsites<1: raise ValueError("positive nsites required")
        edges=np.asarray(edges,int).reshape(-1,2)
        if np.any(edges<0) or np.any(edges>=nsites) or np.any(edges[:,0]==edges[:,1]): raise ValueError("invalid exchange edge")
        coupling=np.asarray(couplings,float)
        if coupling.ndim==0: coupling=np.repeat((coupling*np.eye(3))[None],len(edges),axis=0)
        elif coupling.shape==(len(edges),): coupling=coupling[:,None,None]*np.eye(3)
        if coupling.shape!=(len(edges),3,3): raise ValueError("one scalar/tensor coupling per edge required")
        anisotropy=np.zeros((nsites,3,3)) if anisotropy is None else np.asarray(anisotropy,float)
        field=np.zeros((nsites,3)) if field is None else np.broadcast_to(field,(nsites,3))
        if anisotropy.shape!=(nsites,3,3): raise ValueError("anisotropy must be (nsites,3,3)")
        self.nsites=nsites;self.edges=immutable(edges,np.int64);self.couplings=immutable(coupling,float)
        self.anisotropy=immutable(anisotropy,float);self.field=immutable(field,float)

    def energy_field(self,spins):
        spins=np.ascontiguousarray(spins,dtype=float)
        if spins.shape!=(self.nsites,3) or not np.isfinite(spins).all(): raise ValueError("finite spins (nsites,3) required")
        return exchange_energy_field(spins,self.edges,self.couplings,self.anisotropy,self.field)

    def evolve(self,spins,dt,steps,*,damping=.1,gyromagnetic=1.):
        """Deterministic projected RK4 Landau-Lifshitz-Gilbert dynamics.

        Unit spin vectors only; no thermal noise. User supplies consistent
        time/energy/gyromagnetic units; check dt convergence.
        """
        s=np.asarray(spins,float).copy()
        if s.shape!=(self.nsites,3) or not np.allclose(np.linalg.norm(s,axis=1),1): raise ValueError("unit initial spins required")
        if dt<=0 or damping<0 or not isinstance(steps,int) or steps<0: raise ValueError("invalid integration parameters")
        def derivative(x):
            _,b=self.energy_field(x);cross=np.cross(x,b)
            return -gyromagnetic*(cross+damping*np.cross(x,cross))/(1+damping**2)
        trajectory=[s.copy()]
        for _ in range(steps):
            k1=derivative(s);k2=derivative(s+dt*k1/2);k3=derivative(s+dt*k2/2);k4=derivative(s+dt*k3)
            s+=dt*(k1+2*k2+2*k3+k4)/6;s/=np.linalg.norm(s,axis=1)[:,None]
            trajectory.append(s.copy())
        return np.array(trajectory)


def heisenberg_quantum(nsites,edges,couplings,*,spin=.5,max_dimension=10000):
    """Sparse small-cluster H=+sum J_ij S_i.S_j (antiferromagnetic J>0)."""
    n=int(round(2*spin))+1
    if not isinstance(nsites,int) or nsites<1 or n**nsites>max_dimension: raise ValueError("quantum cluster exceeds max_dimension")
    matrices=angular_momentum(spin);edges=np.asarray(edges,int).reshape(-1,2)
    couplings=np.broadcast_to(np.asarray(couplings,float),(len(edges),))
    if np.any(edges<0) or np.any(edges>=nsites) or np.any(edges[:,0]==edges[:,1]): raise ValueError("invalid exchange edges")
    out=sparse.csr_matrix((n**nsites,)*2,dtype=complex)
    for (i,j),value in zip(edges,couplings):
        for operator in matrices:
            term=sparse.csr_matrix([[1.]])
            for site in range(nsites):term=sparse.kron(term,operator if site in (i,j) else sparse.eye(n),format='csr')
            out+=value*term
    return out
