"""Small reproducible reference models in a periodic orbital gauge."""
import numpy as np
from .hr import HR
from .symbolic import HK

identity=np.eye(2,dtype=complex)
sigma_x=np.array([[0,1],[1,0]],complex)
sigma_y=np.array([[0,-1j],[1j,0]],complex)
sigma_z=np.diag([1,-1]).astype(complex)


def ssh(intracell=0.6,intercell=1.0):
    h0=intracell*sigma_x
    h1=np.array([[0,0],[intercell,0]],complex)
    return HR([[0],[1],[-1]],[h0,h1,h1.conj().T])


def qi_wu_zhang(mass=-1.0):
    """H=sin(kx)sx+sin(ky)sy+(m+cos(kx)+cos(ky))sz."""
    hx=(sigma_z-1j*sigma_x)/2
    hy=(sigma_z-1j*sigma_y)/2
    return HR([[0,0],[1,0],[-1,0],[0,1],[0,-1]],
              [mass*sigma_z,hx,hx.conj().T,hy,hy.conj().T])


def dirac(mass=0.2,velocity=1.0):
    return HK({(0,0):mass*sigma_z,(1,0):velocity*sigma_x,(0,1):velocity*sigma_y})


def bbh(gamma=.5,coupling=1.):
    """Four-band Benalcazar-Bernevig-Hughes quadrupole model, pi flux."""
    from .symbolic import Htrig
    g1=-np.kron(sigma_y,sigma_x)
    g2=-np.kron(sigma_y,sigma_y)
    g3=-np.kron(sigma_y,sigma_z)
    g4=np.kron(sigma_x,identity)
    return Htrig([[0,0],[1,0],[0,1]],
                 [gamma*(g4+g2),coupling*g4,coupling*g2],
                 [np.zeros((4,4)),coupling*g3,coupling*g1]).to_hr()


def topological_insulator(mass=2.):
    """3D Wilson-Dirac lattice TI: (sum cos k-m) tau_z + sin k tau_x sigma."""
    z=np.kron(sigma_z,identity);terms={(0,0,0):-mass*z}
    for a,s in enumerate((sigma_x,sigma_y,sigma_z)):
        r=np.zeros(3,int);r[a]=1
        h=(z-1j*np.kron(sigma_x,s))/2
        terms[tuple(r)]=h;terms[tuple(-r)]=h.conj().T
    return HR(list(terms),list(terms.values()))


def quantum_spin_hall(mass=-1.,mixing=0.):
    """TR pair of QWZ blocks with an optional spin-mixing onsite term.

    Basis is spin block then two orbitals, T=(i sigma_y) tensor I K.
    mixing*sx_spin tensor sy_orb preserves T and can break inversion.
    """
    from .construction import direct_sum
    up=qi_wu_zhang(mass)
    down=HR(up.vectors,up.hopping.conj(),lattice=up.lattice)
    full=direct_sum(up,down)
    return full.with_hopping([0,0],mixing*np.kron(sigma_x,sigma_y),add=True)
