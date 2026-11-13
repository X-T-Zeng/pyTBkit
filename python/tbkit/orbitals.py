"""Real s/p/d orbitals and angular momentum without MATLAB dependencies."""
import numpy as np
from scipy.linalg import expm,block_diag

ORBITALS=('s','px','py','pz','dxy','dyz','dzx','dx2-y2','dz2')
# Orthonormal traceless quadratic tensors: orbital orientation is explicit.
D_TENSORS=np.zeros((5,3,3))
for n,(i,j) in enumerate(((0,1),(1,2),(2,0))):
    D_TENSORS[n,i,j]=D_TENSORS[n,j,i]=1/np.sqrt(2)
D_TENSORS[3]=np.diag([1,-1,0])/np.sqrt(2)
D_TENSORS[4]=np.diag([-1,-1,2])/np.sqrt(6)


def orbital_rotation(rotation):
    """Real orthogonal representation in ORBITALS order, including inversion."""
    q=np.asarray(rotation,float)
    if q.shape!=(3,3) or not np.allclose(q.T@q,np.eye(3),atol=1e-12): raise ValueError("orthogonal 3x3 matrix required")
    d=np.einsum('aij,ik,bkl,jl->ab',D_TENSORS,q,D_TENSORS,q,optimize=True)
    return block_diag(np.ones((1,1)),q,d)


def angular_momentum(j):
    """(Jx,Jy,Jz) in |j,m>, m=j,j-1,...,-j; hbar=1."""
    if j<0 or not np.isclose(2*j,round(2*j)): raise ValueError("nonnegative integer/half-integer j required")
    n=int(round(2*j))+1; m=np.linspace(j,-j,n)
    plus=np.zeros((n,n),complex)
    for col in range(1,n): plus[col-1,col]=np.sqrt(j*(j+1)-m[col]*(m[col]+1))
    minus=plus.conj().T
    return (plus+minus)/2,(plus-minus)/(2j),np.diag(m).astype(complex)


def spin_rotation(axis,angle,*,spin=.5):
    axis=np.asarray(axis,float)
    if axis.shape!=(3,) or np.linalg.norm(axis)==0: raise ValueError("nonzero 3D axis required")
    axis=axis/np.linalg.norm(axis)
    return expm(-1j*angle*sum(v*j for v,j in zip(axis,angular_momentum(spin))))


def real_orbital_angular_momentum():
    """Lx,Ly,Lz in real s/p/d basis; generators of orbital_rotation."""
    generators=[]
    for a in range(3):
        axis=np.eye(3)[a]
        cross=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        dp=cross
        dd=np.einsum('aij,bij->ab',D_TENSORS,cross@D_TENSORS-D_TENSORS@cross)
        generators.append(1j*block_diag(np.zeros((1,1)),dp,dd))
    return tuple(generators)


def spin_orbit_coupling(strength=1.0):
    """lambda L.S on the full real s/p/d tensor spin-1/2 basis (18x18)."""
    return strength*sum(np.kron(l,s) for l,s in zip(real_orbital_angular_momentum(),angular_momentum(.5)))
