"""Real s/p/d orbitals and angular momentum without MATLAB dependencies."""
import math
import numpy as np
from scipy.linalg import expm,block_diag
from .angular_kernels import angular_momentum_doubled

ORBITALS=('s','px','py','pz','dxy','dyz','dzx','dx2-y2','dz2')
# Orthonormal traceless quadratic tensors: orbital orientation is explicit.
D_TENSORS=np.zeros((5,3,3))
for n,(i,j) in enumerate(((0,1),(1,2),(2,0))):
    D_TENSORS[n,i,j]=D_TENSORS[n,j,i]=1/np.sqrt(2)
D_TENSORS[3]=np.diag([1,-1,0])/np.sqrt(2)
D_TENSORS[4]=np.diag([-1,-1,2])/np.sqrt(6)


def _twice_j(value):
    if (isinstance(value,(bool,np.bool_,complex,np.complexfloating)) or not np.isscalar(value)):
        raise ValueError('nonnegative integer/half-integer j required')
    try:
        integer=int(value);remainder=2*(value-integer)
        valid=value>=0 and (remainder==0 or remainder==1)
    except (ValueError,TypeError,OverflowError):valid=False
    if not valid:raise ValueError('finite nonnegative exact integer/half-integer j required')
    return 2*integer+int(remainder)


def _dimension_limit(value):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<1:
        raise ValueError('positive integer max_dimension required')
    return int(value)


def _rotation_axis(axis):
    raw=np.asarray(axis)
    if np.iscomplexobj(raw):raise ValueError('real rotation axis required')
    try:value=np.asarray(raw,dtype=float)
    except (TypeError,ValueError) as exc:raise ValueError('finite real 3D axis required') from exc
    if value.shape!=(3,) or not np.isfinite(value).all():raise ValueError('finite real 3D axis required')
    scale=np.max(np.abs(value))
    if scale==0:raise ValueError('nonzero rotation axis required')
    scaled=value/scale
    return scaled/np.linalg.norm(scaled)


def _rotation_angle(angle):
    if isinstance(angle,(bool,np.bool_,complex,np.complexfloating)) or np.ndim(angle)!=0:
        raise ValueError('finite real scalar rotation angle required')
    try:value=float(angle)
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError('finite real scalar rotation angle required') from exc
    if not np.isfinite(value):raise ValueError('finite real scalar rotation angle required')
    if abs(value)<=2*math.pi:return value
    # Reduce the SU(2) half-angle with libm argument reduction, retaining the
    # central sign at 2*pi. fmod(angle, 2*pi) would erase spinor information.
    return 2*math.atan2(math.sin(value/2),math.cos(value/2))


def orbital_rotation(rotation):
    """Real orthogonal representation in ORBITALS order, including inversion."""
    raw=np.asarray(rotation)
    if np.iscomplexobj(raw):raise ValueError('real orthogonal 3x3 matrix required')
    q=np.asarray(raw,float)
    if (q.shape!=(3,3) or not np.isfinite(q).all()
            or not np.allclose(q.T@q,np.eye(3),atol=1e-12,rtol=0)):
        raise ValueError("real orthogonal 3x3 matrix required (absolute tolerance 1e-12)")
    d=np.einsum('aij,ik,bkl,jl->ab',D_TENSORS,q,D_TENSORS,q,optimize=True)
    return block_diag(np.ones((1,1)),q,d)


def angular_momentum(j,*,max_dimension=1025):
    """Jx,Jy,Jz, m=j..-j; hbar=1 and exactly discrete nonnegative j.

    Near-half-integers are rejected, not silently rounded. Dense dimension
    defaults to 1025; explicitly raise max_dimension for larger allocations.
    """
    doubled=_twice_j(j);limit=_dimension_limit(max_dimension)
    if doubled+1>limit:raise ValueError('angular momentum exceeds max_dimension')
    return angular_momentum_doubled(doubled,limit)


def spin_rotation(axis,angle,*,spin=.5,max_dimension=1025):
    """Active unitary exp(-i angle n.J), for a finite real axis/angle."""
    axis=_rotation_axis(axis);angle=_rotation_angle(angle)
    return expm(-1j*angle*sum(v*j for v,j in zip(axis,angular_momentum(spin,max_dimension=max_dimension))))


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
