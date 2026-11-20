"""Real row-vector cells: unit-independent validation and reciprocal geometry.

Only temporary arrays are scaled. Stored cells and Cartesian coordinates keep
the caller's length units and orientation; no implicit Angstrom/SI conversion.
"""
import numpy as np


def checked_lattice(lattice, dim=None):
    """Copy a finite real square cell with numerical full rank.

    Rank is tested relative to the largest singular value after component
    scaling, using eps*dimension. This is a numerical-domain guard, not a
    proof of accurate inversion for arbitrarily ill-conditioned cells.
    """
    if np.iscomplexobj(lattice):
        raise ValueError('lattice must be real')
    a=np.array(lattice,dtype=float,copy=True,order='C')
    if dim is None:dim=a.shape[0] if a.ndim==2 else 0
    if dim<1 or a.shape!=(dim,dim) or not np.isfinite(a).all():
        raise ValueError('finite square invertible lattice required')
    scale=np.max(abs(a))
    if scale==0.:raise ValueError('invertible lattice required')
    singular=np.linalg.svd(a/scale,compute_uv=False)
    if singular[-1]<=np.finfo(float).eps*dim*singular[0]:
        raise ValueError('singular or numerically rank-deficient lattice')
    return a


def inverse_lattice(lattice):
    a=checked_lattice(lattice);scale=np.max(abs(a))
    with np.errstate(over='ignore',under='ignore'):
        inverse=np.linalg.solve(a/scale,np.eye(len(a)))/scale
    if not np.isfinite(inverse).all():
        raise ValueError('inverse lattice not representable; change length units')
    return inverse


def reciprocal_lattice(lattice):
    """Rows B satisfy A @ B.T = 2*pi*I, with inverse length units."""
    with np.errstate(over='ignore'):
        reciprocal=2*np.pi*inverse_lattice(lattice).T
    if not np.isfinite(reciprocal).all():
        raise ValueError('reciprocal lattice not representable; change length units')
    return reciprocal


def log_cell_volume(lattice):
    a=checked_lattice(lattice);scale=np.max(abs(a))
    return float(np.linalg.slogdet(a/scale)[1]+len(a)*np.log(scale))


def finite_measure(log_measure):
    with np.errstate(over='ignore',under='ignore'):
        measure=float(np.exp(log_measure))
    if not np.isfinite(measure) or measure==0.:
        raise ValueError('cell/BZ measure not representable; change length units or use a normalized average')
    return measure


def same_lattice(a,b,*,rtol=1e-12):
    """Relative Frobenius comparison in COMMON length units, no absolute floor.

    Equal geometry expressed in different unit systems is not guessed. A basis
    rotation or integer relabeling also needs an explicit model transformation.
    """
    if np.ndim(rtol)!=0 or np.iscomplexobj(rtol) or not np.isfinite(rtol) or rtol<0:
        raise ValueError('finite nonnegative lattice tolerance required')
    a=checked_lattice(a);b=checked_lattice(b)
    if a.shape!=b.shape:return False
    if rtol==0.:return bool(np.array_equal(a,b))
    scale=max(np.max(abs(a)),np.max(abs(b)))
    x,y=a/scale,b/scale
    return bool(np.linalg.norm(x-y)<=rtol*max(np.linalg.norm(x),np.linalg.norm(y)))
