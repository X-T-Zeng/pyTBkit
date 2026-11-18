"""Common spectral API. Energies have the units of the input Hamiltonian."""
from abc import ABC, abstractmethod
from numbers import Real
import numpy as np
from scipy.sparse.linalg import eigsh
from .validation_kernels import hermiticity_metrics


def points(k, dim):
    if np.iscomplexobj(k):raise ValueError("physical k points must be real; complex-band continuation is a separate API")
    k = np.asarray(k, dtype=float)
    single = k.ndim == 1
    k = np.atleast_2d(k)
    if k.ndim != 2 or k.shape[1] != dim or not np.isfinite(k).all():
        raise ValueError(f"k must have shape ({dim},) or (nk,{dim}) and be finite")
    return np.ascontiguousarray(k), single


def _hermitian_tolerances(atol, rtol):
    for value in (atol, rtol):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError("Hermiticity tolerances must be finite nonnegative real scalars")
        if not np.isfinite(value) or value < 0:
            raise ValueError("Hermiticity tolerances must be finite nonnegative real scalars")


def _absolute_defect_ok(defect, atol):
    # Used for deliberately absolute-only checks. Work in the defect's own
    # scale so a tiny asymmetry beside a huge diagonal remains observable.
    if not np.isfinite(defect).all():
        return False
    scale = max(np.max(abs(defect.real), initial=0.), np.max(abs(defect.imag), initial=0.))
    if scale == 0.:
        return True
    with np.errstate(over='ignore'):
        bound = atol / scale
    return np.sqrt(np.sum((defect.real/scale)**2 + (defect.imag/scale)**2)) <= bound


def hermitian(h, atol=0., *, rtol=1e-10):
    """Require ||H-H†||_F <= atol + rtol ||H||_F for EACH matrix.

    The default has no implicit energy scale. The Frobenius criterion is
    invariant under unitary basis rotations (up to roundoff), energy-unit
    changes and batch/chunk partitioning. Explicit atol has input units.
    This validates the supplied data; it never projects onto Hermitian data.
    """
    _hermitian_tolerances(atol, rtol)
    h=np.asarray(h, dtype=np.complex128)
    if h.ndim<2 or h.shape[-1]!=h.shape[-2] or h.shape[-1]<1 or not np.isfinite(h).all():
        raise ValueError("finite square Hamiltonian matrices required")
    matrices = np.ascontiguousarray(h).reshape(-1, h.shape[-1], h.shape[-1])
    if rtol == 0:
        with np.errstate(over='ignore', invalid='ignore'):
            valid = all(_absolute_defect_ok(a-a.conj().T, atol) for a in matrices)
    else:
        scales, norms, defects, _ = hermiticity_metrics(matrices)
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            bounds = rtol*norms + atol/np.where(scales == 0., 1., scales)
        valid = not np.any(defects > bounds)
    if not valid:
        raise ValueError("Hamiltonian is not Hermitian; check hopping conjugates")


def hermitian_sparse(h, atol=0., *, rtol=1e-10):
    """Sparse counterpart, O(nnz+n) storage; never densifies or mutates input."""
    from scipy import sparse
    _hermitian_tolerances(atol, rtol)
    if not sparse.issparse(h) or h.ndim != 2 or h.shape[0] != h.shape[1] or not h.shape[0]:
        raise ValueError("finite square sparse Hamiltonian required")
    a = h.tocsr(copy=True).astype(np.complex128, copy=False)
    with np.errstate(over='ignore', invalid='ignore'):
        a.sum_duplicates()
    if not np.isfinite(a.data).all():
        raise ValueError("finite square sparse Hamiltonian required")
    if rtol == 0:
        with np.errstate(over='ignore', invalid='ignore'):
            valid = _absolute_defect_ok((a-a.conj().T).data, atol)
    else:
        scale = max(np.max(abs(a.data.real), initial=0.), np.max(abs(a.data.imag), initial=0.))
        if scale == 0.:
            return
        # Separate components avoid complex reciprocal overflow for tiny scale.
        a.data.real /= scale
        a.data.imag /= scale
        defect = (a-a.conj().T).data
        with np.errstate(over='ignore'):
            bound = rtol*np.linalg.norm(a.data) + atol/scale
        valid = _absolute_defect_ok(defect, bound)
    if not valid:
        raise ValueError("non-Hermitian sparse Hamiltonian")


class TBkit(ABC):
    """Shared interface for HR, HK and Htrig; eigenvectors are columns."""
    @abstractmethod
    def hamiltonian(self, k, *, backend="numba"): ...

    @abstractmethod
    def gradient(self, k, *, backend="numba"): ...

    def bands(self, k, *, eigenvectors=False, backend="numba", chunk_size=2048):
        k, single = points(k, self.dim)
        if chunk_size < 1 or len(k) == 0:
            raise ValueError("positive chunk_size and nonempty k list required")
        energies, vectors = [], []
        for start in range(0, len(k), chunk_size):
            h = self.hamiltonian(k[start:start+chunk_size], backend=backend)
            hermitian(h)
            if eigenvectors:
                e, u = np.linalg.eigh(h)
                vectors.append(u)
            else:
                e = np.linalg.eigvalsh(h)
            energies.append(e)
        e = np.concatenate(energies)
        if eigenvectors:
            u = np.concatenate(vectors)
            return (e[0], u[0]) if single else (e, u)
        return e[0] if single else e

    def __add__(self,other):
        from .algebra import combine
        return combine(self,other)

    def __radd__(self,other):
        # Python sum starts at the additive identity 0; no energy shift implied.
        import numbers
        if isinstance(other,numbers.Number) and other==0:return self
        return NotImplemented

    def __sub__(self,other):
        from .algebra import combine
        return combine(self,other,-1)

    def __mul__(self,value):
        from .algebra import scale
        return scale(self,value)

    def __rmul__(self,value):return self*value

    def __neg__(self):return self*(-1)

    def save(self,path):
        from .io import save_hdf5
        save_hdf5(self,path)

    def sparse_bands(self, k, count, *, sigma=0.0):
        if not hasattr(self, "sparse_hamiltonian"):
            raise TypeError("sparse evaluation is supported by HR")
        if not 0 < count < self.norb-1:
            raise ValueError("count must be between 1 and norb-2")
        h = self.sparse_hamiltonian(k)
        hermitian_sparse(h)
        return np.sort(eigsh(h, k=count, sigma=sigma, return_eigenvectors=False))

    def dos(self, k, energies, *, width=0.02, weights=None, backend="numba"):
        """Gaussian DOS, states per cell per energy; weights sum to one."""
        if width <= 0:
            raise ValueError("width must be positive")
        e = np.atleast_2d(self.bands(k, backend=backend))
        x = np.asarray(energies, dtype=float)
        w = np.ones(len(e))/len(e) if weights is None else np.asarray(weights, float)
        if w.shape != (len(e),) or np.any(w < 0) or not np.isclose(w.sum(), 1):
            raise ValueError("nonnegative normalized k weights required")
        out = np.zeros_like(x)
        for start in range(0, len(e), 256):
            z = (x[..., None, None]-e[start:start+256])/width
            out += (np.exp(-0.5*z*z)*w[start:start+256, None]).sum(axis=(-2,-1))
        return out/(np.sqrt(2*np.pi)*width)

    def fft(self, k, *, backend="numba"):
        """Python equivalent of TBkit fft: (eigenvectors, energies, gradient).

        Gradient axes use Python layout (..., dimension, orbital, orbital).
        """
        e, u = self.bands(k, eigenvectors=True, backend=backend)
        return u, e, self.gradient(k, backend=backend)
