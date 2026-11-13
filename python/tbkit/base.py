"""Common spectral API. Energies have the units of the input Hamiltonian."""
from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse.linalg import eigsh


def points(k, dim):
    if np.iscomplexobj(k):raise ValueError("physical k points must be real; complex-band continuation is a separate API")
    k = np.asarray(k, dtype=float)
    single = k.ndim == 1
    k = np.atleast_2d(k)
    if k.ndim != 2 or k.shape[1] != dim or not np.isfinite(k).all():
        raise ValueError(f"k must have shape ({dim},) or (nk,{dim}) and be finite")
    return np.ascontiguousarray(k), single


def hermitian(h, atol=1e-10):
    h=np.asarray(h)
    if h.ndim<2 or h.shape[-1]!=h.shape[-2] or h.shape[-1]<1 or not np.isfinite(h).all():
        raise ValueError("finite square Hamiltonian matrices required")
    if not np.allclose(h, h.conj().swapaxes(-1, -2), atol=atol, rtol=1e-10):
        raise ValueError("Hamiltonian is not Hermitian; check hopping conjugates")


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
        if np.max(np.abs((h-h.getH()).data), initial=0) > 1e-10:
            raise ValueError("non-Hermitian sparse Hamiltonian")
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
