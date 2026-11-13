"""Numba kernels: no Python objects inside the numerical hot path.

Parallelize k points only: every worker owns its output, avoiding reductions
whose order changes with the thread count. BLAS threading is managed separately.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, nogil=True, parallel=True)
def fourier(k, vectors, hopping):
    nk, dim = k.shape
    nr, n, _ = hopping.shape
    out = np.zeros((nk, n, n), dtype=np.complex128)
    for q in prange(nk):
        for r in range(nr):
            angle = 0.0
            for a in range(dim):
                angle += k[q, a] * vectors[r, a]
            phase = complex(np.cos(angle), np.sin(angle))
            for i in range(n):
                for j in range(n):
                    out[q, i, j] += phase * hopping[r, i, j]
    return out


@njit(cache=True, nogil=True, parallel=True)
def fourier_gradient(k, vectors, hopping):
    nk, dim = k.shape
    nr, n, _ = hopping.shape
    out = np.zeros((nk, dim, n, n), dtype=np.complex128)
    for q in prange(nk):
        for r in range(nr):
            angle = 0.0
            for a in range(dim):
                angle += k[q, a] * vectors[r, a]
            phase = complex(-np.sin(angle), np.cos(angle))
            for a in range(dim):
                for i in range(n):
                    for j in range(n):
                        out[q, a, i, j] += vectors[r, a] * phase * hopping[r, i, j]
    return out


@njit(cache=True, nogil=True, parallel=True)
def polynomial(k, powers, coefficients):
    nk, dim = k.shape
    nt, n, _ = coefficients.shape
    out = np.zeros((nk, n, n), dtype=np.complex128)
    for q in prange(nk):
        for t in range(nt):
            v = 1.0
            for a in range(dim):
                v *= k[q, a] ** powers[t, a]
            for i in range(n):
                for j in range(n):
                    out[q, i, j] += v * coefficients[t, i, j]
    return out


@njit(cache=True, nogil=True, parallel=True)
def trig(k, frequencies, cosine, sine):
    nk, dim = k.shape
    nt, n, _ = cosine.shape
    out = np.zeros((nk, n, n), dtype=np.complex128)
    for q in prange(nk):
        for t in range(nt):
            angle = 0.0
            for a in range(dim):
                angle += k[q, a] * frequencies[t, a]
            c, s = np.cos(angle), np.sin(angle)
            for i in range(n):
                for j in range(n):
                    out[q, i, j] += c*cosine[t, i, j] + s*sine[t, i, j]
    return out


@njit(cache=True, nogil=True, parallel=True)
def polynomial_gradient(k, powers, coefficients):
    nk, dim = k.shape
    nt, n, _ = coefficients.shape
    out = np.zeros((nk, dim, n, n), dtype=np.complex128)
    for q in prange(nk):
        for t in range(nt):
            for a in range(dim):
                if powers[t,a] == 0:
                    continue
                v = float(powers[t,a])
                for b in range(dim):
                    v *= k[q,b] ** (powers[t,b] - (1 if a == b else 0))
                for i in range(n):
                    for j in range(n):
                        out[q,a,i,j] += v * coefficients[t,i,j]
    return out


@njit(cache=True, nogil=True, parallel=True)
def trig_gradient(k, frequencies, cosine, sine):
    nk, dim = k.shape
    nt, n, _ = cosine.shape
    out = np.zeros((nk, dim, n, n), dtype=np.complex128)
    for q in prange(nk):
        for t in range(nt):
            angle = 0.0
            for a in range(dim):
                angle += k[q,a]*frequencies[t,a]
            c, s = np.cos(angle), np.sin(angle)
            for a in range(dim):
                for i in range(n):
                    for j in range(n):
                        out[q,a,i,j] += frequencies[t,a]*(c*sine[t,i,j]-s*cosine[t,i,j])
    return out


@njit(cache=True,nogil=True,parallel=True)
def coo_fourier(k,vectors,rindex,rows,cols,values,norb):
    out=np.zeros((len(k),norb,norb),np.complex128)
    for q in prange(len(k)):
        phase=np.empty(len(vectors),np.complex128)
        for r in range(len(vectors)):
            angle=0.
            for a in range(k.shape[1]): angle+=k[q,a]*vectors[r,a]
            phase[r]=complex(np.cos(angle),np.sin(angle))
        for t in range(len(values)):
            out[q,rows[t],cols[t]]+=phase[rindex[t]]*values[t]
    return out


@njit(cache=True,nogil=True,parallel=True)
def coo_gradient(k,vectors,rindex,rows,cols,values,norb):
    out=np.zeros((len(k),k.shape[1],norb,norb),np.complex128)
    for q in prange(len(k)):
        phase=np.empty(len(vectors),np.complex128)
        for r in range(len(vectors)):
            angle=0.
            for a in range(k.shape[1]): angle+=k[q,a]*vectors[r,a]
            phase[r]=complex(-np.sin(angle),np.cos(angle))
        for t in range(len(values)):
            r=rindex[t]
            for a in range(k.shape[1]):
                out[q,a,rows[t],cols[t]]+=vectors[r,a]*phase[r]*values[t]
    return out


@njit(cache=True,nogil=True,parallel=True)
def band_velocities(eigenvectors,gradient):
    """Transform analytic gradients once; callable from user nopython code."""
    nk,n,_=eigenvectors.shape;dim=gradient.shape[1]
    out=np.empty((nk,dim,n,n),np.complex128)
    for q in prange(nk):
        u=np.ascontiguousarray(eigenvectors[q])
        adjoint=np.ascontiguousarray(u.conj().T)
        for a in range(dim):out[q,a]=adjoint@np.ascontiguousarray(gradient[q,a])@u
    return out


@njit(cache=True,nogil=True,parallel=True)
def geometry_from_velocities(energy,velocities,gap_tol):
    """Clean-limit tensors from shared spectra; no repeated basis rotation.

    Public low-level array kernel. It reports near-degenerate k points so the
    validated Python API can apply an explicit raise/omit policy.
    """
    nk,n=energy.shape;dim=velocities.shape[1]
    velocity=np.zeros((nk,dim,n))
    berry=np.zeros((nk,dim,dim,n));metric=np.zeros_like(berry);bcp=np.zeros_like(berry)
    degenerate=np.zeros(nk,np.bool_)
    for q in prange(nk):
        for a in range(dim):
            for i in range(n):velocity[q,a,i]=velocities[q,a,i,i].real
        for i in range(n):
            for j in range(n):
                if i==j:continue
                gap=energy[q,i]-energy[q,j]
                if abs(gap)<=gap_tol:
                    degenerate[q]=True
                    continue
                inv=1/gap
                for a in range(dim):
                    for b in range(dim):
                        product=velocities[q,a,i,j]*velocities[q,b,j,i]
                        berry[q,a,b,i]+=-2*product.imag*inv**2
                        metric[q,a,b,i]+=product.real*inv**2
                        bcp[q,a,b,i]+=2*product.real*inv**3
    return velocity,berry,metric,bcp,degenerate


@njit(cache=True,nogil=True)
def spectral_geometry(energy,eigenvectors,gradient,gap_tol):
    """Backward-compatible nopython composition of the two reusable kernels."""
    return geometry_from_velocities(energy,band_velocities(eigenvectors,gradient),gap_tol)
