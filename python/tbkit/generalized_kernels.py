"""Array-only diagnostics for positive-Gram generalized Hermitian spectra.

Inputs H and S have separately normalized finite components, C† S C = I,
and E solves H C = S C E in those normalized units. No fastmath or repair.
The caller validates shapes/domain; kernels are composable in nopython.
"""
import numpy as np
from numba import njit


@njit(cache=True, inline='always')
def _abs2(z):
    return z.real*z.real+z.imag*z.imag


@njit(cache=True, nogil=True)
def generalized_diagnostics(hamiltonian, overlap, energy, vectors):
    """Return per-band normwise residuals and Frobenius Gram-frame errors."""
    nk, n = energy.shape
    residual = np.zeros((nk, n))
    metric_error = np.zeros(nk)
    hc = np.empty((n,n),np.complex128)
    sc = np.empty((n,n),np.complex128)
    for q in range(nk):
        # Tiny orbital blocks benefit from fused loops instead of three BLAS
        # dispatches per k point. Larger blocks retain optimized matrix products.
        if n<=8:
            for i in range(n):
                for j in range(n):
                    hv,sv=0.j,0.j
                    for t in range(n):
                        hv+=hamiltonian[q,i,t]*vectors[q,t,j]
                        sv+=overlap[q,i,t]*vectors[q,t,j]
                    hc[i,j],sc[i,j]=hv,sv
        else:
            hc = hamiltonian[q] @ vectors[q]
            sc = overlap[q] @ vectors[q]
        gram = vectors[q].conj().T @ sc if n>8 else np.empty((0,0),np.complex128)
        hn2, sn2, error2 = 0., 0., 0.
        for i in range(n):
            for j in range(n):
                hn2 += _abs2(hamiltonian[q, i, j])
                sn2 += _abs2(overlap[q, i, j])
                if n<=8:
                    value=0.j
                    for t in range(n):value+=vectors[q,t,i].conjugate()*sc[t,j]
                else:
                    value=gram[i,j]
                error2 += _abs2(value-(1. if i == j else 0.))
        hn, sn = np.sqrt(hn2), np.sqrt(sn2)
        metric_error[q] = np.sqrt(error2)
        for band in range(n):
            rn2, cn2 = 0., 0.
            for i in range(n):
                rn2 += _abs2(hc[i, band]-energy[q, band]*sc[i, band])
                cn2 += _abs2(vectors[q, i, band])
            denominator = (hn+abs(energy[q, band])*sn)*np.sqrt(cn2)
            if denominator == 0.:
                residual[q, band] = 0. if rn2 == 0. else np.inf
            else:
                residual[q, band] = np.sqrt(rn2)/denominator
    return residual, metric_error
