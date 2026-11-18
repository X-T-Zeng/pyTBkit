"""Array-only Hermiticity diagnostics, composable inside Numba nopython.

Inputs are validated finite complex128 arrays with shape (batch, n, n).
No fastmath: conjugation, subnormal entries and exact comparisons matter.
"""
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def hermiticity_metrics(matrices):
    """Return component scales, scaled Frobenius norms of H and H-H†, exactness.

    Normalize real/imaginary components separately: abs(complex) and complex
    division can overflow even when both input components are finite.
    A second scaling prevents squaring tiny defects from underflowing.
    """
    batch, n, _ = matrices.shape
    scales = np.zeros(batch)
    norms = np.zeros(batch)
    defects = np.zeros(batch)
    exact = np.ones(batch, dtype=np.bool_)
    # Reuse two matrix-sized scratch arrays across the batch. Preparing these
    # once per matrix avoids repeated divisions in both defect passes.
    real = np.empty((n, n))
    imag = np.empty((n, n))
    for q in range(batch):
        scale = 0.
        for i in range(n):
            for j in range(n):
                z = matrices[q, i, j]
                scale = max(scale, abs(z.real), abs(z.imag))
                other = matrices[q, j, i]
                if z.real != other.real or z.imag != -other.imag:
                    exact[q] = False
        scales[q] = scale
        if scale == 0.:
            continue
        norm2 = 0.
        reciprocal_safe = scale >= np.finfo(np.float64).tiny
        inverse = 1. / scale if reciprocal_safe else 0.
        for i in range(n):
            for j in range(n):
                z = matrices[q, i, j]
                if reciprocal_safe:
                    re, im = z.real * inverse, z.imag * inverse
                else:
                    re, im = z.real / scale, z.imag / scale
                real[i, j], imag[i, j] = re, im
                norm2 += re * re + im * im
        norms[q] = np.sqrt(norm2)
        if exact[q]:
            continue
        dscale = 0.
        for i in range(n):
            for j in range(n):
                dr = real[i, j] - real[j, i]
                di = imag[i, j] + imag[j, i]
                dscale = max(dscale, abs(dr), abs(di))
        if dscale == 0.:
            continue
        defect2 = 0.
        reciprocal_safe = dscale >= np.finfo(np.float64).tiny
        inverse = 1. / dscale if reciprocal_safe else 0.
        for i in range(n):
            for j in range(n):
                dr = real[i, j] - real[j, i]
                di = imag[i, j] + imag[j, i]
                if reciprocal_safe:
                    dr, di = dr * inverse, di * inverse
                else:
                    dr, di = dr / dscale, di / dscale
                defect2 += dr * dr + di * di
        defects[q] = dscale * np.sqrt(defect2)
    return scales, norms, defects, exact
