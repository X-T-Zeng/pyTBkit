"""Array-only finite-group kernels, callable from a user's Numba pipeline."""
import numpy as np
from numba import njit


@njit(cache=True,nogil=True)
def associative(table):
    """Check (a*b)*c=a*(b*c); input is a valid-range square integer table."""
    for a in range(len(table)):
        for b in range(len(table)):
            for c in range(len(table)):
                if table[table[a,b],c]!=table[a,table[b,c]]:
                    return False
    return True


@njit(cache=True,nogil=True)
def character_projector(matrices,characters,degree):
    """d/|G| sum_g conj(chi(g))*D(g), for a validated ordinary irrep chi.

    matrices: (order,n,n) complex128; characters: (order,) complex128.
    Validation/antiunitary and projective exclusions belong to the object API.
    """
    n=matrices.shape[1]
    out=np.zeros((n,n),dtype=np.complex128)
    for g in range(len(matrices)):
        weight=degree*characters[g].conjugate()/len(matrices)
        for i in range(n):
            for j in range(n):
                out[i,j]+=weight*matrices[g,i,j]
    return out


@njit(cache=True,nogil=True)
def regular_blocks(table,basis):
    """Q† L(g) Q without materializing all left-regular matrices.

    table[g,h] indexes g*h, so L(g)|h>=|g*h>. basis is a validated complex128
    (order,dimension) orthonormal invariant column space. Raw arrays are trusted;
    the object API checks unitarity and the complete representation law.
    """
    order,dimension=basis.shape
    out=np.zeros((order,dimension,dimension),dtype=np.complex128)
    for g in range(order):
        for i in range(dimension):
            for j in range(dimension):
                value=0.0j
                for h in range(order):
                    value+=basis[table[g,h],i].conjugate()*basis[h,j]
                out[g,i,j]=value
    return out
