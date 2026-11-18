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
def matrix_unit_column(matrices,irrep):
    """E_i0=(d/|G|) sum_g conj(irrep[g,i,0])*matrices[g], trusted arrays."""
    order,dimension,_=matrices.shape;degree=irrep.shape[1]
    weights=np.empty((degree,order),dtype=np.complex128)
    for i in range(degree):
        for g in range(order):weights[i,g]=degree/order*irrep[g,i,0].conjugate()
    out=np.empty((degree,dimension,dimension),dtype=np.complex128)
    for i in range(degree):
        for a in range(dimension):
            for b in range(dimension):
                value=0.0j
                for g in range(order):value+=weights[i,g]*matrices[g,a,b]
                out[i,a,b]=value
    return out


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
def generated_subgroup_indices(table,generators,identity):
    """Sorted closure indices, for a validated finite group and in-range gens.

    A finite monoid generated inside a group contains the inverses. Callers may
    add inverse generators to reduce path lengths. No model state is cached.
    """
    n=len(table)
    seen=np.zeros(n,dtype=np.bool_)
    pending=np.empty(n,dtype=np.int64)
    seen[identity]=True;pending[0]=identity;count=1;cursor=0
    while cursor<count:
        a=pending[cursor];cursor+=1
        for b in generators:
            c=table[a,b]
            if not seen[c]:
                seen[c]=True;pending[count]=c;count+=1
    result=np.empty(count,dtype=np.int64);cursor=0
    for i in range(n):
        if seen[i]:result[cursor]=i;cursor+=1
    return result


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
