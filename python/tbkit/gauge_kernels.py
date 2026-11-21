"""Point-orbital dressing of validated batched arrays, callable from nopython.

k (nk,dim), centers (n,dim), h (nk,n,n), dh (nk,dim,n,n).
No model objects, caches, hidden coordinate changes or Hermitian projection.
Each k point is owned by one worker; inputs are not modified.
"""
import numpy as np
from numba import njit,prange


@njit(cache=True,nogil=True,parallel=True)
def embed_hamiltonian(k,centers,h):
    nk,n,_=h.shape
    out=np.empty((nk,n,n),np.complex128)
    for q in prange(nk):
        phase=np.empty(n,np.complex128)
        for i in range(n):
            angle=0.
            for a in range(k.shape[1]):angle+=k[q,a]*centers[i,a]
            phase[i]=complex(np.cos(angle),np.sin(angle))
        for i in range(n):
            for j in range(n):out[q,i,j]=phase[i].conjugate()*h[q,i,j]*phase[j]
    return out


@njit(cache=True,nogil=True,parallel=True)
def embed_gradient(k,centers,h,dh):
    nk,dim,n,_=dh.shape
    out=np.empty((nk,dim,n,n),np.complex128)
    for q in prange(nk):
        phase=np.empty(n,np.complex128)
        for i in range(n):
            angle=0.
            for a in range(dim):angle+=k[q,a]*centers[i,a]
            phase[i]=complex(np.cos(angle),np.sin(angle))
        for a in range(dim):
            for i in range(n):
                for j in range(n):
                    corrected=dh[q,a,i,j]+1j*(centers[j,a]-centers[i,a])*h[q,i,j]
                    out[q,a,i,j]=phase[i].conjugate()*corrected*phase[j]
    return out
