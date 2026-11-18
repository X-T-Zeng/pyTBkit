"""Array-only angular generators for a validated doubled quantum number."""
import numpy as np
from numba import njit


@njit(cache=True,nogil=True)
def angular_momentum_doubled(twice_j,max_dimension=1025):
    """Return Jx,Jy,Jz, m=j..-j; twice_j is a nonnegative integer.

    sqrt(k*(2*j-k+1)) avoids subtracting two nearly equal j**2 terms.
    The object API validates discrete labels; nopython callers supply integers.
    """
    if twice_j<0 or twice_j>=max_dimension or max_dimension<1:
        raise ValueError('angular momentum dimension exceeds bounds')
    dimension=twice_j+1
    x=np.zeros((dimension,dimension),dtype=np.complex128)
    y=np.zeros((dimension,dimension),dtype=np.complex128)
    z=np.zeros((dimension,dimension),dtype=np.complex128)
    for row in range(dimension):z[row,row]=twice_j/2-row
    for column in range(1,dimension):
        value=np.sqrt(float(column)*float(twice_j-column+1))/2
        x[column-1,column]=value;x[column,column-1]=value
        y[column-1,column]=-1j*value;y[column,column-1]=1j*value
    return x,y,z
