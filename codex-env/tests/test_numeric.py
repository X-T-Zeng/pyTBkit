import numpy as np
import scipy.integrate as si
import scipy.linalg as la
import sympy as sp
import numba
import h5py
import matplotlib.pyplot as plt
import tbkernel

def test_two_band_fortran():
    k=np.linspace(-np.pi,np.pi,101)
    got=tbkernel.band(k,0.7)
    expected=np.sqrt(np.sin(k)**2+(0.7+np.cos(k))**2)
    np.testing.assert_allclose(got,expected,rtol=1e-13,atol=1e-14)
    for q,e in zip(k,got):
        h=np.array([[0.7+np.cos(q),np.sin(q)],[np.sin(q),-0.7-np.cos(q)]])
        np.testing.assert_allclose(la.eigvalsh(h),[-e,e],atol=1e-13)

def test_integral_symbolic_numba():
    assert abs(si.quad(lambda x: np.sin(x),0,np.pi)[0]-2)<1e-12
    x=sp.symbols('x')
    assert sp.integrate(sp.sin(x),(x,0,sp.pi))==2
    @numba.njit
    def norm(a):
        return (a*a).sum()
    assert norm(np.arange(5,dtype=np.float64))==30

def test_hdf5_plot_fft(tmp_path):
    x=np.arange(16,dtype=float)
    np.testing.assert_allclose(np.fft.ifft(np.fft.fft(x)).real,x,atol=1e-12)
    with h5py.File(tmp_path/'test.h5','w') as f: f['x']=x
    with h5py.File(tmp_path/'test.h5') as f: np.testing.assert_array_equal(f['x'][:],x)
    plt.plot(x,x*x); plt.savefig(tmp_path/'plot.png'); plt.close()
    assert (tmp_path/'plot.png').stat().st_size>0
