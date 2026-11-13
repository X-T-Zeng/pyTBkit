import numpy as np
from numpy.testing import assert_allclose as close
from tbkit import HK,models
from tbkit.continuum import landau_hamiltonian,plane_wave_hamiltonian


def test_landau_scalar_parabola():
    h=HK({(2,0):np.ones((1,1))/2,(0,2):np.ones((1,1))/2})
    result=landau_hamiltonian(h,20,magnetic_length=2)
    # Inspect interior oscillator basis; truncated top state is not physical.
    close(np.diag(result)[:-1].real,(np.arange(19)+.5)/4,atol=1e-14)
    close(result,np.diag(np.diag(result)),atol=1e-14)


def test_landau_dirac_and_weyl_mixed():
    m=models.dirac(.3)
    e=np.linalg.eigvalsh(landau_hamiltonian(m,14))
    expected=np.sort(np.r_[-.3,.3,*[s*np.sqrt(.3**2+2*n) for n in range(1,14) for s in [-1,1]]])
    close(e,expected,atol=1e-13)
    # Weyl ordering of kx*ky produces a Hermitian matrix.
    h=HK({(1,1):np.ones((1,1))})
    a=landau_hamiltonian(h,10)
    close(a,a.conj().T,atol=1e-14)


def test_plane_wave_free_and_potential():
    h=HK({(2,):np.ones((1,1))})
    basis={(-1,):[-2*np.pi],(0,):[0],(1,):[2*np.pi]}
    free=plane_wave_hamiltonian(h,[.2],basis,{})
    close(np.diag(free),[(-2*np.pi+.2)**2,.2**2,(2*np.pi+.2)**2])
    coupled=plane_wave_hamiltonian(h,[.2],basis,{(1,):[[.1]],(-1,):[[.1]]})
    close(coupled-free,np.diag([.1,.1],1)+np.diag([.1,.1],-1))
