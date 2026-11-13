import numpy as np
import pytest
from numpy.testing import assert_allclose as close
from tbkit import models,berry_curvature
from tbkit.responses import geometric_tensors,response_kernel,integrate_bz,fermi_derivative


def test_quantum_metric_bcp_and_berry():
    m=models.dirac(mass=.4);k=np.array([[.1,.2],[.3,.4]])
    data=geometric_tensors(m,k)
    close(data['berry'][:,0,1],berry_curvature(m,k),atol=1e-13)
    close(data['berry'][:,1,0],-data['berry'][:,0,1])
    close(data['metric'][:,0,1],data['metric'][:,1,0])
    e=data['energy'];gap=e[:,0]-e[:,1]
    close(data['bcp'][:,0,0,0],2*data['metric'][:,0,0,0]/gap,atol=1e-13)
    assert np.all(data['metric'][:,0,0]>=0)


def test_fermi_derivatives_finite_difference():
    e=np.linspace(-.1,.1,17);d=1e-5
    close(fermi_derivative(e,temperature=300),(fermi_derivative(e+d,temperature=300,order=0)-fermi_derivative(e-d,temperature=300,order=0))/(2*d),rtol=1e-7)
    close(fermi_derivative(e,temperature=300,order=2),(fermi_derivative(e+d,temperature=300)-fermi_derivative(e-d,temperature=300))/(2*d),atol=1e-5)
    with pytest.raises(ValueError):fermi_derivative(e,temperature=0)


def test_bz_normalizations_and_chern_response():
    one=lambda k:np.ones(len(k))
    lattice=np.diag([2.,3.])
    close(integrate_bz(one,[7,9],lattice=lattice),1)
    close(integrate_bz(one,[7,9],lattice=lattice,normalization='states'),1/6)
    close(integrate_bz(one,[7,9],lattice=lattice,normalization='integral'),(2*np.pi)**2/6)
    m=models.qi_wu_zhang()
    value=integrate_bz(lambda k:response_kernel(m,k,[0],temperature=0),[40,40],chunk_size=131)
    close(value*2*np.pi,[1],atol=1e-6)


def test_response_shapes_and_degeneracy():
    m=models.dirac();k=np.array([[.2,.3],[.4,.7]])
    for kind in ['soahc','ncte']:
        out=response_kernel(m,k,[-.2,0,.2],kind=kind,indices=(0,1,1))
        assert out.shape==(2,3) and np.isfinite(out).all()
        close(out[0],response_kernel(m,k[0],[-.2,0,.2],kind=kind,indices=(0,1,1)))
    with pytest.raises(ValueError):geometric_tensors(models.dirac(0),[0,0])
    close(geometric_tensors(models.dirac(0),[0,0],degeneracy='omit')['berry'],0)


def test_geometry_numba_numpy_parity():
    m=models.topological_insulator()
    # Kramers pairs are degenerate; use an explicit, identical omit policy.
    k=np.array([[.2,.3,.4],[.6,-.4,.1]])
    a=geometric_tensors(m,k,backend='numba',degeneracy='omit')
    b=geometric_tensors(m,k,backend='numpy',degeneracy='omit')
    for key in ('energy','velocity'):close(a[key],b[key],atol=1e-13)
    for key in ('berry','metric','bcp'):
        # Only the trace over each degenerate pair is gauge invariant.
        close(a[key].reshape(2,3,3,2,2).sum(axis=-1),b[key].reshape(2,3,3,2,2).sum(axis=-1),atol=1e-13)


def test_geometry_numba_numpy_nondegenerate():
    m=models.dirac(.4);k=np.array([[.2,.3],[.6,-.4]])
    a=geometric_tensors(m,k,backend='numba')
    b=geometric_tensors(m,k,backend='numpy')
    for key in a:close(a[key],b[key],atol=1e-13)
