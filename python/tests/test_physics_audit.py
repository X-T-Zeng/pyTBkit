"""Regressions derived from operator theory, perturbation theory and gauge laws."""
import numpy as np
import pytest
from numpy.testing import assert_allclose as close
from tbkit import HR,HK,models,berry_curvature,wilson_loop
from tbkit.base import hermitian,points,TBkit
from tbkit.responses import geometric_tensors
from tbkit.nonorthogonal import NonOrthogonal
from tbkit.topology import wilson_transport


def test_nonhermitian_derivative_at_accidentally_hermitian_point():
    # H(0) is Hermitian, but dH/dkx=i*sz is not. Single-k spectral checks
    # cannot certify that this one-way hopping defines a Hermitian family.
    model=HR([[1,0]],[[[1,0],[0,-1]]])
    close(model.bands([0,0]),[-1,1])
    for function in (geometric_tensors,berry_curvature):
        with pytest.raises(ValueError,match='Hermitian'):function(model,[0,0])


def test_finite_observables_and_real_physical_momenta():
    with pytest.raises(ValueError,match='finite'):hermitian([[np.inf]])
    with pytest.raises(ValueError,match='real'):points(np.array([1+2j]),1)
    with pytest.raises(ValueError):berry_curvature(models.dirac(),[.2,.3],gap_tol=0)


def test_degenerate_generalized_band_slope_is_not_arbitrary_diagonal():
    # At k=0, H=k*sx has directional slopes +/-1, while the arbitrary I
    # eigenframe gives diagonal <sx>=0. Reject single-band HF at degeneracy.
    h=HK({(1,):models.sigma_x});s=HK({(0,):np.eye(2)})
    generalized=NonOrthogonal(h,s)
    with pytest.raises(ValueError,match='degeneracy'):generalized.band_gradient([0])
    close(generalized.band_gradient([.2]),[[-1,1]],atol=1e-14)


def test_closed_wilson_gauge_invariance_and_open_endpoint_covariance():
    model=models.ssh();k=np.linspace(-np.pi,np.pi,41,endpoint=False)[:,None]
    phases=np.random.default_rng(17).normal(size=(len(k),2))
    class Gauge(TBkit):
        dim=1;norb=2
        def hamiltonian(self,k,*,backend='numba'):return model.hamiltonian(k,backend=backend)
        def gradient(self,k,*,backend='numba'):return model.gradient(k,backend=backend)
        def bands(self,k,*,eigenvectors=False,backend='numba',**kwargs):
            e,u=model.bands(k,eigenvectors=True,backend=backend)
            u=u*np.exp(1j*phases)[:,None,:]
            return (e,u) if eigenvectors else e
    gauge=Gauge()
    close(np.exp(1j*wilson_loop(gauge,k)),np.exp(1j*wilson_loop(model,k)),atol=1e-13)
    original=wilson_transport(model,k);changed=wilson_transport(gauge,k)
    close(changed,np.exp(-1j*phases[0,0])*original*np.exp(1j*phases[-1,0]),atol=1e-13)
    assert abs(np.angle(changed[0,0]/original[0,0]))>.1
    with pytest.raises(ValueError,match='open paths'):wilson_loop(model,k,closed=False)


def test_bz_invariants_reject_nonperiodic_continuum():
    from tbkit import chern_number
    from tbkit.topology import wilson_flow
    for function in (chern_number,wilson_flow):
        with pytest.raises(ValueError,match='periodic'):function(models.dirac(),(11,11))


def test_trig_conversion_preserves_nonunit_cell_topology_and_centers(tmp_path):
    from tbkit import chern_number,load_hdf5
    base=models.qi_wu_zhang()
    lattice=np.array([[1.7,.2],[0,2.3]])
    orbital=np.array([[.1,.2],[.3,.4]])
    m=HR(base.vectors,base.hopping,lattice=lattice,orbitals=orbital)
    t=m.to_htrig();scaled=2*t
    close(t.lattice,lattice);close(t.orbitals,orbital)
    close(scaled.lattice,lattice);close(t.to_hr().orbitals,orbital)
    close(chern_number(t,(17,17)),chern_number(m,(17,17)),atol=1e-12)
    t.save(tmp_path/'trig.h5');loaded=load_hdf5(tmp_path/'trig.h5')
    close(loaded.lattice,lattice);close(loaded.orbitals,orbital)
    close(chern_number(loaded,(17,17)),1,atol=1e-12)


def test_python_sum_of_generated_terms():
    from tbkit import invariant_kp
    basis=invariant_kp(2,1,1,[])
    combination=sum((i+1)*h for i,h in enumerate(basis))
    expected=sum((i+1)*h.hamiltonian([.13],backend='numpy') for i,h in enumerate(basis))
    close(combination.hamiltonian([.13]),expected,atol=1e-13)


def test_point_orbital_embedding_local_curvature_and_boundary_sewing():
    from tbkit import embed_orbitals,chern_number
    base=models.qi_wu_zhang();a=.37
    model=HR(base.vectors,base.hopping,orbitals=[[0,0],[a,0]])
    embedded=embed_orbitals(model);k=np.array([[.31,.27],[-.4,.67]])
    # Explicit analytic derivative of the point-orbital frame, checked by FD.
    for axis in range(2):
        step=np.eye(2)[axis]*1e-5
        close(embedded.gradient(k)[:,axis],(embedded.hamiltonian(k+step)-embedded.hamiltonian(k-step))/2e-5,atol=1e-9)
    # A_emb=A_periodic+sum |u_j|^2 tau_j. For lower QWZ band p_1=(1+dz/E)/2.
    x,y=k.T;z=-1+np.cos(x)+np.cos(y)
    radius=np.sqrt(np.sin(x)**2+np.sin(y)**2+z*z)
    dz=-np.sin(y);dr=(np.sin(y)*np.cos(y)+z*dz)/radius
    dp=.5*(dz/radius-z*dr/radius**2)
    delta=berry_curvature(embedded,k)[:,0]-berry_curvature(model,k)[:,0]
    close(delta,-a*dp,atol=1e-12)
    # An unsewn torus calculation must not silently identify different frames.
    with pytest.raises(ValueError,match='periodic'):chern_number(embedded,(11,11))
    g=np.array([2*np.pi,0]);s=embedded.boundary_sewing(g)
    close(embedded.hamiltonian(k+g),s@embedded.hamiltonian(k)@s.conj().T,atol=1e-13)


def test_embedded_ssh_wilson_includes_orbital_position():
    from tbkit import embed_orbitals
    base=models.ssh();m=HR(base.vectors,base.hopping,orbitals=[[0],[.4]])
    embedded=embed_orbitals(m);k=2*np.pi*np.arange(101)[:,None]/101
    with pytest.raises(ValueError,match='boundary_sewing'):wilson_loop(embedded,k)
    phase=wilson_loop(embedded,k,boundary_sewing=embedded.boundary_sewing([2*np.pi]))[0]
    reference=wilson_loop(m,k)[0]+.4*np.pi
    close(np.exp(1j*phase),np.exp(1j*reference),atol=1e-12)


def test_fermi_derivative_even_odd_tails_and_invalid_temperature():
    from tbkit import fermi
    from tbkit.responses import fermi_derivative
    from tbkit.constants import KB_EV_PER_K
    temperature=300.;energy=np.array([-100.,-40.,0.,40.,100.])*KB_EV_PER_K*temperature
    first=fermi_derivative(energy,temperature=temperature)
    second=fermi_derivative(energy,temperature=temperature,order=2)
    close(first,first[::-1],rtol=1e-14,atol=0)
    close(second,-second[::-1],rtol=1e-14,atol=0)
    assert first[0]<0 and first[-1]<0 and second[0]<0 and second[-1]>0
    for bad in (np.nan,np.inf,-1):
        with pytest.raises(ValueError):fermi([0],temperature=bad)
        with pytest.raises(ValueError):fermi_derivative([0],temperature=bad)


def test_near_integer_fourier_vectors_round_instead_of_truncate():
    from tbkit import Htrig
    almost=np.nextafter(1.,0.)
    # r=0.9999999999999999 is the R=1 cell up to roundoff, not R=0.
    h=HR([[almost],[-almost]],[[[1]],[[1]]])
    close(h.bands([.3]),[2*np.cos(.3)],atol=1e-14)
    edited=HR([[0]],[[[0]]]).with_hopping([almost],[[1]],conjugate=True)
    close(edited.bands([.3]),[2*np.cos(.3)],atol=1e-14)
    trig=Htrig([[almost]],[[[2]]])
    close(trig.to_hr().bands([.3]),[2*np.cos(.3)],atol=1e-14)
    with pytest.raises(ValueError):HR([[1.000001]],[[[1]]])
    with pytest.raises(ValueError):Htrig([[1.000001]],[[[2]]]).to_hr()
