"""Independent point-position commutators, holonomies and frame contracts."""
import numpy as np
import pytest
import sympy as s
from numba import njit
from numpy.testing import assert_allclose as close
from tbkit import HR,models,embed_orbitals,OrbitalEmbedding,wilson_loop,prepare_geometry,prepare_spectral
from tbkit.base import TBkit
from tbkit.gauge_kernels import embed_hamiltonian,embed_gradient


def ssh(a=.4):
    model=models.ssh()
    return HR(model.vectors,model.hopping,orbitals=[[0],[a]])


@pytest.mark.parametrize('backend',('numpy','numba','fortran'))
def test_constant_dimer_exact_symbolic_derivative_and_commutator(backend):
    # h_12(k)=t exp(ik*a), so d h_12/dk=i*a*t exp(ik*a).
    # This follows independently from i[H,r], hbar=1.
    kx,a,t=s.symbols('k a t',real=True)
    exact=s.Matrix([[0,t*s.exp(s.I*kx*a)],[t*s.exp(-s.I*kx*a),0]])
    analytic=s.lambdify((kx,a,t),exact.diff(kx),'numpy')
    model=HR([[0]],[[[0,1.3],[1.3,0]]],lattice=[[2.]],orbitals=[[.1],[.3]])
    embedded=embed_orbitals(model);k=np.array([[-.23],[.51],[1.4]])
    gradient=embedded.gradient(k,backend=backend)[:,0]
    close(gradient,np.array([analytic(x,.4,1.3) for x in k[:,0]]),rtol=2e-14,atol=2e-16)
    h=model.hamiltonian(k,backend=backend);position=np.diag([.2,.6])
    commutator=1j*(h@position-position@h)
    d=np.exp(1j*k@np.array([[.2,.6]]))
    close(gradient,d.conj()[:,:,None]*commutator*d[:,None,:],rtol=2e-14,atol=2e-16)
    close(embedded.gradient(k[0],backend=backend),gradient[0,None],rtol=2e-14,atol=2e-16)


@pytest.mark.parametrize('operation',('scale','negative','sum','difference','nested'))
def test_algebra_preserves_ssh_holonomy_and_sewing(operation):
    base=ssh();e=embed_orbitals(base)
    candidates={'scale':2*e,'negative':-e,'sum':e+2*e,'difference':3*e-e,'nested':2*(e+e)}
    model=candidates[operation];k=(2*np.pi*np.arange(151)/151)[:,None]
    assert model.orbital_frame=='point-orbital' and model.requires_boundary_sewing
    assert embed_orbitals(model) is model
    sewing=model.boundary_sewing([2*np.pi]);close(sewing,np.diag([1,np.exp(-.8j*np.pi)]),atol=2e-15)
    with pytest.raises(ValueError,match='boundary_sewing'):wilson_loop(model,k)
    phase=wilson_loop(model,k,boundary_sewing=sewing)[0]
    # Chiral SSH has equal site weights in either band, Zak=pi+pi*a.
    close(np.exp(1j*phase),np.exp(1.4j*np.pi),atol=2e-13)
    probe=np.array([[.13],[.27]])
    close(model.hamiltonian(probe+2*np.pi),sewing@model.hamiltonian(probe)@sewing.conj().T,atol=2e-14)


def test_point_geometry_is_applied_once_after_algebra():
    q=models.qi_wu_zhang();q=HR(q.vectors,q.hopping,orbitals=[[0,0],[.37,0]])
    e=embed_orbitals(q);k=np.array([[.31,.27],[-.4,.67]])
    assert embed_orbitals(e) is e
    with pytest.raises(ValueError,match='already'):OrbitalEmbedding(e)
    for model in (e,2*e,e+e,3*e-e,2*(e+e)):
        fixed=prepare_geometry(model,k,backend='numpy')
        requested=prepare_geometry(model,k,backend='numpy',frame='point-orbital')
        assert fixed.frame==requested.frame=='point-orbital'
        close(requested.berry,fixed.berry,rtol=0,atol=0)
        close(requested.metric,fixed.metric,rtol=0,atol=0)
        close(fixed.berry,prepare_geometry(e,k,backend='numpy').berry,rtol=2e-14,atol=2e-15)
    # Exact QWZ occupation derivative: Delta Omega_xy=-a*d_y p_2.
    x,y=k.T;z=-1+np.cos(x)+np.cos(y);r=np.sqrt(np.sin(x)**2+np.sin(y)**2+z*z)
    dz=-np.sin(y);dr=(np.sin(y)*np.cos(y)+z*dz)/r
    correction=-.37*.5*(dz/r-z*dr/r**2)
    close(fixed.berry[:,0,1,0]-prepare_geometry(q,k,backend='numpy').berry[:,0,1,0],correction,atol=2e-14)
    assert prepare_spectral(2*e,k).frame=='point-orbital'


@pytest.mark.parametrize('centers',([[0],[.40000001]],[[100000.],[100001.]]))
def test_distinct_fractional_centers_cannot_be_merged(centers):
    a=ssh();b=HR(a.vectors,a.hopping,orbitals=centers)
    for x,y in ((a,b),(embed_orbitals(a),embed_orbitals(b))):
        with pytest.raises(ValueError,match='centers'):x+y
    # A lattice-vector shift is a k-dependent gauge change, not center equality.
    shifted=HR(a.vectors,a.hopping,orbitals=[[0],[1.4]])
    with pytest.raises(ValueError,match='centers'):a+shifted


def test_mixed_frames_rejected_and_common_unwrapped_centers_preserved():
    a=ssh()
    for operation in (lambda:a+embed_orbitals(a),lambda:embed_orbitals(a)-a):
        with pytest.raises(ValueError,match='frames'):operation()
    shifted=HR(a.vectors,a.hopping,orbitals=[[12.1],[-3.4]])
    e=embed_orbitals(shifted);combined=e+2*e
    close(combined.orbitals,shifted.orbitals,rtol=0,atol=0)
    close(combined.boundary_sewing([2*np.pi]),e.boundary_sewing([2*np.pi]),rtol=0,atol=0)


@pytest.mark.parametrize('shift',([2*np.pi*1000000.25],[np.nan],[np.inf],[1j],[np.pi]))
def test_sewing_requires_finite_real_reciprocal_shift_without_relative_integer_slack(shift):
    with pytest.raises(ValueError):embed_orbitals(ssh()).boundary_sewing(shift)


def test_reciprocal_sewing_composes_and_obeys_dual_lattice():
    q=models.qi_wu_zhang();a=np.array([[1.7,.2],[.1,2.3]])
    e=embed_orbitals(HR(q.vectors,q.hopping,lattice=a,orbitals=[[.13,.2],[.37,.41]]))
    b=2*np.pi*np.linalg.inv(a).T;g1,g2=b[0],-2*b[1]
    close(e.boundary_sewing(g1+g2),e.boundary_sewing(g1)@e.boundary_sewing(g2),atol=3e-15)
    close(e.boundary_sewing(np.zeros(2)),np.eye(2),rtol=0,atol=0)
    k=np.array([[.13,-.22],[.37,.11]])
    sewing=e.boundary_sewing(g1+g2)
    close(e.hamiltonian(k+g1+g2),sewing@e.hamiltonian(k)@sewing.conj().T,atol=5e-14)


@pytest.mark.parametrize('centers',([[np.nan],[0]],[[np.inf],[0]],[[1j],[0]]))
def test_invalid_center_metadata_rejected_without_silent_complex_cast(centers):
    # Use a protocol fixture because legacy HR casts metadata before embedding.
    model=ssh();model.orbitals=np.array(centers)
    with pytest.raises(ValueError,match='real|finite'):embed_orbitals(model)
    with pytest.raises(ValueError,match='real|finite'):model+ssh()


def test_periodic_topology_rejects_embedded_frame_even_when_h_is_constant():
    from tbkit import chern_number
    from tbkit.topology import wilson_flow,nested_polarization
    atomic=HR([[0,0]],[np.diag([-1.,-1.,1.,1.])],orbitals=[[.1,0],[.2,0],[.3,0],[.4,0]])
    for model in (embed_orbitals(atomic),2*embed_orbitals(atomic)):
        for f in (chern_number,wilson_flow,nested_polarization):
            with pytest.raises(ValueError,match='boundary sewing'):f(model,(7,7),occupied=2)


def test_wilson_sewing_unitarity_has_no_relative_default_tolerance():
    model=embed_orbitals(ssh());k=(2*np.pi*np.arange(31)/31)[:,None]
    for sewing in (np.diag([1.,1.000001]),np.full((2,2),np.nan)):
        with pytest.raises(ValueError,match='unitary'):wilson_loop(model,k,boundary_sewing=sewing)


def test_degenerate_occupied_subspace_random_gauge_preserves_holonomy_spectrum():
    from scipy.linalg import block_diag
    h=block_diag(models.sigma_x,models.sigma_x)
    e=embed_orbitals(HR([[0]],[h],orbitals=[[0],[.2],[0],[.6]]))
    k=(2*np.pi*np.arange(73)/73)[:,None];rng=np.random.default_rng(17)
    rotations=np.array([np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))[0] for _ in k])
    class Rotated(TBkit):
        dim=1;norb=4;requires_boundary_sewing=True;orbital_frame='point-orbital'
        def hamiltonian(self,k,*,backend='numba'):return e.hamiltonian(k,backend=backend)
        def gradient(self,k,*,backend='numba'):return e.gradient(k,backend=backend)
        def bands(self,k,*,eigenvectors=False,backend='numba',**kwargs):
            energy,u=e.bands(k,eigenvectors=True,backend=backend)
            u[:,:,:2]=u[:,:,:2]@rotations
            return (energy,u) if eigenvectors else energy
    reference=np.pi*np.array([.2,.6]);sewing=e.boundary_sewing([2*np.pi])
    close(wilson_loop(e,k,occupied=2,boundary_sewing=sewing),reference,atol=2e-13)
    close(wilson_loop(Rotated(),k,occupied=2,boundary_sewing=sewing),reference,atol=2e-13)


def test_new_array_kernels_are_nopython_composable_and_do_not_mutate():
    @njit
    def pipeline(k,t,h,dh):return embed_hamiltonian(k,t,h),embed_gradient(k,t,h,dh)
    model=ssh();k=np.array([[.2],[.3]]);h=model.hamiltonian(k);dh=model.gradient(k)
    before=[a.copy() for a in (k,h,dh)];centers=np.array([[0.],[.4]])
    actual=pipeline(k,centers,h,dh);e=embed_orbitals(model)
    close(actual[0],e.hamiltonian(k,backend='numpy'),atol=2e-15)
    close(actual[1],e.gradient(k,backend='numpy'),atol=2e-15)
    for a,b in zip((k,h,dh),before):close(a,b,rtol=0,atol=0)
    assert pipeline.nopython_signatures
