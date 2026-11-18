"""Independent Gram/basis/Hellmann-Feynman identities and numerical gates."""
import numpy as np
import pytest
from numba import njit
from numpy.testing import assert_allclose as close
from tbkit import HK, NonOrthogonal
from tbkit.generalized_kernels import generalized_diagnostics
from tbkit.nonorthogonal import GeneralizedSpectrum, generalized_diagnostics_numpy


SX=np.array([[0.,1.],[1.,0.]])
SZ=np.diag([1.,-1.])


@pytest.mark.parametrize('energy_scale',(1e-100,1.,1e100))
@pytest.mark.parametrize('basis_scale',(1e-150,1.,1e150))
def test_congruence_spectrum_frame_and_gradient(energy_scale,basis_scale):
    b=np.array([[1.,.3+.2j],[-.1j,1.2]])
    h=HK({(0,):basis_scale*energy_scale*(b.conj().T@(.4*SZ)@b),
          (1,):basis_scale*energy_scale*(b.conj().T@SX@b)})
    s=HK({(0,):basis_scale*(b.conj().T@b)})
    model=NonOrthogonal(h,s);k=np.array([[.2],[.7]])
    result=model.spectrum(k)
    radius=np.sqrt(k[:,0]**2+.16)
    close(result.energy/energy_scale,radius[:,None]*[-1,1],rtol=2e-14,atol=0)
    close(model.band_gradient(k)/energy_scale,(k[:,0]/radius)[:,None,None]*[-1,1],rtol=3e-14,atol=0)
    # Map back to the physical orthonormal basis. Compare projectors, not phases.
    physical=(np.sqrt(basis_scale)*b)@result.eigenvectors
    for q,x in enumerate(k[:,0]):
        for band,sign in enumerate((-1,1)):
            v=physical[q,:,band]
            close(v[:,None]@v.conj()[None,:],(np.eye(2)+sign*(.4*SZ+x*SX)/radius[q])/2,atol=2e-14)
    assert np.max(result.residual)<1e-14
    assert np.max(result.metric_error)<1e-14
    assert np.max(result.overlap_condition)<10
    # NumPy model+diagnostic path is independently exercised, not just JIT twice.
    close(model.bands(k,backend='numpy')/energy_scale,result.energy/energy_scale,rtol=2e-14,atol=0)


@pytest.mark.parametrize('basis_scale',(1e-250,1.,1e250))
def test_k_dependent_gram_terms_cancel_in_band_derivative(basis_scale):
    # B(k)=diag(1+a*k,1), H0=diag(k,2-k); no Berry/velocity assumption.
    a=.3
    h=HK({(0,):basis_scale*np.diag([0.,2.]),(1,):basis_scale*np.diag([1.,-1.]),
          (2,):basis_scale*np.diag([2*a,0.]),(3,):basis_scale*np.diag([a*a,0.])})
    s=HK({(0,):basis_scale*np.eye(2),(1,):basis_scale*np.diag([2*a,0.]),
          (2,):basis_scale*np.diag([a*a,0.])})
    m=NonOrthogonal(h,s);k=np.array([[-.4],[.2],[.6]])
    close(m.bands(k),np.column_stack((k[:,0],2-k[:,0])),atol=2e-14)
    close(m.band_gradient(k),np.broadcast_to([1.,-1.],(len(k),1,2)),atol=2e-14)


@pytest.mark.parametrize('basis_scale',(1e-200,1.,1e200))
def test_ill_conditioned_gram_is_not_silently_pruned(basis_scale):
    eps=1e-10
    m=NonOrthogonal(HK({(0,):basis_scale*np.diag([1.,3*eps])}),
                    HK({(0,):basis_scale*np.diag([1.,eps])}))
    with pytest.raises(ValueError,match='ill-conditioned overlap'):
        m.bands([0.])
    # Explicitly relax the budget. A diagonal fixture is exactly resolvable;
    # this does not prove comparable accuracy for a rounded dense congruence.
    r=m.spectrum([0.],condition_limit=1e12)
    close(r.energy,[[1.,3.]],rtol=1e-14)
    close(r.overlap_condition,[1/eps],rtol=1e-14)


def test_rounded_nearly_dependent_basis_rejected_before_solver(monkeypatch):
    import tbkit.nonorthogonal as module
    q,_=np.linalg.qr(np.random.default_rng(13).normal(size=(3,3)))
    b=np.diag([1.,1.,1e-8])@q
    m=NonOrthogonal(HK({(0,):b.T@np.diag([1.,2.,3.])@b}),HK({(0,):b.T@b}))
    def forbidden(*args,**kwargs):pytest.fail('unreliable metric reached generalized solver')
    monkeypatch.setattr(module,'eigh',forbidden)
    with pytest.raises(ValueError,match='ill-conditioned|positive definite'):
        m.bands([0.])


@pytest.mark.parametrize('kind',('metric','residual'))
def test_solver_result_is_actually_checked(kind,monkeypatch):
    import tbkit.nonorthogonal as module
    original=module.eigh
    def corrupted(*args,**kwargs):
        e,u=original(*args,**kwargs)
        if kind=='metric':u*=2.
        else:e+=1.
        return e,u
    monkeypatch.setattr(module,'eigh',corrupted)
    m=NonOrthogonal(HK({(0,):SZ}),HK({(0,):np.eye(2)}))
    with pytest.raises(ValueError,match='residual/metric'):
        m.bands([0.])


@pytest.mark.parametrize('scale',(1e-200,1e-20,1.,1e20,1e200))
def test_relative_gap_policy_and_explicit_absolute_units(scale):
    m=NonOrthogonal(HK({(1,):scale*SZ}),HK({(0,):np.eye(2)}))
    close(m.band_gradient([.2])/scale,[[-1.,1.]],rtol=1e-14,atol=0)
    with pytest.raises(ValueError,match='degeneracy'):
        m.band_gradient([0.])
    with pytest.raises(ValueError,match='degeneracy'):
        m.band_gradient([.2],gap_tol=scale)
    close(m.band_gradient([.2],gap_tol=scale*.01)/scale,[[-1.,1.]],rtol=1e-14,atol=0)


def test_relative_gap_uses_bandwidth_not_arbitrary_energy_zero():
    for offset in (0.,1e4,-1e4):
        h=HK({(0,):np.diag([-1.,0.,1e-6])+offset*np.eye(3),(1,):np.diag([.1,.2,.3])})
        m=NonOrthogonal(h,HK({(0,):np.eye(3)}))
        close(m.band_gradient([0.]),[[.1,.2,.3]],atol=1e-14)
        with pytest.raises(ValueError,match='degeneracy'):
            m.band_gradient([0.],gap_rtol=1e-5)
    # All exactly degenerate states must reject even when relative tolerance=0.
    m=NonOrthogonal(HK({(0,):np.eye(2),(1,):SX}),HK({(0,):np.eye(2)}))
    with pytest.raises(ValueError,match='degeneracy'):
        m.band_gradient([0.],gap_rtol=0.)


def test_extreme_common_scales_zero_energy_and_finite_rescaling():
    for c in (np.nextafter(0.,1.),1e-300,1e300):
        m=NonOrthogonal(HK({(0,):np.zeros((1,1))}),HK({(0,):np.array([[c]])}))
        close(m.bands([0.]),[0.],atol=0)
        close(m.band_gradient([0.]),[[0.]],atol=0)
    # hscale/sscale=1e400 overflows, but the SMALL normalized eigenvalue
    # corresponds to the representable physical energy 1e200.
    from tbkit.nonorthogonal import _energy_units
    close(_energy_units(np.array([1e-200]),1e200,1e-200),[1e200],rtol=3e-16,atol=0)
    for value,hs,ss in ((1.,1e300,1e-300),(1.,1e-300,1e300)):
        with pytest.raises(ValueError,match='not representable'):
            _energy_units(np.array([value]),hs,ss)


@pytest.mark.parametrize('invalid',(-1.,np.inf,np.nan,True,1j,'bad'))
def test_argument_validation(invalid):
    m=NonOrthogonal(HK({(0,):SZ}),HK({(0,):np.eye(2)}))
    for key in ('condition_limit','check_tol'):
        with pytest.raises(ValueError):m.bands([0.],**{key:invalid})
    for key in ('gap_tol','gap_rtol'):
        with pytest.raises(ValueError):m.band_gradient([0.],**{key:invalid})


@pytest.mark.parametrize('n',(2,8,9,32))
def test_nopython_diagnostics_zero_and_nonzero_errors(n):
    # Fourier columns provide independent orthonormal eigenframes and test
    # both sides of the small-matrix dispatch boundary, without a second solve.
    indices=np.arange(n)
    frame=np.exp(2j*np.pi*indices[:,None]*indices/n)/np.sqrt(n)
    levels=np.linspace(-1.,2.,n)
    h0=(frame*levels)@frame.conj().T
    h=np.array([h0,h0+np.eye(n)]);s=np.tile(np.eye(n,dtype=complex),(2,1,1))
    e=np.array([levels,levels+1]);u=np.tile(frame,(2,1,1))
    @njit
    def composed(a,b,c,d):return generalized_diagnostics(a,b,c,d)
    for vectors,energy in ((u,e),(1.03*u,e+.01)):
        actual=composed(h,s,energy,vectors)
        reference=generalized_diagnostics_numpy(h,s,energy,vectors)
        for a,b in zip(actual,reference):close(a,b,rtol=2e-14,atol=2e-16)
    assert composed.nopython_signatures


def test_snapshot_ownership_shapes_and_chunk_equivalence():
    m=NonOrthogonal(HK({(0,):.4*SZ,(1,):SX}),HK({(0,):np.eye(2)}))
    k=np.array([[.2],[.4],[.8]])
    r=m.spectrum(k,chunk_size=1);other=m.spectrum(k,chunk_size=10)
    for name in ('energy','eigenvectors','overlap_condition','residual','metric_error'):
        close(getattr(r,name),getattr(other,name),rtol=0,atol=0)
        assert not getattr(r,name).flags.writeable
    k[:]=0;assert r.k[0,0]==.2
    e,u=m.bands([.2],eigenvectors=True)
    assert e.flags.writeable and u.flags.writeable
    assert m.spectrum([.2]).energy.shape==(1,2)
    with pytest.raises(ValueError):m.spectrum([],chunk_size=0)
    with pytest.raises(ValueError):m.spectrum([.2],chunk_size=1.5)
