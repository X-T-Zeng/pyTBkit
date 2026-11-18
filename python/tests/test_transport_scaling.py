"""Energy-unit covariance, causality and distinct finite-chain references."""
import numpy as np
import pytest
from numpy.testing import assert_allclose as close
from scipy.linalg import solve

from tbkit import HR
from tbkit.transport import PrincipalLayer,principal_layer,surface_green,transmission,chain_transmission


@pytest.mark.parametrize('scale',[1e-20,1e-12,1.,1e12,1e20])
def test_surface_resolvent_scales_as_inverse_energy(scale):
    for energy in (-3.,-.3,0.,.4,3.):
        eta=1e-6;z=energy+1j*eta
        roots=((z+np.sqrt(z*z-4))/2,(z-np.sqrt(z*z-4))/2)
        analytic=next(g for g in roots if g.imag<0)
        result=surface_green(energy*scale,[[0]],[[scale*np.exp(.7j)]],eta=eta*scale)
        close(result*scale,[[analytic]],atol=2e-10,rtol=1e-10)


@pytest.mark.parametrize('scale',[1e-20,1e-12,1.,1e12,1e20])
def test_perfect_chain_no_hidden_absorber_and_explicit_absorption(scale):
    lead=PrincipalLayer([[0]],[[scale]])
    # At band center, any identical scalar self energies give T=1 for one site
    # in the absence of additional device absorption, independently of unit.
    close(chain_transmission(lead,0,eta=scale*1e-6),1.,atol=2e-13)
    for layers in (2,5):
        close(chain_transmission(lead,.3*scale,layers=layers,eta=scale*1e-7),1.,atol=1e-11)
    eta_relative=.02
    gamma_half=(np.sqrt(eta_relative**2+4)-eta_relative)/2
    extra=.1
    expected=4*gamma_half**2/(2*gamma_half+extra)**2
    close(chain_transmission(lead,0,eta=scale*eta_relative,device_eta=scale*extra),expected,atol=1e-12)


def test_matrix_surface_matches_finite_open_chain_and_basis_rotation():
    h=np.array([[-.2,.12j],[-.12j,.35]])
    v=np.array([[.6,.21j],[-.1,.3]],complex)
    energy=.1;eta=.12;n=160
    finite=np.kron(np.eye(n),h)+np.kron(np.diag(np.ones(n-1),1),v)+np.kron(np.diag(np.ones(n-1),-1),v.conj().T)
    resolvent=solve((energy+1j*eta)*np.eye(2*n)-finite,np.eye(2*n)[:,:2])[:2]
    actual=surface_green(energy,h,v,eta=eta)
    close(actual,resolvent,atol=1e-12)
    opposite=surface_green(energy,h,v.conj().T,eta=eta)
    assert np.max(abs(actual-opposite))>.01  # orientation really matters
    rng=np.random.default_rng(7)
    unitary=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))[0]
    for scale in (1e-20,1e20):
        rotated=surface_green(energy*scale,scale*unitary@h@unitary.conj().T,
                              scale*unitary@v@unitary.conj().T,eta=eta*scale)
        close(rotated*scale,unitary@actual@unitary.conj().T,atol=2e-12)
    close(surface_green(energy+17,h+17*np.eye(2),v,eta=eta),actual,atol=1e-13)


@pytest.mark.parametrize('scale',[1e-20,1.,1e20])
def test_transmission_matches_single_level_formula_at_all_scales(scale):
    energy=.23;onsite=-.17;shift_l=.08;shift_r=-.11;gamma_l=.3;gamma_r=.6
    sl=shift_l-.5j*gamma_l;sr=shift_r-.5j*gamma_r
    expected=gamma_l*gamma_r/((energy-onsite-shift_l-shift_r)**2+(gamma_l+gamma_r)**2/4)
    close(transmission(energy*scale,[[onsite*scale]],[[sl*scale]],[[sr*scale]]),expected,atol=1e-13)
    with pytest.raises(ValueError,match='retarded'):
        transmission(0,[[0]],[[1j*scale]],[[-1j*scale]],eta=.1*scale)


def test_multiorbital_caroli_covariance_and_psd_nullspace():
    rng=np.random.default_rng(5);n=4
    a=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));h=(a+a.conj().T)/2
    coupling_l=rng.normal(size=(n,2))+1j*rng.normal(size=(n,2))
    coupling_r=rng.normal(size=(n,2))+1j*rng.normal(size=(n,2))
    # Wide-band lead self energies, each with a genuine zero broadening space.
    sl=-.5j*coupling_l@coupling_l.conj().T;sr=-.5j*coupling_r@coupling_r.conj().T
    energy=.2
    g=solve(energy*np.eye(n)-h-sl-sr,np.eye(n))
    # Distinct nonnegative Frobenius expression for the Caroli trace.
    expected=float(np.sum(abs(coupling_l.conj().T@g@coupling_r)**2))
    q=np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]
    for scale in (1e-20,1.,1e20):
        close(transmission(energy*scale,scale*h,scale*sl,scale*sr),expected,atol=1e-12)
    close(transmission(energy+5,q@h@q.conj().T+5*np.eye(n),q@sl@q.conj().T,q@sr@q.conj().T),expected,atol=1e-12)
    assert expected<=2+1e-12


@pytest.mark.parametrize('scale',[1e-20,1.,1e20])
def test_nonhermitian_transport_blocks_are_rejected_independent_of_unit(scale):
    for shift in (0.,3.):
        bad=np.array([[shift+1j*scale]])
        with pytest.raises(ValueError,match='Hermitian'):surface_green(shift,bad,[[scale]],eta=scale*1e-6)
        with pytest.raises(ValueError,match='Hermitian'):transmission(shift,bad,[[-1j*scale]],[[-1j*scale]])
        with pytest.raises(ValueError,match='Hermitian'):PrincipalLayer(bad,[[scale]])
    # A missing backward HR hopping must not be hidden by a tiny absolute unit.
    bad=HR([[0],[1]],[[[0]],[[scale]]])
    with pytest.raises(ValueError,match='non-Hermitian lead coupling'):principal_layer(bad)


def test_transport_validation_and_disconnected_surface():
    close(surface_green(.3,[[.1]],[[0]],eta=.02),[[1/(.2+.02j)]])
    for kwargs in ({'eta':np.nan},{'eta':0},{'tol':0},{'tol':np.inf},{'maxiter':2.5}):
        with pytest.raises(ValueError):surface_green(0,[[0]],[[1]],**kwargs)
    for energy in (np.inf,complex(.1),[.1]):
        with pytest.raises(ValueError):surface_green(energy,[[0]],[[1]])
        with pytest.raises(ValueError):transmission(energy,[[0]],[[-1j]],[[-1j]])
    with pytest.raises(ValueError):surface_green(0,[[0]],[[np.nan]])
    with pytest.raises(ValueError):transmission(0,[[0]],[[np.nan]],[[-1j]])
