"""Physics-level invariant completeness, not just residuals of returned terms."""
import itertools
import numpy as np
import pytest
import sympy as sp
from numpy.testing import assert_allclose as close
from scipy.linalg import orth
from tbkit import HK,Oper,models,invariant_kp,symmetrize,finite_group


def spin_rotation(n,*,exact=False):
    t=2*sp.pi/n
    if exact:
        u=sp.diag(sp.exp(-sp.I*t/2),sp.exp(sp.I*t/2))
        q=sp.Matrix([[sp.cos(t),-sp.sin(t)],[sp.sin(t),sp.cos(t)]])
    else:
        t=float(t);u=np.diag(np.exp(-.5j*t*np.array([1,-1])))
        q=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    return Oper.from_spatial(u,q)


def surface_generators():
    c3=spin_rotation(3,exact=True)
    mirror=Oper.from_spatial(sp.I*sp.Matrix(models.sigma_x),sp.diag(-1,1))
    tr=Oper.from_spatial(sp.Matrix([[0,1],[-1,0]]),sp.eye(2),antiunitary=True)
    return [c3,mirror,tr]


def coefficient_vectors(basis,powers):
    vectors=[]
    for h in basis:
        terms={tuple(p):c for p,c in zip(h.powers,h.coefficients)}
        a=np.array([terms.get(p,np.zeros((h.norb,h.norb))) for p in powers])
        vectors.append(np.r_[a.real.ravel(),a.imag.ravel()])
    return np.array(vectors).T


def same_span(actual,expected):
    assert len(actual)==len(expected)
    powers=sorted(set(tuple(p) for h in actual+expected for p in h.powers))
    a=coefficient_vectors(actual,powers);b=coefficient_vectors(expected,powers)
    qa,qb=orth(a,rcond=1e-9),orth(b,rcond=1e-9)
    assert qa.shape[1]==qb.shape[1]==len(expected)
    close(qa@qa.T,qb@qb.T,atol=2e-9)


@pytest.mark.parametrize('n',[3,4,6])
@pytest.mark.parametrize('method',['symbolic','numeric'])
def test_spinful_rotations_complete_linear_space(n,method):
    # Cn alone permits I, sz and both independent isotropic in-plane spin terms.
    sx,sy,sz=models.sigma_x,models.sigma_y,models.sigma_z
    expected=[HK({(0,0):np.eye(2)}),HK({(0,0):sz}),
              HK({(1,0):sx,(0,1):sy}),HK({(1,0):sy,(0,1):-sx})]
    op=spin_rotation(n)
    basis=invariant_kp(2,2,1,[op],method=method)
    same_span(basis,expected)
    for h in basis:assert op.residual(h,[[.137,.247],[-.33,.41]],backend='numpy')<1e-11


@pytest.mark.parametrize('method',['symbolic','numeric'])
def test_c3v_time_reversal_cubic_surface_complete(method):
    # Fu-type surface Hamiltonian through total degree 3, mirror normal x.
    sx,sy,sz=models.sigma_x,models.sigma_y,models.sigma_z;i=np.eye(2)
    expected=[HK({(0,0):i}),HK({(2,0):i,(0,2):i}),
              HK({(1,0):sy,(0,1):-sx}),
              HK({(3,0):sy,(1,2):sy,(2,1):-sx,(0,3):-sx}),
              HK({(3,0):sz,(1,2):-3*sz})]
    generators=surface_generators();basis=invariant_kp(2,2,3,generators,method=method)
    same_span(basis,expected)
    rng=np.random.default_rng(20261113);k=rng.normal(size=(73,2))
    for h in basis:
        close(h.hamiltonian(k,backend='numpy'),h.hamiltonian(k,backend='numpy').conj().swapaxes(-1,-2),atol=1e-12)
        for g in generators:assert g.residual(h,k,backend='numpy')<1e-10
    # An unallowed Dirac mass must actually fail T and mirror.
    mass=HK({(0,0):sz})
    assert generators[1].residual(mass,k,backend='numpy')>1
    assert generators[2].residual(mass,k,backend='numpy')>1


@pytest.mark.parametrize('method',['symbolic','numeric'])
def test_identity_and_global_phase_do_not_remove_terms(method):
    # Relative-only SVD rank selection falsely detects roundoff as constraints.
    op=Oper(np.eye(2)*np.exp(.173j),np.eye(2))
    if method=='symbolic':op=Oper(sp.I*sp.eye(2),sp.eye(2))
    assert len(invariant_kp(2,2,2,[op],method=method))==24


def test_projector_rank_idempotence_and_antiunitary_group_composition():
    generators=surface_generators();group=finite_group(generators)
    assert len(group)==12
    rng=np.random.default_rng(31);k=rng.normal(size=(27,2));functions=[]
    powers=[p for p in itertools.product(range(4),repeat=2) if sum(p)<=3]
    for seed in range(13):
        a=rng.normal(size=(len(powers),2,2))+1j*rng.normal(size=(len(powers),2,2))
        a=(a+a.conj().swapaxes(-1,-2))/2
        h=HK(dict(zip(powers,a)));projected=symmetrize(h,generators)
        y=projected.hamiltonian(k,backend='numpy')
        twice=symmetrize(projected,generators).hamiltonian(k,backend='numpy')
        close(y,twice,atol=1e-11)
        for g in group:assert g.residual(projected,k,backend='numpy')<1e-10
        functions.append(np.r_[y.real.ravel(),y.imag.ravel()])
    assert np.linalg.matrix_rank(np.array(functions),tol=1e-8)==5


@pytest.mark.parametrize('method',['symbolic','numeric'])
def test_chiral_and_particle_hole_are_signed_constraints(method):
    # SSH chiral symmetry: sz H(k) sz = -H(k).
    chiral=Oper(models.sigma_z,np.eye(1),antisymmetry=True)
    assert chiral.residual(models.ssh(),[[.1],[.4]],backend='numpy')<1e-12
    assert len(finite_group([chiral]))==2
    basis=invariant_kp(2,1,1,[chiral],method=method)
    assert len(basis)==4 # sx/sy at degree 0 and 1
    # BdG particle-hole C=sx K: C H(-k) C^-1 = -H(k).
    ph=Oper.from_spatial(models.sigma_x,np.eye(1),antiunitary=True,antisymmetry=True)
    allowed=invariant_kp(2,1,1,[ph],method=method)
    expected=[HK({(0,):models.sigma_z}),HK({(1,):np.eye(2)}),HK({(1,):models.sigma_x}),HK({(1,):models.sigma_y})]
    same_span(allowed,expected)


def test_explicit_exact_rotations_and_unrecoverable_float_rejection():
    # Exact rational infinitesimal rotation is not erased by float cleanup.
    p=sp.Rational(1,10**14)
    r=sp.Matrix([[(1-p*p)/(1+p*p),-2*p/(1+p*p)],[2*p/(1+p*p),(1-p*p)/(1+p*p)]])
    op=Oper(sp.eye(1),r)
    _,recovered=op.exact_matrices()
    assert recovered==r
    # Arbitrary trigonometric matrices have a numerical SVD path.
    t=.173;q=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    arbitrary=Oper(np.eye(1),q)
    assert len(invariant_kp(1,2,2,[arbitrary],method='numeric'))==2 # 1 and kx^2+ky^2


def bhz_generators():
    # Explicit diagonal C4 representation near the start of the MATLAB tutorial.
    i,sx,sy,sz=np.eye(2),models.sigma_x,models.sigma_y,models.sigma_z
    u=sp.diag(sp.exp(sp.I*sp.pi/4),sp.exp(3*sp.I*sp.pi/4),sp.exp(-sp.I*sp.pi/4),sp.exp(-3*sp.I*sp.pi/4))
    c4=Oper.from_spatial(u,sp.Matrix([[0,-1,0],[1,0,0],[0,0,1]]))
    tr=Oper.from_spatial(-1j*np.kron(sy,i),np.eye(3),antiunitary=True)
    inversion=Oper.from_spatial(np.kron(i,sz),-np.eye(3))
    mirror=Oper.from_spatial(1j*np.kron(sy,sz),np.diag([1,-1,1]))
    return [c4,tr,inversion,mirror]


@pytest.mark.parametrize('method',['symbolic','numeric'])
def test_bhz_quadratic_seven_parameter_space(method):
    i,sx,sy,sz=np.eye(2),models.sigma_x,models.sigma_y,models.sigma_z
    expected=[]
    for matrix in (np.eye(4),np.kron(i,sz)):
        expected.extend([HK({(0,0,0):matrix}),HK({(0,0,2):matrix}),HK({(2,0,0):matrix,(0,2,0):matrix})])
    expected.append(HK({(1,0,0):np.kron(sz,sx),(0,1,0):np.kron(i,sy)}))
    basis=invariant_kp(4,3,2,bhz_generators(),method=method)
    same_span(basis,expected)
