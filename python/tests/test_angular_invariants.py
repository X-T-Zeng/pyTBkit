"""Discrete SU(2) labels, scale-free axes and infinitesimal quantum response."""
import math
import numpy as np
import pytest
import sympy as sp
from numba import njit
from scipy.spatial.transform import Rotation
from tbkit import BasisFunc,Qnum,angular_cg
from tbkit.orbitals import angular_momentum,spin_rotation,orbital_rotation
from tbkit.angular_kernels import angular_momentum_doubled


@pytest.mark.parametrize('j',[0,.5,1,1.5,10,50,sp.Rational(1,2)])
def test_su2_commutators_casimir_and_exact_m_spacing(j):
    x,y,z=angular_momentum(j);j=float(j);n=int(2*j)+1
    for matrix in (x,y,z):np.testing.assert_allclose(matrix,matrix.conj().T,atol=0,rtol=0)
    for a,b,c in ((x,y,z),(y,z,x),(z,x,y)):
        assert np.max(abs(a@b-b@a-1j*c))/max(1,j*j)<2e-14
    np.testing.assert_allclose(x@x+y@y+z@z,j*(j+1)*np.eye(n),atol=2e-12*max(1,j*j),rtol=0)
    np.testing.assert_array_equal(z.diagonal(),j-np.arange(n))


@pytest.mark.parametrize('j',[.500001,50.0004,-.5,np.nan,np.inf,True,.5+0j])
def test_invalid_discrete_labels_are_not_rounded(j):
    with pytest.raises(ValueError):angular_momentum(j)


@pytest.mark.parametrize('spin',[.5,2])
@pytest.mark.parametrize('scale',[1e-300,1e300])
def test_axis_rescaling_preserves_group_composition_and_unitarity(spin,scale):
    axis=np.array([1.,2.,-3.]);u=spin_rotation(axis*scale,.7,spin=spin)
    np.testing.assert_allclose(u,spin_rotation(axis,.7,spin=spin),atol=3e-15,rtol=0)
    np.testing.assert_allclose(u.conj().T@u,np.eye(len(u)),atol=5e-15,rtol=0)
    np.testing.assert_allclose(u@spin_rotation(axis*scale,-.2,spin=spin),spin_rotation(axis,.5,spin=spin),atol=5e-15,rtol=0)
    np.testing.assert_allclose(spin_rotation(-axis*scale,.7,spin=spin),u.conj().T,atol=5e-15,rtol=0)


@pytest.mark.parametrize('theta',[1e-20,1e-16,1e-9])
def test_infinitesimal_response_matches_independent_commutator(theta):
    up=Qnum(0,0,.5,.5);down=Qnum(0,0,.5,-.5)
    rotated=BasisFunc({up:1}).rotate([1,0,0],theta)
    state=np.array([complex(rotated.terms.get(up,0)),complex(rotated.terms.get(down,0))])
    sx=np.array([[0,1],[1,0]],complex)/2;sy=np.array([[0,-1j],[1j,0]])/2
    initial=np.array([1,0],complex)
    derivative=(1j*np.vdot(initial,(sx@sy-sy@sx)@initial)).real
    response=np.vdot(state,sy@state).real
    assert derivative==-.5
    np.testing.assert_allclose(response,-math.sin(theta)/2,rtol=2e-14,atol=0)
    assert abs(response/theta-derivative)<1e-13
    assert down in rotated.terms


def test_unnormalized_symbolic_amplitude_retains_order_one_mixing():
    up=Qnum(0,0,.5,.5);down=Qnum(0,0,.5,-.5)
    rotated=BasisFunc({up:sp.Integer(10)**20}).rotate([1,0,0],1e-20)
    np.testing.assert_allclose(complex(rotated.terms[down]),-.5j,atol=1e-14,rtol=0)


def test_combined_orbital_spin_rotation_scale_and_double_cover():
    initial=BasisFunc({Qnum(1,1,.5,.5):1});axis=np.array([1.,2.,-3.])
    labels=[Qnum(1,m,.5,z) for m in (1,0,-1) for z in (.5,-.5)]
    expected=initial.rotate(axis,.7)
    for scale in (1e-300,1e300):
        result=initial.rotate(axis*scale,.7)
        np.testing.assert_allclose([complex(result.terms.get(q,0)) for q in labels],
                                   [complex(expected.terms.get(q,0)) for q in labels],atol=4e-15,rtol=0)
    full=initial.rotate(axis,2*np.pi)
    np.testing.assert_allclose(complex(initial.inner(full)),-1,atol=6e-15,rtol=0)


def test_large_angles_use_four_pi_period_without_losing_spinor_sign():
    for theta in (2*np.pi,4*np.pi,5*np.pi,1e300,-1e300):
        expected=np.array([[complex(math.cos(theta/2),-math.sin(theta/2)),0],
                           [0,complex(math.cos(theta/2),math.sin(theta/2))]])
        np.testing.assert_allclose(spin_rotation([0,0,1],theta),expected,atol=3e-15,rtol=0)
        expected_x=math.cos(theta/2)*np.eye(2)-1j*math.sin(theta/2)*np.array([[0,1],[1,0]])
        np.testing.assert_allclose(spin_rotation([1,0,0],theta),expected_x,atol=3e-15,rtol=0)
    np.testing.assert_allclose(spin_rotation([0,0,1],2*np.pi),-np.eye(2),atol=3e-15,rtol=0)
    np.testing.assert_allclose(spin_rotation([0,0,1],4*np.pi),np.eye(2),atol=3e-15,rtol=0)


def test_nonorthogonal_complex_orbital_input_is_rejected_and_inversion_retained():
    for matrix in ((1+1e-7)*np.eye(3),np.eye(3)+1e-3j*np.ones((3,3)),np.eye(3)*np.nan):
        with pytest.raises(ValueError):orbital_rotation(matrix)
    np.testing.assert_allclose(orbital_rotation(-np.eye(3)),np.diag([1,-1,-1,-1,1,1,1,1,1]),atol=1e-15,rtol=0)
    first=Rotation.from_rotvec([.3,-.4,.2]).as_matrix();second=Rotation.from_rotvec([-.7,.1,.5]).as_matrix()
    np.testing.assert_allclose(orbital_rotation(first)@orbital_rotation(second),orbital_rotation(first@second),atol=2e-15,rtol=0)


def test_rotation_invalid_axes_angles_and_dimension_budgets():
    state=BasisFunc({Qnum(0,0):1})
    for axis in ([0,0,0],[np.inf,0,0],[np.nan,0,0],[1+1j,0,0],[1,2]):
        with pytest.raises(ValueError):spin_rotation(axis,.2)
        with pytest.raises(ValueError):state.rotate(axis,.2)
    for angle in (np.nan,np.inf,.2j,True,[.2]):
        with pytest.raises(ValueError):spin_rotation([1,0,0],angle)
        with pytest.raises(ValueError):state.rotate([1,0,0],angle)
    with pytest.raises(ValueError):angular_momentum(50,max_dimension=100)
    with pytest.raises(ValueError):angular_momentum(513)
    with pytest.raises(ValueError):angular_momentum(np.uint64(2**63))
    with pytest.raises(ValueError):angular_momentum(np.nextafter(np.longdouble(.5),np.longdouble(1)))
    with pytest.raises(ValueError):angular_momentum(.5,max_dimension=True)
    with pytest.raises(ValueError):spin_rotation([1,0,0],.2,spin=2,max_dimension=4)


def test_numba_angular_generator_can_be_composed_without_object_mode():
    @njit
    def pipeline(twice_j):
        x,y,z=angular_momentum_doubled(twice_j)
        return np.max(np.abs(x@y-y@x-1j*z)),x
    residual,x=pipeline(1)
    assert residual==0
    np.testing.assert_array_equal(x,np.array([[0,.5],[.5,0]]))
    assert pipeline.nopython_signatures
    with pytest.raises(ValueError):angular_momentum_doubled(-1)


@pytest.mark.parametrize('j',[sp.Rational(1,2),1,5,50,100,128])
def test_scalar_cg_high_j_singlet_closed_form_and_phase(j):
    for m in (j,j-1):
        expected=(-1)**(j-m)/sp.sqrt(2*j+1)
        assert sp.simplify(angular_cg(j,j,0,m,-m,0)-expected)==0
        np.testing.assert_allclose(angular_cg(j,j,0,m,-m,0,exact=False),float(expected),atol=0,rtol=2e-15)
    assert angular_cg(j,j,2*j,j,j,2*j)==1


def test_scalar_cg_stretched_multiplet_binomial_closed_form():
    j=50
    for m in (0,10,50):
        expected=sp.sqrt(sp.binomial(2*j,j+m)*sp.binomial(2*j,j-m)/sp.binomial(4*j,2*j))
        assert sp.simplify(angular_cg(j,j,2*j,m,-m,0)-expected)==0


def test_scalar_cg_fixed_m_block_normalization_and_total_j_squared():
    j1,j2,m=3,2,1
    labels=[(a,m-a) for a in range(j1,-j1-1,-1) if abs(m-a)<=j2]
    degrees=list(range(j1+j2,abs(j1-j2)-1,-1))
    coefficients=np.array([[angular_cg(j1,j2,j,a,b,m,exact=False) for a,b in labels] for j in degrees])
    np.testing.assert_allclose(coefficients@coefficients.T,np.eye(5),atol=5e-15,rtol=0)
    first=angular_momentum(j1);second=angular_momentum(j2)
    total=[np.kron(a,np.eye(2*j2+1))+np.kron(np.eye(2*j1+1),b) for a,b in zip(first,second)]
    indices=[(j1-a)*(2*j2+1)+(j2-b) for a,b in labels]
    block=sum(a@a for a in total)[np.ix_(indices,indices)]
    np.testing.assert_allclose(coefficients@block@coefficients.T,np.diag([j*(j+1) for j in degrees]),atol=3e-14,rtol=0)


def test_scalar_cg_distinguishes_invalid_states_and_forbidden_channels():
    assert angular_cg(1,1,0,1,0,0)==0
    assert angular_cg(1,1,1,0,0,0)==0  # exchange-odd J cannot contain |0,0>
    assert angular_cg(1,1,3,0,0,0,exact=False)==0.0
    for arguments in ((-.5,.5,0,-.5,.5,0),(.5,.5,0,0,0,0),(.500001,.5,0,.5,-.5,0),
                      (np.nan,.5,0,.5,-.5,0),(True,1,0,0,0,0)):
        with pytest.raises(ValueError):angular_cg(*arguments)
    with pytest.raises(ValueError):angular_cg(1,1,0,0,0,0,exact='yes')
