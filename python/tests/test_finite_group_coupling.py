"""Independent ket coefficients, Schur multiplicities and angular-CG boundaries."""
import numpy as np
import pytest
import sympy as sp
from numba import njit
from tbkit import (MatrixRepresentation,cyclic_group,dihedral_group,matrix_group,
                   symmetry_adapted_basis,clebsch_gordan,irreducible_representations)
from tbkit.group_kernels import matrix_unit_column
from tbkit.quantum import coupled_basis
from tbkit.orbitals import angular_momentum


@pytest.fixture(scope='module')
def d3():
    group=dihedral_group(3)
    rotation=sp.Matrix([[-sp.Rational(1,2),-sp.sqrt(3)/2],[sp.sqrt(3)/2,-sp.Rational(1,2)]])
    mirror=sp.diag(1,-1)
    return (MatrixRepresentation(group,[sp.eye(1) for _ in group],name='A1'),
            MatrixRepresentation(group,[sp.Matrix([[(-1)**b]]) for a,b in group],name='A2'),
            MatrixRepresentation(group,[rotation**a*mirror**b for a,b in group],name='E'))


def analytic_d3_coefficients():
    # Coupled ket columns A1, A2, E_x, E_y in xx,xy,yx,yy order.
    return sp.Matrix([[1,0,1,0],[0,1,0,-1],[0,-1,0,-1],[1,0,-1,0]])/sp.sqrt(2)


def test_exact_d3_tensor_matches_analytic_cg_up_to_allowed_phase(d3):
    a1,a2,e=d3;result=clebsch_gordan(e,e,d3)
    expected=analytic_d3_coefficients();assert result.exact
    assert result.multiplicities==(1,1,1)
    assert result.basis.adjoint()*result.basis==sp.eye(4)
    for index,irrep in enumerate(d3):
        actual=result.copies(index)[0];known=expected[:,result.slices[index]]
        overlap=(known.adjoint()*actual).applyfunc(sp.simplify)
        assert (overlap-overlap[0,0]*sp.eye(irrep.dimension)).applyfunc(sp.simplify)==sp.zeros(irrep.dimension)
        assert sp.simplify(sp.conjugate(overlap[0,0])*overlap[0,0])==1
    swap=sp.Matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
    assert result.transform(swap).applyfunc(sp.simplify)==sp.diag(1,-1,1,1)
    assert result.coefficients.shape==(2,2,4)
    for a in range(2):
        for b in range(2):
            for c in range(4):assert result.coefficients[a,b,c]==result.basis[2*a+b,c]


@pytest.mark.parametrize('backend',['numpy','numba'])
def test_numeric_d3_matches_analytic_projectors_and_full_intertwining(d3,backend):
    e=d3[2];result=clebsch_gordan(e,e,d3,backend=backend)
    expected=np.array(analytic_d3_coefficients(),complex)
    assert not result.exact and not result.basis.flags.writeable
    for index,irrep in enumerate(d3):
        actual=result.copies(index)[0];known=expected[:,result.slices[index]]
        np.testing.assert_allclose(actual@actual.conj().T,known@known.conj().T,atol=1e-12)
        np.testing.assert_allclose(result.representation.as_arrays()@actual,actual@irrep.as_arrays(),atol=1e-12)
    np.testing.assert_allclose(result.coefficients.reshape(4,4),result.basis,atol=0)


def test_three_tensor_factors_resolve_three_equivalent_doublets_exactly(d3):
    e=d3[2];target=e.tensor(e).tensor(e)
    result=symmetry_adapted_basis(target,d3)
    assert result.multiplicities==(1,1,3)
    assert len(result.copies(2))==3
    assert (result.basis.adjoint()*result.basis-sp.eye(8)).applyfunc(sp.simplify)==sp.zeros(8)
    for q in result.copies(2):
        for big,small in zip(target.matrices,e.matrices):
            assert (big*q-q*small).applyfunc(sp.simplify)==sp.zeros(8,2)


def test_exact_repeated_irrep_retains_symbolic_multiplicity_hamiltonian(d3):
    e=d3[2];result=symmetry_adapted_basis(e.direct_sum(e),d3)
    assert result.multiplicities==(0,0,2)
    x,y,tr,ti=sp.symbols('x y t_r t_i',real=True)
    multiplicity=sp.Matrix([[x,tr+sp.I*ti],[tr-sp.I*ti,y]])
    h=sp.kronecker_product(multiplicity,sp.eye(2))
    # In this explicit copy basis the exact construction chooses the identity.
    assert result.basis==sp.eye(4)
    assert result.transform(h)==h
    # A general operator is transformed as supplied, without Hermitization.
    raising=sp.zeros(4);raising[0,1]=1
    assert result.transform(raising)==raising


@pytest.mark.parametrize('backend',['numpy','numba'])
def test_random_copy_gauge_has_schur_blocks_not_independent_irrep_rotations(d3,backend):
    e=d3[2];rng=np.random.default_rng(73)
    rotation=np.linalg.qr(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))[0]
    target=MatrixRepresentation(e.group,rotation@e.direct_sum(e).as_arrays()@rotation.conj().T)
    result=symmetry_adapted_basis(target,d3,backend=backend)
    assert result.multiplicities==(0,0,2)
    resolved=result.basis.conj().T@target.as_arrays()@result.basis
    np.testing.assert_allclose(resolved,np.array([np.kron(np.eye(2),m) for m in e.as_arrays()]),atol=2e-12)
    m=np.array([[.3,.4+.2j],[.4-.2j,-.5]])
    h=rotation@np.kron(m,np.eye(2))@rotation.conj().T
    block=result.transform(h).reshape(2,2,2,2)
    reduced=block[:,0,:,0]
    np.testing.assert_allclose(block,np.kron(reduced,np.eye(2)).reshape(2,2,2,2),atol=2e-12)
    # Independent two-level roots; each must remain exactly double in theory.
    middle=(m[0,0].real+m[1,1].real)/2
    gap=np.sqrt(((m[0,0].real-m[1,1].real)/2)**2+abs(m[0,1])**2)
    np.testing.assert_allclose(np.linalg.eigvalsh(h),[middle-gap]*2+[middle+gap]*2,atol=1e-13)


@pytest.mark.parametrize('backend',['sympy','numpy','numba'])
def test_reference_irrep_basis_is_matched_beyond_characters(d3,backend):
    e=d3[2];change=sp.diag(1,sp.I)
    changed=MatrixRepresentation(e.group,[change.adjoint()*m*change for m in e.matrices],name='Eprime')
    result=symmetry_adapted_basis(e.direct_sum(e),[changed],backend=backend)
    assert result.multiplicities==(2,)
    for copy in result.copies(0):
        for big,small in zip(e.direct_sum(e).matrices,changed.matrices):
            if result.exact:assert (copy.adjoint()*big*copy-small).applyfunc(sp.simplify)==sp.zeros(2)
            else:np.testing.assert_allclose(copy.conj().T@np.array(big,complex)@copy,np.array(small,complex),atol=1e-12)


@pytest.mark.parametrize('backend',['sympy','numpy','numba'])
def test_complex_cyclic_charge_addition_needs_conjugated_matrix_weights(backend):
    group=cyclic_group(3);root=-sp.Rational(1,2)+sp.I*sp.sqrt(3)/2
    irreps=[MatrixRepresentation(group,[sp.Matrix([[(root**(q*g)).expand(complex=True)]]) for g in group],name=f'q{q}') for q in range(3)]
    result=clebsch_gordan(irreps[1],irreps[1],irreps,backend=backend)
    assert result.exact is (backend=='sympy') and result.multiplicities==(0,0,1)
    if result.exact:assert result.basis==sp.eye(1)
    else:np.testing.assert_allclose(result.basis,np.eye(1),atol=1e-12)


def test_quaternionic_tensor_has_antisymmetric_invariant_ket():
    sx=sp.Matrix([[0,1],[1,0]]);sy=sp.Matrix([[0,-sp.I],[sp.I,0]])
    defining=matrix_group([sp.I*sx,sp.I*sy],name='Q8');group=defining.group
    one_dimensional=[]
    for x,y in ((1,1),(1,-1),(-1,1),(-1,-1)):
        lookup={}
        for matrix,character in ((sp.eye(2),1),(sp.I*sx,x),(sp.I*sy,y),((sp.I*sx)*(sp.I*sy),x*y)):
            lookup[sp.ImmutableMatrix(matrix)]=character;lookup[sp.ImmutableMatrix(-matrix)]=character
        one_dimensional.append(MatrixRepresentation(group,[sp.Matrix([[lookup[m]]]) for m in defining.matrices],name=f'{x}_{y}'))
    result=clebsch_gordan(defining,defining,one_dimensional)
    assert result.multiplicities==(1,1,1,1)
    singlet=sp.Matrix([0,1,-1,0])/sp.sqrt(2)
    q=result.copies(0)[0]
    assert (q*q.adjoint()-singlet*singlet.adjoint()).applyfunc(sp.simplify)==sp.zeros(4)
    # Restriction to Q8 splits the other three Bell sectors into 1D irreps;
    # this does not identify a single 3D irreducible SU(2) triplet.


def test_automatic_numeric_catalogue_and_absent_irrep_metadata():
    group=dihedral_group(3);irreps=irreducible_representations(group)
    e=next(r for r in irreps if r.dimension==2)
    result=clebsch_gordan(e,e,irreps)
    assert result.multiplicities==(1,1,1)
    trivial=symmetry_adapted_basis(irreps[0],irreps)
    assert trivial.multiplicities==(1,0,0)
    assert trivial.copies(1)==()
    with pytest.raises(ValueError):trivial.coefficients
    with pytest.raises(ValueError):trivial.copies(9)


def test_matrix_unit_kernel_against_analytic_operators_and_nopython(d3):
    e=d3[2];target=e.tensor(e);q=np.array(analytic_d3_coefficients()[:,2:],complex)
    expected=np.array([np.outer(q[:,i],q[:,0].conj()) for i in range(2)])
    @njit
    def pipeline(big,small):return matrix_unit_column(big,small)
    np.testing.assert_allclose(pipeline(target.as_arrays(),e.as_arrays()),expected,atol=1e-14)
    assert pipeline.nopython_signatures


def test_incomplete_duplicate_nonirrep_and_mismatched_groups_rejected(d3):
    a1,a2,e=d3;target=e.tensor(e)
    for catalogue in ([a1,e],[a1,a1,a2,e],[target],[]):
        with pytest.raises(ValueError):symmetry_adapted_basis(target,catalogue)
    other=MatrixRepresentation(cyclic_group(1),[sp.eye(1)])
    with pytest.raises(ValueError):symmetry_adapted_basis(a1,[other])
    with pytest.raises(TypeError):symmetry_adapted_basis(None,d3)


def test_bounds_backends_and_operator_shapes_rejected(d3):
    e=d3[2]
    for options in ({'backend':'unknown'},{'max_dimension':1},{'tolerance':0},{'tolerance':True},{'tolerance':.5}):
        with pytest.raises(ValueError):symmetry_adapted_basis(e,d3,**options)
    with pytest.raises(ValueError):clebsch_gordan(e,e,d3,max_dimension=3)
    numeric=MatrixRepresentation(e.group,e.as_arrays())
    with pytest.raises(ValueError):symmetry_adapted_basis(numeric,d3,backend='sympy')
    result=symmetry_adapted_basis(e,d3)
    with pytest.raises(ValueError):result.transform(sp.eye(3))
    with pytest.raises(ValueError):result.transform(np.eye(3))


def test_exact_parameterized_representations_require_binding():
    group=cyclic_group(2);theta=sp.symbols('theta',real=True)
    reflection=sp.Matrix([[sp.cos(theta),sp.sin(theta)],[sp.sin(theta),-sp.cos(theta)]])
    target=MatrixRepresentation(group,[sp.eye(2),reflection])
    plus=MatrixRepresentation(group,[sp.eye(1),sp.eye(1)])
    minus=MatrixRepresentation(group,[sp.eye(1),-sp.eye(1)])
    with pytest.raises(ValueError,match='bind representation parameters'):
        symmetry_adapted_basis(target,[plus,minus])


@pytest.mark.parametrize('l,s',[(1,sp.Rational(1,2)),(2,1)])
def test_existing_angular_cg_diagonalizes_total_j_squared(l,s):
    matrix,coupled,_=coupled_basis(l,s)
    assert matrix*matrix.adjoint()==sp.eye(matrix.rows)
    orbital=angular_momentum(l);spin=angular_momentum(float(s))
    total=[np.kron(a,np.eye(int(2*s+1)))+np.kron(np.eye(2*l+1),b) for a,b in zip(orbital,spin)]
    c=np.array(matrix,complex)
    np.testing.assert_allclose(c@sum(j@j for j in total)@c.conj().T,np.diag([float(j*(j+1)) for j,m in coupled]),atol=1e-13)
    np.testing.assert_allclose(c@total[2]@c.conj().T,np.diag([float(m) for j,m in coupled]),atol=1e-13)
