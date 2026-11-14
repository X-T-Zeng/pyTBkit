"""Small-group theorems and analytic symmetry-adapted quantum states."""
from dataclasses import FrozenInstanceError
import numpy as np
import pytest
import sympy as sp
from numba import njit
from sympy.combinatorics import Permutation,PermutationGroup
from tbkit import (AbstractFiniteGroup,FiniteGroup,cyclic_group,dihedral_group,permutation_group,
                   MatrixRepresentation,CharacterTable,matrix_group,Oper,finite_group)
from tbkit.group_kernels import associative,character_projector


def d3_representations():
    g=dihedral_group(3)
    r=sp.Matrix([[-sp.Rational(1,2),-sp.sqrt(3)/2],[sp.sqrt(3)/2,-sp.Rational(1,2)]])
    s=sp.diag(1,-1)
    a1=MatrixRepresentation(g,[sp.eye(1) for _ in g],name='A_1')
    a2=MatrixRepresentation(g,[sp.Matrix([[(-1)**b]]) for k,b in g],name='A_2')
    e=MatrixRepresentation(g,[r**k*s**b for k,b in g],name='E')
    matrices=[]
    for k,b in g:
        m=sp.zeros(3)
        for j in range(3):m[(k+(-1)**b*j)%3,j]=1
        matrices.append(m)
    return g,(a1,a2,e),MatrixRepresentation(g,matrices,name='vertices')


def test_abstract_group_extension_and_declared_contract():
    with pytest.raises(TypeError):AbstractFiniteGroup()
    class AddMod3(AbstractFiniteGroup):
        @property
        def elements(self):return (0,1,2)
        @property
        def identity(self):return 0
        def multiply(self,a,b):return (a+b)%3
        def inverse(self,a):return (-a)%3
    g=FiniteGroup.from_group(AddMod3())
    assert g.order==3 and g.power(1,-1)==2
    class WrongInverse(AddMod3):
        def inverse(self,a):return a
    with pytest.raises(ValueError,match='inverse'):FiniteGroup.from_group(WrongInverse())


@pytest.mark.parametrize('n',[1,2,3,5,8])
def test_cyclic_axioms_and_known_element_orders(n):
    g=cyclic_group(n)
    assert g.is_abelian and len(g.center())==n
    assert len(g.conjugacy_classes())==n
    for k in g:
        assert g.multiply(g.identity,k)==g.multiply(k,g.identity)==k
        assert g.multiply(k,g.inverse(k))==g.identity
        assert g.element_order(k)==n//int(sp.gcd(k,n))
    assert g.generated_subgroup([]).order==1


def test_dihedral_s3_oracle_subgroups_cosets_and_quotient():
    g=dihedral_group(3)
    oracle=PermutationGroup(Permutation(0,1,2),Permutation(1,2))
    p=permutation_group([(1,2,0),(0,2,1)])
    assert g.order==p.order==oracle.order()==6
    assert sorted(map(len,g.conjugacy_classes()))==sorted(map(len,oracle.conjugacy_classes()))==[1,2,3]
    assert not g.is_abelian and g.center()==(g.identity,)
    assert [h.order for h in g.subgroups()]==[1,2,2,2,3,6]
    rotation=g.generated_subgroup([(1,0)])
    reflection=g.generated_subgroup([(0,1)])
    assert g.is_normal(rotation) and not g.is_normal(reflection)
    for side in ('left','right'):
        cosets=g.cosets(reflection,side=side)
        assert len(cosets)==3 and all(len(c)==2 for c in cosets)
        assert len(set(x for c in cosets for x in c))==6
    assert {frozenset(c) for c in g.cosets(reflection)}!={frozenset(c) for c in g.cosets(reflection,side='right')}
    quotient=g.quotient(iter(rotation))
    assert quotient.order==2 and quotient.is_abelian
    assert g.commutator_subgroup().elements==rotation.elements
    with pytest.raises(ValueError,match='normal'):g.quotient(reflection)
    with pytest.raises(ValueError,match='closed'):g.subgroup([(0,0),(1,0)])


def test_dihedral_d4_known_subgroup_structure_and_relabelled_identity():
    g=dihedral_group(4)
    assert len(g.subgroups())==10
    assert sum(g.is_normal(h) for h in g.subgroups())==6
    assert g.center()==((0,0),(2,0))
    assert g.commutator_subgroup().order==2
    order=[1,0,2];inverse_order=np.argsort(order)
    table=inverse_order[cyclic_group(3).table[np.ix_(order,order)]]
    relabelled=FiniteGroup(table,['a','e','b'])
    assert relabelled.identity=='e' and relabelled.power('a',3)=='e'
    assert relabelled.inverse('a')=='b'
    with pytest.raises(ValueError):relabelled.table[0,0]=0
    with pytest.raises(FrozenInstanceError):relabelled.name='other'


def test_a_latin_loop_with_identity_and_inverses_is_not_necessarily_a_group():
    # (1*1)*2=2, while 1*(1*2)=4. Latin rows and two-sided inverses alone fail.
    loop=np.array([[0,1,2,3,4],[1,0,3,4,2],[2,4,0,1,3],[3,2,4,0,1],[4,3,1,2,0]])
    assert not associative(loop)
    with pytest.raises(ValueError,match='associative'):FiniteGroup(loop)
    with pytest.raises(ValueError):FiniteGroup([[0,.9],[1,0]])
    with pytest.raises(ValueError):FiniteGroup([[0,1],[0,1]])
    with pytest.raises(ValueError):cyclic_group(3).subgroups(max_subgroups=1)
    with pytest.raises(ValueError):permutation_group([(1,1,0)])


def test_s3_character_table_orthogonality_and_tensor_product():
    g,irreps,vertices=d3_representations();a1,a2,e=irreps
    table=CharacterTable(irreps)
    assert table.values==sp.Matrix([[1,1,1],[1,1,-1],[2,-1,0]])
    weights=sp.diag(1,2,3)
    assert table.values*weights*table.values.adjoint()==6*sp.eye(3)
    assert table.values.adjoint()*table.values==sp.diag(6,3,2)
    assert vertices.decompose(irreps)==(1,0,1)
    assert e.tensor(e).decompose(irreps)==(1,1,1)
    assert a1.direct_sum(e).decompose(irreps)==(1,0,1)
    regular=g.regular_representation(exact=True)
    assert regular.characters()==(6,0,0,0,0,0)
    assert regular.decompose(irreps)==(1,1,2)
    assert r'\begin{array}' in table.latex() and table._repr_latex_().startswith('$')
    assert not table.values.atoms(sp.Float)
    with pytest.raises(ValueError):CharacterTable([a1,e])
    with pytest.raises(ValueError):CharacterTable([a1,a1,e])
    with pytest.raises(ValueError):vertices.decompose([a1,e,e])
    with pytest.raises(ValueError):vertices.decompose([a1])
    assert vertices.decompose([a1],require_complete=False)==(1,)


def test_projectors_against_known_three_site_states_and_quantum_degeneracy():
    g,irreps,vertices=d3_representations();a1,a2,e=irreps
    symmetric=sp.ones(3)/3
    assert vertices.projector(a1)==symmetric
    assert vertices.projector(a2)==sp.zeros(3)
    assert vertices.projector(e)==sp.eye(3)-symmetric
    for p in (vertices.projector(a1),vertices.projector(e)):
        assert p*p==p and p.adjoint()==p
        assert all(p*d==d*p for d in vertices.matrices)
    epsilon,t=sp.symbols('epsilon t',real=True)
    h=epsilon*sp.eye(3)+t*(sp.ones(3)-sp.eye(3))
    basis=sp.Matrix.hstack(sp.ones(3,1)/sp.sqrt(3),sp.Matrix([2,-1,-1])/sp.sqrt(6),sp.Matrix([0,1,-1])/sp.sqrt(2))
    assert (basis.adjoint()*h*basis-sp.diag(epsilon+2*t,epsilon-t,epsilon-t)).applyfunc(sp.simplify)==sp.zeros(3)
    np.testing.assert_allclose(vertices.projector(e,backend='numba'),np.eye(3)-np.ones((3,3))/3,atol=1e-14)
    np.testing.assert_allclose(vertices.projector(e,backend='numpy'),vertices.projector(e,backend='numba'),atol=1e-14)
    @njit
    def user_pipeline(matrices,characters,degree):
        p=character_projector(matrices,characters,degree)
        return np.trace(p),p@p-p
    trace,residual=user_pipeline(vertices.as_arrays(),np.array(e.characters(),complex),e.dimension)
    np.testing.assert_allclose(trace,2,atol=1e-14);np.testing.assert_allclose(residual,0,atol=1e-14)
    assert user_pipeline.nopython_signatures


def test_restriction_to_c3_and_complex_characters():
    g,irreps,vertices=d3_representations();rotation=g.generated_subgroup([(1,0)])
    restricted=vertices.restrict(rotation)
    omega=sp.exp(2*sp.pi*sp.I/3)
    irreps_c3=[MatrixRepresentation(rotation,[sp.Matrix([[omega**(j*k)]]) for k,b in rotation],name=f'chi{j}') for j in range(3)]
    assert restricted.decompose(irreps_c3)==(1,1,1)
    assert vertices.restrict(iter(rotation)).dimension==3
    assert CharacterTable(irreps_c3).values.shape==(3,3)


def test_spinor_central_phase_survives_in_matrix_group():
    u=sp.diag(-sp.I,sp.I)
    actions=finite_group([Oper(u,-sp.eye(2))])
    assert len(actions)==2
    defining=matrix_group([u],name='double cover of C2')
    assert defining.group.order==4
    assert defining.characters()==(2,0,-2,0)
    assert defining.group.element_order(1)==4
    assert defining.matrix(2)==-sp.eye(2)
    numeric=matrix_group([np.diag([-1j,1j])])
    assert numeric.group.order==4
    np.testing.assert_allclose(numeric.characters(),[2,0,-2,0],atol=1e-14)
    with pytest.raises(ValueError,match='multiplication law'):
        MatrixRepresentation(cyclic_group(2),[sp.eye(2),u])


def test_antiunitary_trace_is_not_an_ordinary_character():
    # K has matrix I. Rephasing one basis vector by i gives U'=diag(-1,1).
    b=sp.diag(sp.I,1)
    assert sp.trace(sp.eye(2))==2 and sp.trace(b.adjoint()*sp.eye(2)*b.conjugate())==0
    g=cyclic_group(2)
    with pytest.raises(ValueError,match='corepresentations'):
        MatrixRepresentation.from_operators(g,[Oper(sp.eye(2),sp.eye(1)),Oper(sp.eye(2),-sp.eye(1),antiunitary=True)])


def test_numeric_representation_basis_invariance_and_invalid_inputs():
    g,irreps,vertices=d3_representations();rng=np.random.default_rng(8)
    q=np.linalg.qr(rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)))[0]
    rotated=MatrixRepresentation(g,q@vertices.as_arrays()@q.conj().T,name='rotated')
    assert rotated.decompose(irreps)==(1,0,1)
    np.testing.assert_allclose(rotated.projector(irreps[0]),q@(np.ones((3,3))/3)@q.conj().T,atol=1e-14)
    with pytest.raises(ValueError):rotated.matrices[0,0,0]=2
    with pytest.raises(ValueError):MatrixRepresentation(g,[np.eye(2)]*5)
    with pytest.raises(ValueError):MatrixRepresentation(cyclic_group(2),[sp.eye(1),sp.Matrix([[sp.Rational(1,2)]])])
    with pytest.raises(ValueError):MatrixRepresentation(cyclic_group(1),[sp.Matrix([[1.]])],exact=True)
    with pytest.raises(ValueError):matrix_group([sp.Matrix([[2]])])
    with pytest.raises(ValueError):matrix_group([np.array([[np.exp(.31j)]])],max_order=8)


def test_frobenius_schur_real_complex_and_quaternionic_types():
    g,irreps,vertices=d3_representations()
    assert [r.frobenius_schur() for r in irreps]==[1,1,1]
    c3=cyclic_group(3);omega=sp.exp(2*sp.pi*sp.I/3)
    complex_irrep=MatrixRepresentation(c3,[sp.Matrix([[omega**k]]) for k in c3],name='complex')
    assert complex_irrep.frobenius_schur()==0
    sx=sp.Matrix([[0,1],[1,0]]);sy=sp.Matrix([[0,-sp.I],[sp.I,0]])
    quaternionic=matrix_group([sp.I*sx,sp.I*sy],name='Q8')
    assert quaternionic.group.order==8 and quaternionic.is_irreducible()
    assert quaternionic.frobenius_schur()==-1
    assert len(quaternionic.group.center())==2
    with pytest.raises(ValueError):vertices.frobenius_schur()


def test_invariant_multiplet_character_not_per_band_diagonal():
    g,irreps,vertices=d3_representations()
    basis=sp.Matrix.hstack(sp.Matrix([2,-1,-1])/sp.sqrt(6),sp.Matrix([0,1,-1])/sp.sqrt(2))
    rotation=sp.Matrix([[1,1],[1,-1]])/sp.sqrt(2)
    e1=vertices.on_subspace(basis);e2=vertices.on_subspace(basis*rotation)
    assert e1.characters()==e2.characters()==irreps[2].characters()
    assert e1.decompose(irreps)==(0,0,1)
    numeric=vertices.on_subspace(np.array(basis,complex))
    np.testing.assert_allclose(numeric.characters(),np.array(e1.characters(),complex),atol=1e-14)
    with pytest.raises(ValueError,match='invariant'):vertices.on_subspace(sp.Matrix([1,0,0]))
    with pytest.raises(ValueError,match='orthonormal'):vertices.on_subspace(2*basis)
    assert g.regular_representation(exact=True).projector(irreps[2]).trace()==4
