"""Analytic character oracles, including complex and quaternionic irreps."""
import numpy as np
import pytest
import sympy as sp
from numba import njit
from tbkit import (FiniteGroup,MatrixRepresentation,CharacterTable,cyclic_group,
                   dihedral_group,permutation_group,matrix_group,
                   irreducible_representations,IrrepGenerationError)
from tbkit.group_kernels import regular_blocks


def assert_rows_match(actual,expected,tolerance=2e-8):
    actual=list(np.asarray(actual,dtype=complex));expected=list(np.asarray(expected,dtype=complex))
    assert len(actual)==len(expected)
    for row in actual:
        matches=[i for i,value in enumerate(expected) if np.allclose(row,value,atol=tolerance,rtol=0)]
        assert len(matches)==1,(row,expected)
        expected.pop(matches[0])
    assert not expected


@pytest.mark.parametrize('order',[1,3,5,8])
def test_cyclic_irreps_against_roots_of_unity(order):
    group=cyclic_group(order);irreps=irreducible_representations(group)
    expected=np.exp(2j*np.pi*np.arange(order)[:,None]*np.arange(order)/order)
    assert_rows_match([r.characters() for r in irreps],expected)
    assert all(r.dimension==1 and not r.exact for r in irreps)
    assert all(r.as_arrays().dtype==np.complex128 for r in irreps)
    assert all(not r.as_arrays().flags.writeable for r in irreps)
    assert irreps[0].name=='rho_0'
    np.testing.assert_allclose(irreps[0].characters(),1,atol=1e-9)
    if order==3:assert sorted(r.frobenius_schur() for r in irreps)==[0,0,1]
    assert_rows_match(np.asarray(CharacterTable.from_group(group).values,dtype=complex),
                      [r.characters(by_class=True) for r in irreps])


@pytest.mark.parametrize('n',[3,4,5,6])
def test_dihedral_character_formula(n):
    group=dihedral_group(n);irreps=irreducible_representations(group)
    expected=[]
    for rotation in ([1,-1] if n%2==0 else [1]):
        for reflection in (1,-1):
            expected.append([rotation**a*reflection**b for a,b in group])
    for k in range(1,(n-1)//2+1):
        expected.append([2*np.cos(2*np.pi*k*a/n) if b==0 else 0 for a,b in group])
    assert_rows_match([r.characters() for r in irreps],expected)
    assert all(r.frobenius_schur()==1 for r in irreps)
    regular=group.regular_representation()
    assert regular.decompose(irreps)==tuple(r.dimension for r in irreps)


def test_q8_retains_quaternionic_complex_irrep_and_central_minus_identity():
    sx=sp.Matrix([[0,1],[1,0]]);sy=sp.Matrix([[0,-sp.I],[sp.I,0]])
    defining=matrix_group([sp.I*sx,sp.I*sy],name='Q8')
    irreps=irreducible_representations(defining.group)
    assert sorted(r.dimension for r in irreps)==[1,1,1,1,2]
    spinor=next(r for r in irreps if r.dimension==2)
    assert spinor.frobenius_schur()==-1
    np.testing.assert_allclose(spinor.characters(),np.asarray(defining.characters(),dtype=complex),atol=1e-9)
    central=next(g for g in defining.group if defining.matrix(g)==-sp.eye(2))
    np.testing.assert_allclose(spinor.matrix(central),-np.eye(2),atol=1e-9)
    assert all(r.frobenius_schur()==1 for r in irreps if r.dimension==1)


def cycle_type(permutation):
    unseen=set(range(len(permutation)));lengths=[]
    while unseen:
        start=min(unseen);node=start;length=0
        while node in unseen:
            unseen.remove(node);length+=1;node=permutation[node]
        lengths.append(length)
    return tuple(sorted(lengths,reverse=True))


def test_s4_known_character_table_by_cycle_types():
    group=permutation_group([(1,2,3,0),(1,0,2,3)])
    types=[(1,1,1,1),(2,1,1),(2,2),(3,1),(4,)]
    representatives=[next(g for g in group if cycle_type(g)==kind) for kind in types]
    irreps=irreducible_representations(group)
    actual=[[r.characters()[group.index(g)] for g in representatives] for r in irreps]
    expected=[[1,1,1,1,1],[1,-1,1,1,-1],[2,0,2,-1,0],[3,1,-1,0,-1],[3,-1,-1,0,1]]
    assert_rows_match(actual,expected)
    assert sorted(r.dimension for r in irreps)==[1,1,2,3,3]


def test_a4_complex_pair_and_three_dimensional_irrep():
    group=permutation_group([(1,2,0,3),(1,0,3,2)])
    assert group.order==12
    irreps=irreducible_representations(group)
    assert sorted(r.dimension for r in irreps)==[1,1,1,3]
    assert sorted(r.frobenius_schur() for r in irreps)==[0,0,1,1]
    spatial=next(r for r in irreps if r.dimension==3)
    expected=[3 if cycle_type(g)==(1,1,1,1) else (-1 if cycle_type(g)==(2,2) else 0) for g in group]
    np.testing.assert_allclose(spatial.characters(),expected,atol=1e-9)


def test_a5_golden_ratio_characters_and_dimensions():
    group=permutation_group([(1,2,3,4,0),(1,2,0,3,4)])
    assert group.order==60
    irreps=irreducible_representations(group)
    assert sorted(r.dimension for r in irreps)==[1,3,3,4,5]
    phi=(1+np.sqrt(5))/2
    for irrep in irreps:
        chars=irrep.characters();d=irrep.dimension
        known={1:{2:1,3:1,5:[1,1]},3:{2:-1,3:0,5:[phi,1-phi]},4:{2:0,3:1,5:[-1,-1]},5:{2:1,3:-1,5:[0,0]}}[d]
        for order in (2,3):
            np.testing.assert_allclose([chars[group.index(g)] for g in group if group.element_order(g)==order],known[order],atol=1e-8)
        values=[chars[group.index(c[0])] for c in group.conjugacy_classes() if group.element_order(c[0])==5]
        np.testing.assert_allclose(sorted(np.real(values)),sorted(known[5]),atol=1e-8)
        assert np.max(np.abs(np.imag(values)))<1e-9


def test_irrep_characters_relabel_with_nonzero_identity_index():
    original=dihedral_group(4);order=np.array([4,1,0,3,2,5,6,7]);back=np.argsort(order)
    group=FiniteGroup(back[original.table[np.ix_(order,order)]],[f'g{i}' for i in order])
    assert group.index(group.identity)==2
    expected=[r.characters()[order] for r in irreducible_representations(original)]
    actual=[r.characters() for r in irreducible_representations(group)]
    assert_rows_match(actual,expected)


def test_regular_block_kernel_against_explicit_permutation_matrices_and_njit():
    group=dihedral_group(4)
    rng=np.random.default_rng(17)
    basis=np.ascontiguousarray(np.linalg.qr(rng.normal(size=(8,3))+1j*rng.normal(size=(8,3)))[0])
    # Arbitrary Q need not be invariant to check the algebraic compression.
    expected=basis.conj().T@group.regular_representation().as_arrays()@basis
    @njit
    def pipeline(table,q):return regular_blocks(table,q)
    np.testing.assert_allclose(pipeline(group.table,basis),expected,atol=1e-14)
    assert pipeline.nopython_signatures


def test_tetrahedron_quantum_projectors_and_tensor_product():
    group=permutation_group([(1,2,3,0),(1,0,2,3)])
    irreps=irreducible_representations(group)
    sites=MatrixRepresentation(group,[np.eye(4)[:,g] for g in group],name='sites')
    multiplicities=sites.decompose(irreps)
    assert sorted((r.dimension,m) for r,m in zip(irreps,multiplicities) if m)==[(1,1),(3,1)]
    spatial=next(r for r,m in zip(irreps,multiplicities) if m and r.dimension==3)
    # Standard S4 rep tensor square = trivial + dimension-2 + both dimension-3.
    assert sorted((r.dimension,m) for r,m in zip(irreps,spatial.tensor(spatial).decompose(irreps)) if m)==[(1,1),(2,1),(3,1),(3,1)]
    symmetric=np.ones((4,4))/4
    np.testing.assert_allclose(sites.projector(irreps[0]),symmetric,atol=1e-9)
    np.testing.assert_allclose(sites.projector(spatial),np.eye(4)-symmetric,atol=1e-9)
    epsilon,t=.2,-.7;hamiltonian=epsilon*np.eye(4)+t*(np.ones((4,4))-np.eye(4))
    np.testing.assert_allclose(np.linalg.eigvalsh(hamiltonian),[epsilon+3*t,*([epsilon-t]*3)],atol=1e-14)
    rng=np.random.default_rng(53)
    basis=np.linalg.eigh(np.eye(4)-symmetric)[1][:,1:]
    rotation=np.linalg.qr(rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)))[0]
    block=sites.on_subspace(basis@rotation)
    np.testing.assert_allclose(block.characters(),spatial.characters(),atol=1e-9)
    np.testing.assert_allclose((basis@rotation).conj().T@hamiltonian@(basis@rotation),(epsilon-t)*np.eye(3),atol=1e-14)


def test_bounds_invalid_tolerances_and_unresolved_search_fail_closed():
    group=cyclic_group(5)
    with pytest.raises(ValueError,match='max_order'):irreducible_representations(group,max_order=4)
    for option in ({'tolerance':0},{'tolerance':np.nan},{'tolerance':1j},{'tolerance':'small'},
                   {'spectral_tolerance':-1},{'max_order':False}):
        with pytest.raises(ValueError):irreducible_representations(group,**option)
    with pytest.raises(TypeError):irreducible_representations('C5')
    with pytest.raises(IrrepGenerationError):irreducible_representations(group,spectral_tolerance=2)
    with pytest.raises(IrrepGenerationError,match='ambiguous spectral gap'):
        irreducible_representations(cyclic_group(3),spectral_tolerance=.2)
