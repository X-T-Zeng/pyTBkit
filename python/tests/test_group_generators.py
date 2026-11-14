"""Global cardinality minima, not only greedy irredundant generating sets."""
from dataclasses import FrozenInstanceError
from itertools import combinations
from math import gcd
import numpy as np
import pytest
from numba import njit
from sympy.combinatorics import Permutation,PermutationGroup
from tbkit import FiniteGroup,cyclic_group,dihedral_group,permutation_group,GroupSearchLimitError
from tbkit.group_kernels import generated_subgroup_indices


def elementary_abelian(rank):
    indices=np.arange(2**rank)
    return FiniteGroup(np.bitwise_xor(indices[:,None],indices),name=f'C2^{rank}')


def brute_minimum(group):
    # Independent exhaustive subsets and a set-based closure, without the new
    # subgroup BFS or its Numba closure. Used only for small oracle cases.
    def closure(gens):
        known={group.identity}
        while True:
            expanded=known|{group.multiply(a,b) for a in known for b in gens}
            if expanded==known:return known
            known=expanded
    for size in range(group.order+1):
        for gens in combinations(group.elements,size):
            if len(closure(gens))==group.order:return size
    raise AssertionError('oracle failed')


@pytest.mark.parametrize('n',[1,2,3,6,12])
def test_cyclic_minimum_and_empty_set_convention(n):
    group=cyclic_group(n);result=group.generator_search()
    assert result.rank==(0 if n==1 else 1)
    assert group.is_generating(result.generators)
    assert group.rank()==result.rank
    assert group.minimum_generators()==result.generators
    assert group.is_generating([]) is (n==1)
    for a in group:
        actual=group.generated_subgroup([a])
        assert actual.order==n//gcd(n,a)
    if n==1:
        assert result.generators==() and result.extensions==0
        assert group.rank(max_extensions=0)==0
    with pytest.raises(FrozenInstanceError):result.extensions=0


def test_irredundant_is_not_global_minimum():
    group=cyclic_group(6)
    assert group.is_generating([2,3])
    assert not group.is_generating([2]) and not group.is_generating([3])
    assert group.minimum_generators()==(1,)
    assert group.rank()==1


@pytest.mark.parametrize('group',[dihedral_group(3),dihedral_group(4),dihedral_group(5),elementary_abelian(2)])
def test_small_group_minimum_matches_independent_exhaustion(group):
    result=group.generator_search()
    assert result.rank==brute_minimum(group)==2
    assert group.is_generating(result.generators)
    assert len(set(result.generators))==result.rank


@pytest.mark.parametrize('rank',[3,4])
def test_elementary_abelian_lower_bound_from_vector_space_dimension(rank):
    group=elementary_abelian(rank)
    # k vectors in F_2^rank span at most 2**k elements, independently proving
    # the lower bound. Verify the returned vectors have full GF(2) row rank.
    result=group.generator_search()
    assert result.rank==rank
    rows=list(result.generators);pivots=0
    for bit in range(rank-1,-1,-1):
        found=next((i for i in range(pivots,len(rows)) if rows[i]&(1<<bit)),None)
        if found is None:continue
        rows[pivots],rows[found]=rows[found],rows[pivots]
        for i in range(len(rows)):
            if i!=pivots and rows[i]&(1<<bit):rows[i]^=rows[pivots]
        pivots+=1
    assert pivots==rank
    assert group.is_generating(result.generators)


def test_nonabelian_permutation_group_s4_and_sympy_oracle():
    group=permutation_group([(1,2,3,0),(1,0,2,3)])
    oracle=PermutationGroup(Permutation(0,1,2,3),Permutation(0,1))
    assert group.order==oracle.order()==24
    result=group.generator_search()
    # A nonabelian group is not cyclic; the returned two matrices prove upper
    # bound two. SymPy independently verifies their generated group order.
    assert not group.is_abelian and result.rank==2
    independent=PermutationGroup(*[Permutation(list(g)) for g in result.generators])
    assert independent.order()==24


def test_rank_invariant_under_relabelling_with_identity_not_first():
    original=dihedral_group(3)
    order=np.array([3,4,0,2,1,5]);back=np.argsort(order)
    table=back[original.table[np.ix_(order,order)]]
    group=FiniteGroup(table,[f'label_{i}' for i in order])
    assert group.identity=='label_0'
    result=group.generator_search()
    assert result.rank==2 and group.is_generating(result.generators)
    # Witness ordering is deterministic for this label order, not canonical
    # across isomorphisms; the rank itself is invariant.
    assert result==group.generator_search()


def test_budget_exhaustion_raises_and_only_reports_a_valid_lower_bound():
    group=elementary_abelian(3)
    with pytest.raises(GroupSearchLimitError) as no_work:
        group.rank(max_extensions=0)
    assert no_work.value.limit=='max_extensions'
    assert no_work.value.extensions==0 and no_work.value.lower_bound==1
    with pytest.raises(GroupSearchLimitError) as little_memory:
        group.minimum_generators(max_subgroups=1)
    assert little_memory.value.limit=='max_subgroups'
    assert little_memory.value.subgroup_states==1
    assert 1<=little_memory.value.lower_bound<=3
    with pytest.raises(GroupSearchLimitError) as deeper:
        group.generator_search(max_extensions=7)
    assert deeper.value.lower_bound==2  # all single elements were checked
    assert deeper.value.extensions==7
    for kwargs in ({'max_subgroups':0},{'max_extensions':-1},{'max_extensions':1.5}):
        with pytest.raises(ValueError):group.rank(**kwargs)
    assert group.rank()==3


def test_numba_closure_kernel_user_pipeline_and_nonzero_identity():
    group=cyclic_group(12)
    @njit
    def user_pipeline(table,gens,identity):
        indices=generated_subgroup_indices(table,gens,identity)
        return indices,len(indices)
    indices,count=user_pipeline(group.table,np.array([4,6],dtype=np.int64),group.index(group.identity))
    np.testing.assert_array_equal(indices,[0,2,4,6,8,10]);assert count==6
    assert user_pipeline.nopython_signatures
    order=np.array([1,0,2]);back=np.argsort(order)
    table=back[cyclic_group(3).table[np.ix_(order,order)]]
    relabelled=FiniteGroup(table,['a','e','b'])
    empty=user_pipeline(relabelled.table,np.empty(0,dtype=np.int64),relabelled.index('e'))[0]
    np.testing.assert_array_equal(empty,[1])
    assert relabelled.generated_subgroup([]).elements==('e',)
