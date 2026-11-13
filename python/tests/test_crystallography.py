import numpy as np
import pytest
from numpy.testing import assert_allclose as close
from tbkit.structure import Structure
from tbkit.crystallography import spacegroup,crystal_operations,standardize

spglib=pytest.importorskip('spglib')


def test_cubic_spacegroup_and_mappings():
    structure=Structure(np.eye(3),[[0,0,0]],['X'])
    data=spacegroup(structure)
    assert data['number']==221 and data['international']=='Pm-3m'
    operations=crystal_operations(structure);assert len(operations)==48
    for op in operations:
        close(op.cartesian_rotation.T@op.cartesian_rotation,np.eye(3),atol=1e-12)
        close(structure.positions@op.rotation.T+op.translation,structure.positions[op.permutation]+op.cell_shifts,atol=1e-12)


def test_diamond_primitive_and_conventional():
    structure=Structure(np.array([[0,.5,.5],[.5,0,.5],[.5,.5,0]])*5.43,[[0,0,0],[.25,.25,.25]],['Si','Si'])
    assert spacegroup(structure)['number']==227
    conventional=standardize(structure);assert len(conventional.positions)==8
    primitive=standardize(conventional,primitive=True);assert len(primitive.positions)==2
    close(conventional.volume/primitive.volume,4,atol=1e-12)
    for op in crystal_operations(structure):
        close(structure.positions@op.rotation.T+op.translation,structure.positions[op.permutation]+op.cell_shifts,atol=1e-12)


def test_diamond_nonsymmorphic_sewing_and_symmetrization():
    from tbkit.crystallography import sewing_matrix,crystal_residual,symmetrize_hr
    from tbkit.slater_koster import from_structure
    structure=Structure(np.array([[0,.5,.5],[.5,0,.5],[.5,.5,0]])*5.43,[[0,0,0],[.25,.25,.25]],['Si','Si'])
    orbitals=[['s'],['s']]
    m=from_structure(structure,orbitals,2.4,{'ss_sigma':-1.})
    operations=crystal_operations(structure)
    assert any(np.linalg.norm(op.translation)>1e-4 for op in operations)
    k=np.array([[.137,.247,.317],[-.13,.24,.31]])
    for op in operations:
        u=sewing_matrix(structure,op,orbitals,k)
        close(u.conj().swapaxes(-1,-2)@u,np.repeat(np.eye(2)[None],len(k),axis=0),atol=1e-12)
        assert crystal_residual(m,structure,op,orbitals,k)<1e-12
    broken=m.with_hopping([0,0,0],np.diag([.2,-.2]),add=True)
    projected=symmetrize_hr(broken,structure,operations,orbitals)
    for op in operations:assert crystal_residual(projected,structure,op,orbitals,k)<1e-12
    close(projected.hamiltonian([.2,.3,.4]),m.hamiltonian([.2,.3,.4]),atol=1e-12)


def test_spinful_real_orbital_sewing():
    from tbkit import HR
    from tbkit.crystallography import crystal_residual
    from tbkit.orbitals import spin_orbit_coupling,ORBITALS
    structure=Structure(np.eye(3),[[0,0,0]],['X'])
    m=HR([[0,0,0]],[spin_orbit_coupling()],lattice=structure.lattice)
    for op in crystal_operations(structure)[::5]:
        assert crystal_residual(m,structure,op,[ORBITALS],[[.1,.2,.3]],spinful=True)<1e-12


def test_crystal_projection_does_not_erase_small_physical_energy():
    from tbkit import HR
    from tbkit.crystallography import symmetrize_hr
    structure=Structure(np.eye(3),[[0,0,0]],['X'])
    m=HR([[0,0,0]],[[[1e-20]]])
    projected=symmetrize_hr(m,structure,crystal_operations(structure),[['s']])
    close(projected.hamiltonian([.1,.2,.3]),[[1e-20]],rtol=1e-12,atol=0)
