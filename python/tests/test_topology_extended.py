import numpy as np
import pytest
from numpy.testing import assert_allclose as close
from tbkit import models,HR
from tbkit.construction import direct_sum
from tbkit.topology import mirror_chern,nested_polarization,wilson_flow


def test_mirror_chern_decoupled_sectors():
    a=models.qi_wu_zhang(-1);b=models.qi_wu_zhang(1)
    m=direct_sum(a,b);mirror=np.diag([1j,1j,-1j,-1j])
    close(mirror_chern(m,mirror,shape=(17,17)),[1,-1,1],atol=1e-12)
    mixed=m.with_hopping([0,0],np.ones((4,4))*.1,add=True)
    with pytest.raises(ValueError):mirror_chern(mixed,mirror,shape=(7,7))


@pytest.mark.parametrize('gamma,expected',[(.5,.5),(1.5,0.)])
def test_bbh_nested_polarization(gamma,expected):
    m=models.bbh(gamma)
    p=nested_polarization(m,(21,21))
    assert abs(np.angle(np.exp(2j*np.pi*(p-expected))))<1e-9
    x,phases=wilson_flow(m,(17,19),occupied=2)
    assert phases.shape==(19,2)
    close(phases[:,0],-phases[:,1],atol=1e-12)


@pytest.mark.parametrize('mass,mixing,expected',[(-1,0,1),(-1,.17,1),(3,0,0),(3,.17,0)])
def test_z2_wilson_without_inversion(mass,mixing,expected):
    from tbkit.topology import z2_wilson
    m=models.quantum_spin_hall(mass,mixing)
    t=np.kron(1j*models.sigma_y,np.eye(2))
    result=z2_wilson(m,t,shape=(31,31),diagnostics=True)
    assert result['z2']==expected
    assert result['phases'].shape==(31,2)
