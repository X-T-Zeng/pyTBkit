import numpy as np
import pytest
from numpy.testing import assert_allclose as close
from tbkit import models
from tbkit.topology import inversion_indicators
from tbkit.spin import SpinModel,heisenberg_quantum
from tbkit.circuit import Component,Subckt,Hckt


@pytest.mark.parametrize('mass,strong,weak,z4',[(2,1,(0,0,0),1),(0,0,(1,1,1),0),(4,0,(0,0,0),0)])
def test_inversion_indicators(mass,strong,weak,z4):
    m=models.topological_insulator(mass)
    p=np.kron(models.sigma_z,np.eye(2));t=np.kron(np.eye(2),1j*models.sigma_y)
    result=inversion_indicators(m,p,t,occupied=2)
    assert result['strong_z2']==strong and result['weak_z2']==weak and result['z4']==z4


def test_exchange_field_and_dynamics():
    rng=np.random.default_rng(2);s=rng.normal(size=(3,3));s/=np.linalg.norm(s,axis=1)[:,None]
    model=SpinModel(3,[[0,1],[1,2]],[1.,.4],field=[0,0,.2])
    energy,field=model.energy_field(s)
    for i in range(3):
        for a in range(3):
            d=np.zeros((3,3));d[i,a]=1e-6
            derivative=(model.energy_field(s+d)[0]-model.energy_field(s-d)[0])/2e-6
            close(derivative,-field[i,a],atol=1e-9)
    trajectory=model.evolve(s,.01,100,damping=.3)
    close(np.linalg.norm(trajectory,axis=-1),1,atol=1e-14)
    assert model.energy_field(trajectory[-1])[0]<energy
    close(np.linalg.eigvalsh(heisenberg_quantum(2,[[0,1]],1.).toarray()),[-.75,.25,.25,.25],atol=1e-14)


def test_rlc_circuits(tmp_path):
    r=Component('R',1000);c=Component('C',1e-6);l=Component('L',1.)
    close(r.admittance(0),.001);close(c.admittance(100),2j*np.pi*100e-6)
    close(l.admittance(100),1/(2j*np.pi*100))
    with pytest.raises(ValueError):l.admittance(0)
    circuit=Subckt(2,[(0,1,r),(1,-1,r)])
    close(circuit.voltages(0,[.001,0]),[2,1],atol=1e-14)
    circuit.to_spice(tmp_path/'network.cir')
    assert 'R2 2 0 1000' in (tmp_path/'network.cir').read_text()
    chain=Hckt(1,1,[(0,0,[1],r)],ground=[(0,c)])
    k=np.array([[.2],[.7]])
    close(chain.laplacian(k,100)[:,0,0],2*(1-np.cos(k[:,0]))/1000+2j*np.pi*100e-6,atol=1e-14)
