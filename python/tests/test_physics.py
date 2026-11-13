import numpy as np
import pytest
from numpy.testing import assert_allclose as close
from tbkit import models, berry_curvature, chern_number, wilson_loop, kmesh, fermi, surface_green, transmission, fit_bands


def test_dirac_berry():
    m=models.dirac(mass=.4); k=np.array([[.1,.2],[.3,-.4]])
    e=np.sqrt((k*k).sum(axis=1)+.4**2)
    expected=.4/(2*e**3)
    close(berry_curvature(m,k),np.stack([expected,-expected],axis=1),atol=1e-12)

@pytest.mark.parametrize('mass,expected',[(-3,0),(-1,1),(1,-1),(3,0)])
def test_chern(mass,expected):
    m=models.qi_wu_zhang(mass)
    assert abs(chern_number(m,(25,25))-expected)<1e-12
    k=kmesh([60,60]); integrated=berry_curvature(m,k)[:,0].mean()*2*np.pi
    assert abs(integrated-expected)<1e-6


def test_gap_closing_rejected():
    with pytest.raises(ValueError): berry_curvature(models.dirac(mass=0),[0,0])
    with pytest.raises(ValueError): chern_number(models.qi_wu_zhang(-2),(20,20))


def test_ssh_wilson():
    phase=wilson_loop(models.ssh(),kmesh([80]))[0]
    assert abs(abs(phase)-np.pi)<1e-12
    assert abs(wilson_loop(models.ssh(2,1),kmesh([80]))[0])<1e-12


def test_fermi_stable():
    close(fermi([-1e5,0,1e5],temperature=300),[1,.5,0])
    close(fermi([-1,0,1]),[1,.5,0])


def test_chain_surface_and_transmission():
    energy=.3; eta=1e-5; z=energy+1j*eta
    analytic=(z-np.sqrt(z*z-4))/2
    g=surface_green(energy,[[0]],[[1]],eta=eta)
    close(g,[[analytic]],atol=1e-11)
    close(g,1/(z-g),atol=1e-11)
    assert abs(transmission(energy,[[0]],g,g)-1)<1e-8
    with pytest.raises(RuntimeError): surface_green(energy,[[0]],[[1]],maxiter=1)


def test_band_fit():
    k=np.linspace(0,np.pi,20)[:,None]
    target=models.ssh(.6,1.2).bands(k)
    result=fit_bands(lambda p:models.ssh(*p),[.5,1.1],k,target,bounds=(0,2))
    assert result.success
    close(result.x,[.6,1.2],atol=1e-7)


@pytest.mark.parametrize('energy',[-3.,-1.,0.,.3,1.,3.])
def test_surface_center_and_outside_band(energy):
    z=energy+1j*1e-6
    roots=[(z+np.sqrt(z*z-4))/2,(z-np.sqrt(z*z-4))/2]
    analytic=next(x for x in roots if x.imag<0)
    close(surface_green(energy,[[0]],[[1]])[0,0],analytic,atol=1e-10)


def test_long_range_lead_grouping_and_unit_transmission():
    from tbkit import HR
    from tbkit.transport import principal_layer,chain_transmission
    model=HR([[0],[1],[-1],[2],[-2]],[[[0]],[[1]],[[1]],[[.2]],[[.2]]])
    lead=principal_layer(model)
    assert lead.width==2 and lead.onsite.shape==(2,2)
    for q in [.13,.41]:
        represented=lead.onsite+lead.hopping*np.exp(2j*np.pi*q)+lead.hopping.conj().T*np.exp(-2j*np.pi*q)
        close(represented,model.supercell([2]).hamiltonian([np.pi*q]),atol=1e-13)
    assert abs(chain_transmission(lead,.3,layers=3)-1)<1e-6
