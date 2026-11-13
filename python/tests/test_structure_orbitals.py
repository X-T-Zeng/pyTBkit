import itertools
import numpy as np
import pytest
from numpy.testing import assert_allclose as close
from scipy.spatial.transform import Rotation
from tbkit.structure import Structure
from tbkit.orbitals import angular_momentum,orbital_rotation,real_orbital_angular_momentum,spin_rotation,spin_orbit_coupling
from tbkit.slater_koster import slater_koster,from_structure,PARAMETERS


@pytest.mark.parametrize('mode',['Direct','Cartesian'])
@pytest.mark.parametrize('scale',['2.0','-64.0','2.0 3.0 4.0'])
def test_poscar_scale_and_roundtrip(tmp_path,mode,scale):
    path=tmp_path/'POSCAR'
    path.write_text(f'test\n{scale}\n1 0 0\n0 1 0\n0 0 1\nC H\n1 1\nSelective dynamics\n{mode}\n0 0 0 T T F\n0.5 0.5 0.5 F T F\n')
    s=Structure.from_poscar(path)
    expected=np.array([2,3,4]) if len(scale.split())==3 else np.repeat(4 if scale.startswith('-') else 2,3)
    close(s.lattice,np.diag(expected))
    close(s.positions,[[0,0,0],[.5,.5,.5]])
    s.to_poscar(tmp_path/'again');t=Structure.from_poscar(tmp_path/'again')
    close(s.positions,t.positions);close(s.lattice,t.lattice)
    assert s.species==t.species and np.array_equal(s.selective,t.selective)


def test_vasp4_and_cartesian(tmp_path):
    p=tmp_path/'p';p.write_text('legacy\n2\n1 0 0\n0 1 0\n0 0 1\n1\nCartesian\n.2 .3 .4\n')
    s=Structure.from_poscar(p)
    assert s.species==('X1',)
    close(s.cartesian,[[.4,.6,.8]])


def test_neighbors_skew_cell_against_bruteforce():
    lattice=np.array([[1,0,0],[.8,.4,0],[.1,.2,1.5]])
    s=Structure(lattice,[[.1,.2,.3],[.7,.8,.3]],['C','C']);cutoff=.75
    n=s.neighbors(cutoff)
    actual={(int(i),int(j),tuple(r)) for i,j,r in zip(n.i,n.j,n.cells)}
    expected=set()
    for i,j,r in itertools.product(range(2),range(2),itertools.product(range(-5,6),repeat=3)):
        d=(s.positions[j]+r-s.positions[i])@lattice
        if 1e-10<np.linalg.norm(d)<=cutoff+1e-10: expected.add((i,j,r))
    assert actual==expected
    close(n.distance,np.linalg.norm(n.displacement,axis=1))
    assert len(s.neighbors(cutoff,periodic=[False]*3).distance)==0


@pytest.mark.parametrize('j',[0,.5,1,1.5,2])
def test_angular_momentum_algebra(j):
    x,y,z=angular_momentum(j)
    close(x@y-y@x,1j*z,atol=1e-14)
    close(x@x+y@y+z@z,j*(j+1)*np.eye(len(x)),atol=1e-14)
    u=spin_rotation([1,2,3],.72,spin=j)
    close(u.conj().T@u,np.eye(len(u)),atol=1e-14)


def test_real_orbitals_and_soc_spectrum():
    q=Rotation.from_rotvec([.3,-.4,.2]).as_matrix();d=orbital_rotation(q)
    close(d.T@d,np.eye(9),atol=1e-14)
    lx,ly,lz=real_orbital_angular_momentum()
    close(lx@ly-ly@lx,1j*lz,atol=1e-14)
    close(lx@lx+ly@ly+lz@lz,np.diag([0]+[2]*3+[6]*5),atol=1e-14)
    soc=spin_orbit_coupling()
    close(np.linalg.eigvalsh(soc),np.sort([0]*2+[-1]*2+[.5]*4+[-1.5]*4+[1]*6),atol=1e-14)


def test_sk_axis_reverse_and_covariance():
    p=dict(zip(PARAMETERS,np.arange(1.,11.)))
    d=np.array([.2,-.7,.8]);q=Rotation.from_rotvec([.3,.1,-.4]).as_matrix()
    t=slater_koster(d,p)
    close(slater_koster(-d,p),t.T,atol=1e-13)
    u=orbital_rotation(q)
    close(slater_koster(q@d,p),u@t@u.T,atol=1e-13)
    axis=slater_koster([0,0,1],p)
    close(axis[0,3],p['sp_sigma']);close(axis[3,3],p['pp_sigma']);close(axis[1,1],p['pp_pi'])
    close(axis[8,8],p['dd_sigma'])
    close(axis[3,8],p['pd_sigma']);close(axis[1,6],p['pd_pi'])


def test_sk_chain_model():
    s=Structure(np.diag([1.,10.,10.]),[[0,0,0]],['C'])
    m=from_structure(s,[['s']],1.01,{'ss_sigma':-1},onsite=[.2])
    k=np.array([[.2,0,0],[.7,0,0]])
    close(m.bands(k)[:,0],.2-2*np.cos(k[:,0]),atol=1e-14)
    assert m.norb==1
