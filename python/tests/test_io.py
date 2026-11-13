import numpy as np
import pytest
from numpy.testing import assert_allclose as close
from tbkit import HR,read_wannier90,write_wannier90,load_hdf5


def test_wannier_roundtrip(tmp_path):
    m=HR([[0,0,0],[1,0,0],[-1,0,0]],[np.eye(2),np.array([[1,1j],[0,2]]),np.array([[1,0],[-1j,2]])])
    p=tmp_path/'wannier90_hr.dat';write_wannier90(m,p)
    other=read_wannier90(p)
    close(other.hamiltonian([.3,.2,.1]),m.hamiltonian([.3,.2,.1]))
    m.save(tmp_path/'hr.h5'); loaded=load_hdf5(tmp_path/'hr.h5')
    close(loaded.hopping,m.hopping)
    close(loaded.lattice,m.lattice)


def test_wannier_degeneracy(tmp_path):
    p=tmp_path/'hr.dat'
    p.write_text('test\n1\n1\n2\n0 0 0 1 1 6.0 0.0\n')
    close(read_wannier90(p).hamiltonian([0,0,0]),[[3]])
    p.write_text('test\n1\n1\n0\n0 0 0 1 1 6.0 0.0\n')
    with pytest.raises(ValueError): read_wannier90(p)
