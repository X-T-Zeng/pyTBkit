import numpy as np
import pytest
import sympy as sp
from numpy.testing import assert_allclose as close
from tbkit.quantum import Qnum,BasisFunc,spherical_harmonic,coupled_basis
from tbkit.orbitals import angular_momentum


def test_harmonic_normalization_and_real_p_orbitals():
    t,p=sp.symbols('theta phi',real=True)
    y=spherical_harmonic(1,0,t,p)
    assert sp.simplify(y-sp.sqrt(3/(4*sp.pi))*sp.cos(t))==0
    for m in (-1,0,1):
        real=spherical_harmonic(1,m,t,p,real=True)
        norm=sp.integrate(sp.integrate(sp.expand_trig(real**2)*sp.sin(t),(p,0,2*sp.pi)),(t,0,sp.pi))
        assert sp.simplify(norm-1)==0


def test_qnum_basis_rotation_and_inner():
    state=BasisFunc({Qnum(1,1):1})
    assert state.inner(state)==1
    rotated=state.rotate([0,0,1],.4)
    close(complex(rotated.terms[Qnum(1,1)]),np.exp(-.4j),atol=1e-14)
    close(complex(rotated.inner(rotated)),1,atol=1e-14)
    assert (state+(-1)*state).inner(state)==0
    with pytest.raises(ValueError):Qnum(1,2)


def test_clebsch_gordan_diagonalizes_ls():
    c,labels,_=coupled_basis(1,.5)
    assert sp.simplify(c*c.T)==sp.eye(6)
    numeric=np.array(c).astype(complex)
    ls=sum(np.kron(l,s) for l,s in zip(angular_momentum(1),angular_momentum(.5)))
    expected=[float((j*(j+1)-2-sp.Rational(3,4))/2) for j,m in labels]
    close(numeric@ls@numeric.conj().T,np.diag(expected),atol=1e-14)
