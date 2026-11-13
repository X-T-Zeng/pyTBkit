import numpy as np
import pytest
from numpy.testing import assert_allclose as close
from tbkit import models

native=pytest.importorskip('tbkit._tbkit_fortran')


def test_fortran_numba_numpy_hr_backends():
    model=models.topological_insulator(2.);k=np.array([[.2,.3,.4],[-.1,.4,.7]])
    close(model.hamiltonian(k,backend='fortran'),model.hamiltonian(k),atol=1e-14)
    close(model.gradient(k,backend='fortran'),model.gradient(k),atol=1e-14)
    close(model.bands(k,backend='fortran'),model.bands(k),atol=1e-13)
    close(model.hamiltonian(k,backend='fortran'),model.hamiltonian(k,backend='numpy'),atol=1e-14)
