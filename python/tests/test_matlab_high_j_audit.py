"""Characterize pristine high-j numerical failure, not a candidate MATLAB fix."""
import shutil
import subprocess
import numpy as np
import pytest
import sympy as sp
from scipy.io import loadmat
from reference_source import source_root


def test_pristine_factorial_overflow_and_log_racah_cancellation(tmp_path):
    octave=shutil.which('octave')
    if octave is None:pytest.skip('Octave unavailable')
    source=source_root()/'src/classes/@Y_l__m'
    for name in ('CG','Wigner3j'):
        shutil.copy2(source/(name+'.m'),tmp_path/(name+'.m'))
    (tmp_path/'reference.m').write_text("""
warning('off','all');
small=CG(10,10,10,-10,0,0);
large=CG(50,50,50,-50,0,0);
huge=CG(100,100,100,-100,0,0);
w=Wigner3j([100,100,100],[0,0,0]);
save('-mat7-binary','result.mat','small','large','huge','w');
""")
    subprocess.run([octave,'--no-gui','--quiet','reference.m'],cwd=tmp_path,capture_output=True,check=True,timeout=60)
    result=loadmat(tmp_path/'result.mat',squeeze_me=True)
    np.testing.assert_allclose(result['small'],1/np.sqrt(21),atol=1e-15,rtol=0)
    assert not np.isfinite(result['large']) and not np.isfinite(result['huge'])
    # Independent all-zero-m closed form, not the floating Racah summation.
    j=100;g=3*j//2
    exact=(-1)**g*sp.factorial(g)/sp.factorial(j//2)**3*sp.sqrt(sp.factorial(j)**3/sp.factorial(3*j+1))
    reference=float(exact.evalf(40));physical_bound=1/np.sqrt(2*j+1)
    assert abs(reference)<physical_bound
    assert abs(result['w'])>physical_bound
    assert abs(result['w']-reference)>1
