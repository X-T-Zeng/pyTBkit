"""Reaffirm M-07 on a flat dimer, executing the pristine four-output body.

Only MATLAB argument validation and the function name are adapted for Octave;
no original arithmetic is patched. This is not a full MATLAB class execution.
"""
import re
import shutil
import subprocess
import numpy as np
import pytest
from scipy.io import loadmat
from numpy.testing import assert_allclose as close
from tbkit import HR,embed_orbitals
from reference_source import source_root


def run_pristine_dimer(tmp_path):
    octave=shutil.which('octave')
    if octave is None:pytest.skip('Octave unavailable')
    source=(source_root()/'src/classes/@HR/fft.m').read_text()
    source=re.sub(r'\narguments\b.*?\nend\n','\n',source,flags=re.S)
    source=source.replace('= fft(', '= hr_fft_reference(',1)
    (tmp_path/'hr_fft_reference.m').write_text(source)
    shutil.copy(__import__('pathlib').Path(__file__).parent/'matlab_fixtures/tensorprod.m',tmp_path/'tensorprod.m')
    (tmp_path/'reference.m').write_text("""Ham.HnumL=[0,1.3;1.3,0];
Ham.vectorL=[0,0,0];Ham.Rm=eye(3);Ham.Dim=3;Ham.WAN_NUM=2;Ham.NRPTS=1;
Ham.orbL_cart=[.2,0,0;.6,0,0];
[W,D,dH,velocity]=hr_fft_reference(Ham,[0,0,0],[]);
position=diag(Ham.orbL_cart(:,1));
commutator=1i*(Ham.HnumL*position-position*Ham.HnumL);
reference=W'*commutator*W;
original=velocity(:,:,1);
save('-mat7-binary','result.mat','W','D','original','reference','commutator');
""")
    subprocess.run([octave,'--no-gui','--quiet','reference.m'],cwd=tmp_path,check=True,capture_output=True,timeout=60)
    return loadmat(tmp_path/'result.mat')


def test_pristine_atomic_velocity_against_position_commutator(tmp_path):
    r=run_pristine_dimer(tmp_path)
    # Original -i(E_n-E_m)*tau_nm has the opposite sign to i[H,r].
    close(r['original'],-r['reference'],atol=3e-15)
    assert np.linalg.norm(r['original']-r['reference'])>1.
    model=HR([[0,0,0]],[[[0,1.3],[1.3,0]]],orbitals=[[.2,0,0],[.6,0,0]])
    gradient=embed_orbitals(model).gradient([0,0,0],backend='numpy')[0]
    close(gradient,r['commutator'],atol=3e-15)
    close(r['W'].conj().T@gradient@r['W'],r['reference'],atol=3e-15)
