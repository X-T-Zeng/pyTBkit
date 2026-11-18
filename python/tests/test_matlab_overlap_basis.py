"""Pristine selected MATLAB statements; full HR class is outside this test."""
import re
import shutil
import subprocess
import numpy as np
import pytest
from scipy.io import loadmat
from numpy.testing import assert_allclose as close
from reference_source import source_root


def test_official_umat_keeps_old_overlap_and_changes_generalized_spectrum(tmp_path):
    octave=shutil.which('octave')
    if octave is None:pytest.skip('Octave unavailable')
    source=(source_root()/'src/classes/@HR/EIGENCAR_gen.m').read_text()
    block=re.search(r'        if UmatMode\n            Hout = Umat\\Hout\*Umat;.*?\n        end',source,re.S).group()
    assert 'Sout' not in block
    solver=re.search(r'\[A, D\]=eig\(full\(Hout\),full\(Sout\)\);',source).group()
    script="""Hout=diag([1,3]); Sout=diag([1,2]); Umat=[0,1;1,0];
UmatMode=true; subband=[1,2]; Hermite=false;
"""+solver+"\ninitial=sort(diag(D));\n"+block+'\n'+solver+"\noriginal_path=sort(diag(D));\n"
    # Independent physical reference: BOTH quadratic forms transform.
    script+="Sout=Umat'*Sout*Umat;\n"+solver+"\ncongruence=sort(diag(D));\n"
    script+="save('-mat7-binary','result.mat','initial','original_path','congruence');\n"
    (tmp_path/'audit.m').write_text(script)
    subprocess.run([octave,'--no-gui','--quiet','audit.m'],cwd=tmp_path,check=True,capture_output=True,timeout=60)
    r=loadmat(tmp_path/'result.mat')
    close(r['initial'].ravel(),[1.,1.5],atol=0)
    close(r['original_path'].ravel(),[.5,3.],atol=0)
    close(r['congruence'].ravel(),[1.,1.5],atol=0)
