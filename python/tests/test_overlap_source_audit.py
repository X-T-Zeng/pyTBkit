"""Characterize pristine official expressions, not full MATLAB class execution."""
import re
import shutil
import subprocess
import numpy as np
import pytest
from scipy.io import loadmat
from numpy.testing import assert_allclose as close
from reference_source import source_root


def test_pristine_overlap_projection_changes_problem_and_allows_indefinite_metric(tmp_path):
    octave = shutil.which('octave')
    if octave is None:
        pytest.skip('Octave unavailable')
    source = (source_root()/'src/classes/@HR/EIGENCAR_gen.m').read_text()
    projection = re.search(r"Sout = \(Sout\+Sout'\)/2;", source).group()
    solver = re.search(r'\[A, D\]=eig\(full\(Hout\),full\(Sout\)\);', source).group()
    # Execute exactly these two statements with minimal numeric fixtures.
    code = "Hout=diag([1,2]); Sout=[1,.4i;0,1]; original=Sout;\n"
    code += projection+'\n'+solver+'\nprojected=Sout; energy=sort(diag(D));\n'
    code += "Sout=diag([1,-1]);\n"+solver+'\nindefinite=sort(diag(D)); gram=diag(A\'*Sout*A);\n'
    code += "save('-mat7-binary','result.mat','original','projected','energy','indefinite','gram');\n"
    (tmp_path/'audit.m').write_text(code)
    subprocess.run([octave, '--no-gui', '--quiet', 'audit.m'], cwd=tmp_path,
                   check=True, capture_output=True, timeout=60)
    r = loadmat(tmp_path/'result.mat')
    close(r['projected'], [[1, .2j], [-.2j, 1]], atol=0)
    assert np.linalg.norm(r['original']-r['projected']) > .2
    # det(H-E S_projected)=.96E²-3E+2, different from triangular S: (E-1)(E-2).
    close(r['energy'].ravel(), np.sort(np.roots([.96, -3, 2])), rtol=1e-14)
    close(r['indefinite'].ravel(), [-2, 1], atol=0)
    assert np.min(r['gram'].real) < 0  # invalid Gram norm for ordinary orbital states
