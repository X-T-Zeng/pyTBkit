"""Pristine official reciprocal expression, separately from patched classes."""
import shutil
import subprocess
import numpy as np
import pytest
from scipy.io import loadmat
from numpy.testing import assert_allclose as close
from tbkit.lattice import reciprocal_lattice
from reference_source import source_root


def test_pristine_official_reciprocal_convention_under_length_rescaling(tmp_path):
    octave=shutil.which('octave')
    if octave is None:pytest.skip('Octave unavailable')
    source=(source_root()/'src/classes/@TBkit/TBkit.m').read_text()
    statements=[line.strip() for line in source.splitlines() if line.strip().startswith('Gk = ') and 'TBkitobj.Rm' in line]
    assert len(statements)==1
    expression=statements[0]
    # Entire original arithmetic statement; no substitute inverse formula.
    code="""base=[1.3,.2,0;.1,1.1,.3;0,.1,1.6]; scales=[1e-100,1,1e100];
all_g=zeros(3,3,3);
for j=1:3
TBkitobj.Rm=base*scales(j);
"""+expression+"\nall_g(:,:,j)=Gk;\nend\nsave('-mat7-binary','result.mat','all_g');\n"
    (tmp_path/'reference.m').write_text(code)
    subprocess.run([octave,'--no-gui','--quiet','reference.m'],cwd=tmp_path,check=True,capture_output=True,timeout=60)
    official=loadmat(tmp_path/'result.mat')['all_g'].transpose(2,0,1)
    base=np.array([[1.3,.2,0],[.1,1.1,.3],[0,.1,1.6]])
    for a,g in zip((1e-100,1.,1e100),official):
        close(reciprocal_lattice(a*base)*a,g*a,rtol=2e-14,atol=2e-15)
        close((a*base)@g.T,2*np.pi*np.eye(3),rtol=0,atol=4e-15)
    # NN explicitly documents Angstroms: do not call its unit-specific Accuracy
    # a proven bug merely because Python now accepts arbitrary common units.
    nn=(source_root()/'src/classes/@TBkit/nn.m').read_text()
    assert 'Angstroms' in nn and 'Accuracy = 1e-4' in nn
