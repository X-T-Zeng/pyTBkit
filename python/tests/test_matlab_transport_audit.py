"""Characterize pristine official GW_iter; a pass here reproduces its defect."""
import re
import shutil
import subprocess
import numpy as np
import pytest
from scipy.io import loadmat
from reference_source import source_root


def test_original_gw_absolute_threshold_breaks_energy_covariance(tmp_path):
    octave=shutil.which('octave')
    if octave is None:pytest.skip('Octave unavailable')
    source=(source_root()/'src/classes/@TBkit/GW_iter.m').read_text()
    # Only MATLAB argument validation is removed. All six arguments are passed;
    # numerical equations, threshold and original branch logic stay unchanged.
    adapted=re.sub(r'\narguments\b.*?\nend\n','\n',source,flags=re.S)
    (tmp_path/'GW_iter.m').write_text(adapted)
    (tmp_path/'reference.m').write_text("""
scales=[1,1e-8,1e-16,1e-20];
surface=zeros(size(scales));residual=zeros(size(scales));
for i=1:length(scales)
    s=scales(i);w=.3*s;eta=1e-6*s;
    [g,gb,gr]=GW_iter(0,s,w,eta,100,1e-10);
    surface(i)=s*g;
    residual(i)=abs(((w+1i*eta)-s*g*s)*g-1);
end
save('-mat7-binary','result.mat','scales','surface','residual');
""")
    subprocess.run([octave,'--no-gui','--quiet','reference.m'],cwd=tmp_path,
                   capture_output=True,check=True,timeout=60)
    r=loadmat(tmp_path/'result.mat');g=r['surface'].ravel();residual=r['residual'].ravel()
    z=.3+1e-6j;analytic=(z-np.sqrt(z*z-4))/2
    np.testing.assert_allclose(g[0],analytic,atol=1e-10)
    assert residual[0]<1e-9
    assert abs(g[-1]-analytic)>.9
    assert residual[-1]>1.
    # The repaired Python solver is checked against the analytic root at every
    # scale in test_transport_scaling, rather than against this failed result.
