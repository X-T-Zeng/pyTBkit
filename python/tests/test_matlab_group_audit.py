"""Original group base-class identity defect vs separately patched source."""
import re
import shutil
import subprocess
import numpy as np
import pytest
from scipy.io import loadmat
from reference_source import source_root


@pytest.mark.parametrize('patched',[False,True])
def test_group_left_identity_and_power(patched,tmp_path):
    octave=shutil.which('octave')
    if octave is None:pytest.skip('Octave unavailable')
    source=(source_root(patched=patched)/'src/classes/@group/group.m').read_text()
    # Octave adapters: remove MATLAB arguments blocks, restoring only the
    # constructor's documented U=nan default; group algebra stays unchanged.
    adapted=re.sub(r'\n[ \t]*arguments\b.*?\n[ \t]*end\b','\n',source,flags=re.S)
    adapted=adapted.replace('function obj = group(U)','function obj = group(U)\n            if nargin<1;U=nan;end',1)
    (tmp_path/'group.m').write_text(adapted)
    script="""
g=group([0,-1;1,0]);e=group.identity();
result=times(e,g);first=power(g,1);
nested=isa(result.U,'group');nested_power=isa(first.U,'group');
if nested
    underlying=result.U.U;
else
    underlying=result.U;
    assert(isequal(underlying,g.U));
    assert(isequal(power(g,2).U,-eye(2)));
    assert(isequal(power(g,-1).U,inv(g.U)));
end
save('-mat7-binary','result.mat','nested','nested_power','underlying');
"""
    (tmp_path/'reference.m').write_text(script)
    subprocess.run([octave,'--no-gui','--quiet','reference.m'],cwd=tmp_path,capture_output=True,check=True,timeout=60)
    r=loadmat(tmp_path/'result.mat')
    assert bool(r['nested'].item()) is (not patched)
    assert bool(r['nested_power'].item()) is (not patched)
    # The erroneous branch stores an inverse *group object* inside U.
    expected=np.array([[0,-1],[1,0]]) if patched else np.array([[0,1],[-1,0]])
    np.testing.assert_array_equal(r['underlying'],expected)


def test_original_band_diagonals_are_not_degenerate_symmetry_eigenvalues(tmp_path):
    octave=shutil.which('octave')
    if octave is None:pytest.skip('Octave unavailable')
    source=(source_root()/'src/classes/@Oper/EIGEN_Kone.m').read_text()
    (tmp_path/'EIGEN_Kone.m').write_text(source)
    (tmp_path/'reference.m').write_text("""
D=diag([1,-1]);V=eye(2);W=eye(2);Q=[1,1;1,-1]/sqrt(2);
original=EIGEN_Kone(W,D,V);rotated=EIGEN_Kone(W*Q,D,V);
save('-mat7-binary','result.mat','original','rotated');
""")
    subprocess.run([octave,'--no-gui','--quiet','reference.m'],cwd=tmp_path,capture_output=True,check=True,timeout=60)
    r=loadmat(tmp_path/'result.mat')
    np.testing.assert_allclose(r['original'].ravel(),[1,-1],atol=1e-14)
    np.testing.assert_allclose(r['rotated'].ravel(),[0,0],atol=1e-14)
    # H=epsilon*I admits both eigenbases. Block trace=0 is invariant; the
    # individual returned expectations are not the symmetry eigenvalues +/-1.
    assert abs(np.sum(r['original'])-np.sum(r['rotated']))<1e-14
