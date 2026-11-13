"""Execute the numerical section of the repository's actual HR/fft.m in Octave.

The MATLAB class/arguments/persistent initialization is bypassed, not ported.
This compares its Fourier, LAPACK eigenvalues, eigenvectors and derivative
formula against Python on a 3D Hermitian multi-hopping fixture.
"""
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from scipy.io import savemat,loadmat
from tbkit import HR
from reference_source import source_root


def test_original_fft_numerical_body(tmp_path):
    octave=shutil.which('octave')
    if octave is None: pytest.skip('Octave unavailable')
    root=source_root()
    source=(root/'src/classes/@HR/fft.m').read_text()
    # Execute the actual function with only MATLAB argument validation removed.
    import re
    source=re.sub(r'\narguments\b.*?\nend\n','\n',source,flags=re.S)
    source=source.replace('= fft(', '= hr_fft_reference(',1)
    (tmp_path/'hr_fft_reference.m').write_text(source)
    rng=np.random.default_rng(2)
    r=np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,1],[0,-1,-1]])
    h=rng.normal(size=(5,3,3))+1j*rng.normal(size=(5,3,3))
    model=HR(r,h,lattice=np.diag([2.,3.,4.])).hermitize()
    k=np.array([.23,.14,-.35])
    savemat(tmp_path/'input.mat',dict(HnumList=model.hopping.transpose(1,2,0),vectors=model.vectors,lattice=model.lattice,klist_cart=k))
    script="""load('input.mat');
Ham.HnumL=HnumList;Ham.vectorL=vectors;Ham.Rm=lattice;Ham.Dim=3;Ham.WAN_NUM=size(HnumList,1);Ham.NRPTS=size(vectors,1);
[W,D,dH_dk_R]=hr_fft_reference(Ham,klist_cart,[]);
Hk=W*diag(D)*W';
save('-mat7-binary','result.mat','Hk','D','W','dH_dk_R');
"""
    (tmp_path/'reference.m').write_text(script)
    subprocess.run([octave,'--no-gui','--quiet','reference.m'],cwd=tmp_path,check=True,capture_output=True,timeout=60)
    result=loadmat(tmp_path/'result.mat')
    np.testing.assert_allclose(model.hamiltonian(k),result['Hk'],atol=1e-13)
    np.testing.assert_allclose(model.bands(k),np.sort(result['D'].ravel()),atol=1e-13)
    np.testing.assert_allclose(model.gradient(k),result['dH_dk_R'].transpose(2,0,1),atol=1e-13)


def test_original_response_functions(tmp_path):
    import re
    from tbkit import HK,models
    from tbkit.responses import geometric_tensors,response_kernel
    octave=shutil.which('octave')
    if octave is None: pytest.skip('Octave unavailable')
    root=source_root()
    source=root/'src/functions_transport'
    # Remove only MATLAB argument validation (unsupported in Octave), preserve
    # all equations, branch logic and the original Fermi functions.
    for name in ['BCP_k','NCTE_k','SOAHC_int_k','Fermi_0','Fermi_1','Fermi_2']:
        text=(source/f'{name}.m').read_text()
        text=re.sub(r'\narguments\b.*?\nend\n','\n',text,flags=re.S)
        (tmp_path/f'{name}.m').write_text(text)
    (tmp_path/'tensorprod.m').write_text('function out=tensorprod(A,B,a,b)\nassert(a==1 && b==1);out=A(:).\'*B;\nend\n')
    m=HK({(0,0,0):.4*models.sigma_z,(1,0,0):models.sigma_x,(0,1,0):models.sigma_y,(0,0,1):.2*models.sigma_z+.1*np.eye(2)})
    k=np.array([.2,-.3,.4]);e,u=m.bands(k,eigenvectors=True);dh=m.gradient(k)
    mu=np.array([-.5,.1,.6]);temperature=300.
    savemat(tmp_path/'input.mat',dict(E=e[:,None],U=u,dH=dh.transpose(1,2,0),mu=mu,T=temperature,k=k))
    code="""load('input.mat');
Ham.Basis_num=length(E);Ham.fft=@(k) deal(U,E,dH);
bcp=BCP_k(Ham,[1,2],k,1:length(E),1e-8);
ncte=NCTE_k(Ham,[1,2,3],k,mu,T,1e-8,[1,1,-1],0);
soahc=SOAHC_int_k(Ham,[1,2,3],k,mu,T,1e-8);
save('-mat7-binary','result.mat','bcp','ncte','soahc');
"""
    (tmp_path/'reference.m').write_text(code)
    subprocess.run([octave,'--no-gui','--quiet','reference.m'],cwd=tmp_path,check=True,capture_output=True,timeout=60)
    r=loadmat(tmp_path/'result.mat')
    np.testing.assert_allclose(geometric_tensors(m,k)['bcp'][0,1],r['bcp'].ravel(),atol=1e-12)
    # Match the original thermal energy k_B*T; Python uses the current CODATA constant.
    for name in ('ncte','soahc'):
        np.testing.assert_allclose(response_kernel(m,k,mu,kind=name,indices=(0,1,2),temperature=temperature*8.6173324e-5/8.617333262145e-5),r[name].ravel(),atol=1e-12)
