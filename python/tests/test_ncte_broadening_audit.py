"""Pin and characterize the official finite-eta prescription, do not adopt it."""
import re
import shutil
import subprocess
import numpy as np
import pytest
from scipy.io import savemat,loadmat
from numpy.testing import assert_allclose as close
from tbkit import HK,models,prepare_geometry,response_from_geometry
from tbkit.constants import KB_EV_PER_K
from reference_source import source_root


def test_official_finite_eta_is_not_real_band_curvature(tmp_path):
    octave=shutil.which('octave')
    if octave is None:pytest.skip('Octave unavailable')
    source=source_root()/'src/functions_transport'
    for name in ('NCTE_k','Fermi_0','Fermi_1'):
        text=(source/f'{name}.m').read_text()
        text=re.sub(r'\narguments\b.*?\nend\n','\n',text,flags=re.S)
        (tmp_path/f'{name}.m').write_text(text)
    sx,sy,sz=models.sigma_x,models.sigma_y,models.sigma_z
    model=HK({(1,0,0):sx+.17*np.eye(2),(0,1,0):sy,(0,0,1):sz})
    k=np.array([.2,.3,.4]);e,u=model.bands(k,eigenvectors=True);dh=model.gradient(k)
    mu=np.array([e[1]+.012]);eta=.05;temperature=300.
    savemat(tmp_path/'input.mat',dict(E=e[:,None],U=u,dH=dh.transpose(1,2,0),mu=mu,k=k,eta=eta,T=temperature))
    code="""load('input.mat');
Ham.Basis_num=length(E);Ham.fft=@(k) deal(U,E,dH);
finite=NCTE_k(Ham,[1,2,3],k,mu,T,1e-8,[1,1,-1],eta);
clean=NCTE_k(Ham,[1,2,3],k,mu,T,1e-8,[1,1,-1],0);
save('-mat7-binary','result.mat','finite','clean');
"""
    (tmp_path/'reference.m').write_text(code)
    subprocess.run([octave,'--no-gui','--quiet','reference.m'],cwd=tmp_path,check=True,capture_output=True,timeout=60)
    reference=loadmat(tmp_path/'result.mat')
    v=np.array([u.conj().T@d@u for d in dh]);delta=e[:,None]-e[None,:]
    # Independent explicit band-pair sum of the documented finite pole recipe.
    omega=np.zeros((3,3,2),complex)
    for a in range(3):
        for b in range(3):
            for n in range(2):
                for m in range(2):
                    if n!=m:omega[a,b,n]+=-2*np.imag(v[a,n,m]*v[b,m,n])/(delta[n,m]+1j*eta)**2
    velocity=v.diagonal(axis1=-2,axis2=-1).real
    band=velocity[0]*omega[1,2]+velocity[1]*omega[2,0]-velocity[2]*omega[0,1]
    beta=1/(8.6173324e-5*temperature);f=1/(1+np.exp((e-mu[0])*beta))
    expected=np.sum(band*(-beta*f*(1-f))*(e-mu[0]))
    close(reference['finite'].item(),expected,atol=1e-12)
    # Berry curvature of normalized Hermitian eigenvectors is real, and the
    # sum over a complete orbital frame is zero. The finite recipe fails both.
    assert np.max(abs(omega.imag))>1e-3
    assert np.max(abs(omega.sum(axis=-1)))>1e-3
    assert abs(reference['finite'].item().imag)>1e-5
    geometry=prepare_geometry(model,k)
    close(geometry.berry.sum(axis=-1),0,atol=1e-13)
    temperature_python=temperature*8.6173324e-5/KB_EV_PER_K
    python=response_from_geometry(geometry,mu,kind='ncte',indices=(0,1,2),temperature=temperature_python)
    close(python,reference['clean'].ravel(),atol=1e-12)
    assert np.isrealobj(python)
