"""Execute patched MATLAB numerical code via Octave with minimal data adapters."""
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
import pytest
from scipy.io import savemat,loadmat
from numpy.testing import assert_allclose as close
from tbkit import HR,HK,Oper,models
from reference_source import source_root

ROOT=Path(__file__).resolve().parents[2]


def install_function(folder,source,newname):
    text=(source_root(patched=True)/source).read_text()
    text=re.sub(r'\narguments\b.*?\nend\n','\n',text,flags=re.S)
    text=re.sub(r'(?m)^(function\s+.*?=\s*)\w+(\s*\()',r'\g<1>'+newname+r'\2',text,count=1)
    (folder/(newname+'.m')).write_text(text)


def run_octave(folder,code):
    octave=shutil.which('octave')
    if octave is None:pytest.skip('Octave unavailable')
    (folder/'run_audit.m').write_text(code)
    p=subprocess.run([octave,'--no-gui','--quiet','run_audit.m'],cwd=folder,text=True,capture_output=True,timeout=60)
    assert p.returncode==0,p.stdout+p.stderr
    return loadmat(folder/'result.mat')


def test_matlab_hr_cache_isolation_updates_frames_and_nonhermiticity(tmp_path):
    install_function(tmp_path,'src/classes/@HR/fft.m','hr_fft')
    r=run_octave(tmp_path,"""
a.HnumL=reshape([1,3],2,1);a.HnumL=diag([1,3]);
a.vectorL=[0,0,0];a.Rm=eye(3);a.Dim=3;a.WAN_NUM=2;
b=a;b.HnumL=diag([4,8]);
[Wa,Ea,Da]=hr_fft(a,[0,0,0],[]);
[Wb,Eb,Db]=hr_fft(b,[0,0,0],[]);
a.HnumL=diag([9,12]);[Wu,Eu,Du]=hr_fft(a,[0,0,0],[]);
c=a;c.WAN_NUM=1;c.vectorL=[1,0,0;-1,0,0];c.HnumL=reshape([1,1],1,1,2);
[Wc,Ec,Dc]=hr_fft(c,[.3,0,0],[]);
rotation=[0,1,0;-1,0,0;0,0,1];
[Wr,Er,Dr]=hr_fft(c,[.3,0,0],rotation);
rejected=false;bad=c;bad.vectorL=[1,0,0];bad.HnumL=1;
try;hr_fft(bad,[.3,0,0],[]);catch err;rejected=strcmp(err.identifier,'TBkit:NonHermitian');end;
save('-mat7-binary','result.mat','Ea','Eb','Eu','Ec','Dc','Dr','rejected');
""")
    close(r['Ea'].ravel(),[1,3]);close(r['Eb'].ravel(),[4,8]);close(r['Eu'].ravel(),[9,12])
    close(r['Ec'],[[2*np.cos(.3)]],atol=1e-14)
    close(r['Dc'].ravel(),[-2*np.sin(.3),0,0],atol=1e-14)
    close(r['Dr'].ravel(),[0,2*np.sin(.3),0],atol=1e-14)
    assert r['rejected'].item()


def test_matlab_applyru_numeric_internal_tr_and_particle_hole(tmp_path):
    shutil.copy(ROOT/'python/tests/matlab_fixtures/HK.m',tmp_path/'HK.m')
    install_function(tmp_path,'src/classes/@HK/applyRU.m','applyRU')
    install_function(tmp_path,'src/classes/@HK/applyU.m','applyU')
    # pagemtimes adapter is just batch matrix multiplication, not transformation logic.
    (tmp_path/'pagemtimes.m').write_text('function out=pagemtimes(a,b)\nnp=max(size(a,3),size(b,3));out=zeros(size(a,1),size(b,2),np);for i=1:np;out(:,:,i)=a(:,:,min(i,size(a,3)))*b(:,:,min(i,size(b,3)));end;end\n')
    sx,sy,sz=models.sigma_x,models.sigma_y,models.sigma_z
    coeff=np.stack([sz+.3*sx,sx,sy,.4*sz],axis=-1)
    savemat(tmp_path/'input.mat',dict(coeff=coeff,U=1j*sy,C=sx))
    result=run_octave(tmp_path,"""
load('input.mat');h=HK(coeff);
internal.R=eye(3);internal.U=U;internal.conjugate=false;internal.antisymmetry=false;
[hi,unused]=applyRU(h,internal);ci=hi.HnumL;
tr=internal;tr.conjugate=true;
[ht,unused]=applyRU(h,tr);ct=ht.HnumL;
hu=applyU(h,U,true,false);cu=hu.HnumL;
ph=tr;ph.U=C;ph.antisymmetry=true;
[hp,unused]=applyRU(h,ph);cp=hp.HnumL;
save('-mat7-binary','result.mat','ci','ct','cp','cu');
""")
    close(result['cu'],result['ct'],atol=1e-13)
    original=HK({(0,0,0):coeff[:,:,0],(1,0,0):coeff[:,:,1],(0,1,0):coeff[:,:,2],(0,0,1):coeff[:,:,3]})
    k=np.random.default_rng(44).normal(size=(13,3))
    ops=[Oper.from_spatial(1j*sy,np.eye(3)),Oper.from_spatial(1j*sy,np.eye(3),antiunitary=True),Oper.from_spatial(sx,np.eye(3),antiunitary=True,antisymmetry=True)]
    for key,op in zip(('ci','ct','cp'),ops):
        transformed=np.einsum('kp,ijp->kij',np.c_[np.ones(len(k)),k],result[key])
        close(transformed,op.apply(original).hamiltonian(k,backend='numpy'),atol=1e-13)


def test_matlab_fermi_zero_temperature_and_distribution_derivatives(tmp_path):
    for name in ('Fermi_0','Fermi_1','Fermi_2'):
        shutil.copy(source_root(patched=True)/'src/functions_transport'/f'{name}.m',tmp_path/f'{name}.m')
    result=run_octave(tmp_path,"""
z=Fermi_0([-1,0,1],0);hot=Fermi_0([-1e6,0,1e6],300);
first=false;second=false;
try;Fermi_1([0],0);catch err;first=strcmp(err.identifier,'TBkit:InvalidTemperature');end;
try;Fermi_2([0],0);catch err;second=strcmp(err.identifier,'TBkit:InvalidTemperature');end;
save('-mat7-binary','result.mat','z','hot','first','second');
""")
    close(result['z'],[[1,.5,0]]);close(result['hot'],[[1,.5,0]])
    assert result['first'].item() and result['second'].item()


def test_original_bhz_tutorial_equations_and_generated_space(tmp_path):
    # Extract the six original Term scalar/matrix expressions, execute their
    # actual Pauli tensor-product implementation, not a recoded model formula.
    from tbkit import invariant_kp
    from test_symmetry_construction import bhz_generators,coefficient_vectors
    from scipy.linalg import orth
    source=(source_root()/'example/example06-HK/HK_symmetry/Tutorial_HK_under_symmetry.m').read_text()
    section=source.split('BHZ = HK(4,3)',1)[1].split('BHZ_PT = HK',1)[0]
    expressions=re.findall(r'Term\(([^,]+),([^\)]+)\)',section)
    assert len(expressions)==6
    shutil.copy(source_root()/'src/classes/@pauli_matrix/pauli_matrix.m',tmp_path/'pauli_matrix.m')
    terms='\n'.join(f'H=H+({scalar.strip()})*double({matrix.strip()});' for scalar,matrix in expressions)
    rng=np.random.default_rng(71);k=rng.normal(size=(19,3))
    savemat(tmp_path/'input.mat',dict(k=k))
    script="""load('input.mat');
sigma_0=pauli_matrix(0);sigma_x=pauli_matrix(1);sigma_y=pauli_matrix(2);sigma_z=pauli_matrix(3);
tau_0=pauli_matrix(0);tau_x=pauli_matrix(1);tau_y=pauli_matrix(2);tau_z=pauli_matrix(3);
C0=.13;C1=.2;C2=-.3;M0=.4;M1=.7;M2=.9;A=1.2;B=0;
matrices=zeros(4,4,size(k,1));
for point=1:size(k,1)
k_x=k(point,1);k_y=k(point,2);k_z=k(point,3);
M=M0-M2*(k_x^2+k_y^2)-M1*k_z^2;E0k=C0+C2*(k_x^2+k_y^2)+C1*k_z^2;
H=zeros(4);
"""+terms+"\nmatrices(:,:,point)=H;\nend\nsave('-mat7-binary','result.mat','matrices');\n"
    result=run_octave(tmp_path,script)['matrices'].transpose(2,0,1)
    i,sx,sy,sz=np.eye(2),models.sigma_x,models.sigma_y,models.sigma_z
    z=np.kron(i,sz)
    h=HK({(0,0,0):.13*np.eye(4)+.4*z,(0,0,2):.2*np.eye(4)-.7*z,
          (2,0,0):-.3*np.eye(4)-.9*z,(0,2,0):-.3*np.eye(4)-.9*z,
          (1,0,0):1.2*np.kron(sz,sx),(0,1,0):1.2*np.kron(i,sy)})
    close(result,h.hamiltonian(k),atol=1e-13)
    close(np.linalg.eigvalsh(result),h.bands(k),atol=1e-12)
    basis=invariant_kp(4,3,2,bhz_generators(),method='numeric')
    powers=sorted(set(tuple(p) for item in basis+[h] for p in item.powers))
    q=orth(coefficient_vectors(basis,powers));v=coefficient_vectors([h],powers)
    close(q@(q.T@v),v,atol=1e-11)


def test_matlab_kp_fft_does_not_hide_nonhermitian_family(tmp_path):
    install_function(tmp_path,'src/classes/@HK/fft.m','kp_fft')
    result=run_octave(tmp_path,"""
h.Hfun=@(x,y,z) [1,x-1i*y;x+1i*y,-1];
h.dH_dk_fun=@(x,y,z) cat(3,[0,1;1,0],[0,-1i;1i,0],zeros(2));
[w,e,d]=kp_fft(h,[.2,.3,0],[]);
bad=h;bad.Hfun=@(x,y,z) [1,1;0,-1];rejected=false;
try;kp_fft(bad,[0,0,0],[]);catch err;rejected=strcmp(err.identifier,'TBkit:NonHermitian');end;
save('-mat7-binary','result.mat','e','rejected');
""")
    close(result['e'].ravel(),[-np.sqrt(1.13),np.sqrt(1.13)],atol=1e-14)
    assert result['rejected'].item()


def test_matlab_fermi_derivatives_retain_both_tails(tmp_path):
    for name in ('Fermi_0','Fermi_1','Fermi_2'):
        shutil.copy(source_root(patched=True)/'src/functions_transport'/f'{name}.m',tmp_path/f'{name}.m')
    result=run_octave(tmp_path,"""
e=[-100,-40,0,40,100]*8.617333262145e-5*300;
f1=Fermi_1(e,300);f2=Fermi_2(e,300);
save('-mat7-binary','result.mat','f1','f2');
""")
    first=result['f1'].ravel();second=result['f2'].ravel()
    close(first,first[::-1],rtol=1e-14,atol=0);close(second,-second[::-1],rtol=1e-14,atol=0)
    assert first[0]<0 and second[0]<0


def test_official_velocity_position_sign_against_real_space_commutator(tmp_path):
    # Compare pristine and patched official fourth-output API. The reference is
    # v=i[H,r] with exp(+ikR), not a copy of either implementation's formula.
    raw=(source_root()/'src/classes/@HR/fft.m').read_text()
    raw=re.sub(r'\narguments\b.*?\nend\n','\n',raw,flags=re.S)
    raw=raw.replace('= fft(', '= official_fft(',1)
    (tmp_path/'official_fft.m').write_text(raw)
    install_function(tmp_path,'src/classes/@HR/fft.m','fixed_fft')
    (tmp_path/'tensorprod.m').write_text('function out=tensorprod(A,b,dim,other)\nassert(dim==3 && other==1);out=reshape(reshape(A,[],size(A,3))*b(:),size(A,1),size(A,2));end\n')
    base=models.qi_wu_zhang()
    r=np.c_[base.vectors,np.zeros(len(base.vectors),int)]
    centers=np.array([[0,0,0],[.37,.11,0]])
    model=HR(r,base.hopping,orbitals=centers)
    k=np.array([.31,.27,0])
    savemat(tmp_path/'input.mat',dict(h=model.hopping.transpose(1,2,0),r=r,centers=centers,k=k))
    result=run_octave(tmp_path,"""
load('input.mat');model.HnumL=h;model.vectorL=r;model.Rm=eye(3);model.Dim=3;model.WAN_NUM=2;model.NRPTS=size(r,1);model.orbL_cart=centers;
[w,e,d,v]=official_fft(model,k,[]);raw=zeros(size(v));
for a=1:3;raw(:,:,a)=w*v(:,:,a)*w';end;
[w,e,d,v]=fixed_fft(model,k,[]);fixed=zeros(size(v));
for a=1:3;fixed(:,:,a)=w*v(:,:,a)*w';end;
save('-mat7-binary','result.mat','raw','fixed');
""")
    expected=np.zeros((3,2,2),complex)
    for cell,h in zip(model.vectors,model.hopping):
        for i in range(2):
            for j in range(2):
                displacement=cell+centers[j]-centers[i]
                expected[:,i,j]+=1j*displacement*h[i,j]*np.exp(1j*np.dot(k,cell))
    close(result['fixed'].transpose(2,0,1),expected,atol=1e-13)
    assert np.max(abs(result['raw'].transpose(2,0,1)-expected))>.1
