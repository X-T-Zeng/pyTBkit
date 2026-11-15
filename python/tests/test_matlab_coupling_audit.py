"""Execute pristine angular-CG defects separately from minimal candidates."""
import shutil
import subprocess
import numpy as np
import pytest
import sympy as sp
from scipy.io import loadmat,savemat
from sympy.physics.wigner import clebsch_gordan
from reference_source import source_root


def wrapper(folder,patched):
    octave=shutil.which('octave')
    if octave is None:pytest.skip('Octave unavailable')
    source=source_root(patched=patched)/'src/classes/@Y_l__m'
    # Static wrapper only: verbatim function bodies, no formula/sign adapter.
    methods='\n'.join((source/(name+'.m')).read_text() for name in ('ClebschGordan','Wigner3j','CG'))
    (folder/'Y_l__m.m').write_text('classdef Y_l__m\nmethods (Static)\n'+methods+'\nend\nend\n')
    return octave


def run(folder,octave,script):
    (folder/'run_reference.m').write_text(script)
    subprocess.run([octave,'--no-gui','--quiet','run_reference.m'],cwd=folder,capture_output=True,check=True,timeout=60)
    return loadmat(folder/'result.mat',squeeze_me=True)


@pytest.mark.parametrize('patched',[False,True],ids=['pristine','candidate'])
@pytest.mark.parametrize('entry',['ClebschGordan','CG'])
def test_original_call_signature_and_zero_output_defects(patched,entry,tmp_path):
    octave=wrapper(tmp_path,patched)
    expression=('Y_l__m.ClebschGordan(.5,.5,0,.5,-.5,0)' if entry=='ClebschGordan'
                else 'Y_l__m.CG(1,1,1,1,1,1)')
    result=run(tmp_path,octave,"""
failed=false;message='';value=NaN;
try
    value=EXPRESSION;
catch err
    failed=true;message=err.message;
end
save('-mat7-binary','result.mat','failed','message','value');
""".replace('EXPRESSION',expression))
    assert bool(result['failed']) is (not patched)
    if patched:
        expected=1/np.sqrt(2) if entry=='ClebschGordan' else 0
        np.testing.assert_allclose(result['value'],expected,atol=1e-14)
    elif entry=='ClebschGordan':assert 'too many inputs' in str(result['message']).lower()
    else:
        # Octave 8 names the unassigned output; Octave 10 names the assignment.
        message=str(result['message']).lower()
        assert "'cg' undefined" in message or 'assignment is undefined' in message


def test_candidate_angular_cg_against_exact_condon_shortley_coefficients(tmp_path):
    octave=wrapper(tmp_path,True);cases=[];expected=[]
    for j1,j2 in ((sp.Rational(1,2),sp.Rational(1,2)),(1,sp.Rational(1,2)),(1,1),(2,1)):
        for step in range(int(2*min(j1,j2))+1):
            j=abs(j1-j2)+step
            for a in range(int(2*j1)+1):
                m1=j1-a
                for b in range(int(2*j2)+1):
                    m2=j2-b
                    for c in range(int(2*j)+1):
                        m=j-c;cases.append([float(v) for v in (j1,j2,j,m1,m2,m)])
                        expected.append(float(clebsch_gordan(j1,j2,j,m1,m2,m)))
    savemat(tmp_path/'cases.mat',{'cases':np.array(cases)})
    result=run(tmp_path,octave,"""
warning('off','all');load('cases.mat');values=zeros(rows(cases),2);
for i=1:rows(cases)
    p=cases(i,:);
    values(i,1)=Y_l__m.ClebschGordan(p(1),p(2),p(3),p(4),p(5),p(6));
    values(i,2)=Y_l__m.CG(p(1),p(4),p(2),p(5),p(3),p(6));
end
save('-mat7-binary','result.mat','values');
""")
    np.testing.assert_allclose(result['values'],np.array(expected)[:,None]+np.zeros((1,2)),atol=2e-13,rtol=0)
