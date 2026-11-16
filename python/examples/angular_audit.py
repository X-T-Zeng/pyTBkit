"""Infinitesimal response and high-j reference failures, with pristine snapshots."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from tbkit import Qnum,BasisFunc,angular_cg
from tbkit.orbitals import spin_rotation


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    audit=Path(__file__).resolve().parents[1]/'audit'
    pristine=json.loads((audit/'angular-python-pristine.json').read_text())
    matlab=json.loads((audit/'angular-matlab-pristine.json').read_text())
    angles=np.array(pristine['small_angle']['angles']);old=np.array(pristine['small_angle']['sy_expectation'])
    up=Qnum(0,0,.5,.5);down=Qnum(0,0,.5,-.5);sy=np.array([[0,-1j],[1j,0]])/2
    fixed=[]
    for theta in angles:
        state=BasisFunc({up:1}).rotate([1,0,0],theta)
        vector=np.array([complex(state.terms.get(up,0)),complex(state.terms.get(down,0))])
        fixed.append(np.vdot(vector,sy@vector).real)
    fixed=np.array(fixed);analytic=-np.sin(angles)/2
    relative=float(np.max(abs((fixed-analytic)/analytic)));assert relative<2e-14
    base=np.array([1.,2.,-3.]);reference=spin_rotation(base,.7)
    axis_error=float(max(np.max(abs(spin_rotation(base*scale,.7)-reference)) for scale in (1e-300,1e300)))
    assert axis_error<3e-15
    large=spin_rotation([1,0,0],1e300)
    assert np.isfinite(large).all()
    large_error=float(np.max(abs(large.conj().T@large-np.eye(2))))
    js=np.array([row['j'] for row in matlab['cases']]);raw=np.array([row['wigner_original'] for row in matlab['cases']])
    wigner=[];singlets=[];wigner_errors=[]
    for j in js:
        j=int(j);g=3*j//2
        closed=(-1)**g*sp.factorial(g)/sp.factorial(j//2)**3*sp.sqrt(sp.factorial(j)**3/sp.factorial(3*j+1))
        value=angular_cg(j,j,j,0,0,0)/sp.sqrt(2*j+1)
        assert sp.simplify(value-closed)==0
        numeric=float(value.evalf(30));wigner.append(numeric)
        wigner_errors.append(abs(numeric-float(closed.evalf(30))))
        singlet=angular_cg(j,j,0,j,-j,0,exact=False);singlets.append(singlet)
        np.testing.assert_allclose(singlet,1/np.sqrt(2*j+1),atol=0,rtol=2e-15)
    wigner=np.array(wigner);singlets=np.array(singlets)
    style={'font.family':'DejaVu Sans','font.size':10,'mathtext.fontset':'stixsans',
           'axes.spines.top':False,'axes.spines.right':False,'axes.titlelocation':'left',
           'axes.titlesize':11,'axes.labelcolor':'#263442','text.color':'#263442',
           'figure.constrained_layout.use':True,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False}
    with plt.rc_context(style):
        fig,axes=plt.subplots(1,2,figsize=(9.5,3.9),gridspec_kw={'width_ratios':[1,1.08]})
        axes[0].plot(angles,-2*fixed/angles,color='#0072B2',lw=2,label='Fixed Python')
        axes[0].plot(angles,-2*old/angles,color='#D55E00',lw=1.3,marker='o',ms=2.2,label='Pristine Python')
        axes[0].axvline(2e-15,color='#9AA7B4',lw=.8,ls=':')
        axes[0].set(xscale='log',ylim=(-.08,1.18),xlabel=r'Rotation angle $\theta$ (rad)',ylabel=r'$-2\langle S_y\rangle/\theta$',title='(a)  Infinitesimal spin response')
        axes[0].set_yticks([0,.5,1]);axes[0].legend(loc='lower right',fontsize=8)
        axes[0].text(.03,.96,r'$d\langle S_y\rangle/d\theta=-1/2$',transform=axes[0].transAxes,va='top',fontsize=9)
        normalized=raw*np.sqrt(2*js+1);safe=wigner*np.sqrt(2*js+1)
        axes[1].axhspan(-1,1,color='#E3F2ED',label='Normalization bound')
        axes[1].plot(js,normalized,'o-',color='#D55E00',lw=1.4,ms=4,label='Official raw (Octave 10.3)')
        axes[1].plot(js,safe,'s-',color='#0072B2',lw=1.5,ms=3.5,label='Exact preparation → float')
        axes[1].set_yscale('symlog',linthresh=1,linscale=1.2)
        axes[1].set(xlabel=r'Angular momentum $j$ (even integers)',ylabel=r'$\sqrt{2j+1}\,W_j(0,0,0)$',title='(b)  High-j Wigner 3j audit')
        axes[1].set_xticks([10,50,80,100,120]);axes[1].legend(loc='lower left',fontsize=7.5)
        for extension in ('png','pdf','svg'):fig.savefig(output/('angular-audit.'+extension),dpi=300,bbox_inches='tight')
        plt.close(fig)
    sections=[('Active spin rotation',r'U_x(\theta)=\exp(-i\theta S_x),\quad\langle S_y\rangle=-\tfrac12\sin\theta'),
              ('Factorized ladder coefficient',r'J_+|j,m\rangle=\sqrt{(j-m)(j+m+1)}|j,m+1\rangle'),
              ('Exact singlet CG',r'\langle j,m;j,-m|0,0\rangle=\frac{(-1)^{j-m}}{\sqrt{2j+1}}'),
              ('All-zero-m Wigner symbol for even j',r'W_j=\frac{(-1)^{3j/2}(3j/2)!}{[(j/2)!]^3}\sqrt{\frac{(j!)^3}{(3j+1)!}}')]
    tex=r'\documentclass{article}'+'\n'+r'\usepackage{amsmath,amssymb}'+'\n'+r'\begin{document}'+'\n'
    markdown='# Angular invariants and high-j audit\n\n'
    for title,formula in sections:
        tex+=r'\section*{'+title+'}\n'+r'\['+formula+r'\]'+'\n';markdown+='## '+title+'\n\n$$\n'+formula+'\n$$\n\n'
    (output/'angular-report.tex').write_text(tex+r'\end{document}'+'\n');(output/'angular-report.md').write_text(markdown)
    np.savez(output/'angular-data.npz',angles=angles,pristine_sy=old,fixed_sy=fixed,analytic_sy=analytic,
             j=js,official_wigner=raw,exact_wigner=wigner,stable_singlet_cg=singlets)
    summary={'max_small_angle_relative_error':relative,'max_axis_rescaling_error':axis_error,
             'large_angle_unitarity_error':large_error,'max_wigner_closed_form_error':float(max(wigner_errors)),
             'pristine_source_python':pristine['python_commit'],'official_source':matlab['official_commit'],
             'high_j_matlab_fix_claimed':False,'exact_cg_preparation_not_nopython':True,
             'assumptions':'hbar=1, active rotations, spin-up initial ket, SU(2) 4pi period; bound follows CG normalization',
             'latex_exported_not_document_compiled':True}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='work/angular-audit')
    run(parser.parse_args().output)
