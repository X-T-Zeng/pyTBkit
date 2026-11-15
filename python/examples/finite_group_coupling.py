"""Exact D3 tensor coefficients and mixing between two equivalent doublets."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from tbkit import dihedral_group,MatrixRepresentation,clebsch_gordan,symmetry_adapted_basis


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    group=dihedral_group(3)
    r=sp.Matrix([[-sp.Rational(1,2),-sp.sqrt(3)/2],[sp.sqrt(3)/2,-sp.Rational(1,2)]])
    mirror=sp.diag(1,-1)
    a1=MatrixRepresentation(group,[sp.eye(1) for _ in group],name='A_1')
    a2=MatrixRepresentation(group,[sp.Matrix([[(-1)**b]]) for k,b in group],name='A_2')
    e=MatrixRepresentation(group,[r**k*mirror**b for k,b in group],name='E')
    cg=clebsch_gordan(e,e,[a1,a2,e]);assert cg.multiplicities==(1,1,1)
    exchange=sp.Matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
    assert cg.transform(exchange).applyfunc(sp.simplify)==sp.diag(1,-1,1,1)
    # Constant complex unitary hides the two equivalent copies in site labels.
    mix=sp.eye(4);mix[0,0]=mix[3,3]=sp.Rational(3,5);mix[0,3]=mix[3,0]=4*sp.I/5
    hadamard=sp.Matrix([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])/2
    change=hadamard*mix
    target=MatrixRepresentation(group,[change*m*change.adjoint() for m in e.direct_sum(e).matrices],name='two_doublets')
    decomposition=symmetry_adapted_basis(target,[a1,a2,e]);assert decomposition.multiplicities==(0,0,2)
    delta,tr,ti=sp.symbols('delta t_r t_i',real=True)
    multiplicity=sp.Matrix([[delta,tr+sp.I*ti],[tr-sp.I*ti,-delta]])
    canonical=sp.kronecker_product(multiplicity,sp.eye(2))
    h=(change*canonical*change.adjoint()).applyfunc(sp.simplify)
    assert (h-h.adjoint()).applyfunc(sp.simplify)==sp.zeros(4)
    assert all((h*d-d*h).applyfunc(sp.simplify)==sp.zeros(4) for d in target.matrices)
    resolved=decomposition.transform(h)
    reduced=sp.Matrix([[resolved[2*i,2*j] for j in range(2)] for i in range(2)])
    assert (resolved-sp.kronecker_product(reduced,sp.eye(2))).applyfunc(sp.simplify)==sp.zeros(4)
    assert sp.simplify(reduced.trace())==0
    assert sp.simplify(reduced.det()+delta**2+tr**2+ti**2)==0
    # An arbitrary multiplicity gauge can change reduced's entries, not its roots.
    detunings=np.linspace(-1.1,1.1,151);hopping=.3+.2j;numeric_change=np.array(change,complex)
    spectra=np.array([np.linalg.eigvalsh(numeric_change@np.kron([[x,hopping],[hopping.conjugate(),-x]],np.eye(2))@numeric_change.conj().T) for x in detunings])
    roots=np.sqrt(detunings**2+abs(hopping)**2)
    analytic=np.column_stack([-roots,-roots,roots,roots])
    error=float(np.max(np.abs(spectra-analytic)))
    assert error<1e-13
    splits=float(max(np.max(np.abs(spectra[:,0]-spectra[:,1])),np.max(np.abs(spectra[:,2]-spectra[:,3]))))
    style={'font.family':'DejaVu Sans','font.size':10,'mathtext.fontset':'stixsans',
           'axes.spines.top':False,'axes.spines.right':False,'axes.titlelocation':'left',
           'axes.titlesize':11,'axes.labelcolor':'#263442','text.color':'#263442',
           'figure.constrained_layout.use':True,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False}
    with plt.rc_context(style):
        fig,axes=plt.subplots(1,2,figsize=(9.3,3.9),gridspec_kw={'width_ratios':[1.1,1]})
        axes[0].axis('off');axes[0].set_title(r'(a)  Exact $D_3$: $E\otimes E=A_1\oplus A_2\oplus E$',pad=18)
        graphic=axes[0].table(cellText=[['$'+sp.latex(value)+'$' for value in row] for row in cg.basis.tolist()],
                             rowLabels=[r'$xx$',r'$xy$',r'$yx$',r'$yy$'],
                             colLabels=[r'$A_1$',r'$A_2$',r'$E_x$',r'$E_y$'],loc='center',cellLoc='center',rowLoc='center')
        graphic.auto_set_font_size(False);graphic.set_fontsize(11);graphic.scale(.94,2.3)
        for (row,col),cell in graphic.get_celld().items():
            cell.set_edgecolor('#DCE2E9');cell.set_linewidth(.7)
            if row==0 or col==-1:cell.set_facecolor('#EDF3F8')
        axes[0].text(.5,.07,'Columns are coupled kets; overall irrep phases are conventional.',ha='center',transform=axes[0].transAxes,fontsize=8,color='#647181')
        axes[1].set_title(r'(b)  Two equivalent $E$ doublets',pad=18)
        axes[1].plot(detunings,spectra[:,2],color='#0072B2',lw=2,label=r'$E_+$ (doublet)')
        axes[1].plot(detunings,spectra[:,0],color='#D55E00',lw=2,label=r'$E_-$ (doublet)')
        axes[1].axhline(0,color='#CAD2DC',lw=.6)
        axes[1].annotate('',xy=(0,abs(hopping)),xytext=(0,-abs(hopping)),arrowprops=dict(arrowstyle='<->',color='#647181',lw=.8))
        axes[1].text(.07,0,r'$2|t|$',color='#647181',va='center',fontsize=9,
                     bbox=dict(facecolor='white',edgecolor='none',pad=2))
        axes[1].set(xlim=(-1.1,1.1),ylim=(-1.4,1.4),xlabel=r'Detuning $\delta$ (model units)',ylabel='Energy (model units)')
        axes[1].set_xticks([-1,-.5,0,.5,1])
        axes[1].legend(loc='upper center',fontsize=8,ncol=2,handlelength=1.6,columnspacing=1)
        axes[1].text(.98,.03,r'$t=0.3+0.2i$',ha='right',transform=axes[1].transAxes,color='#647181',fontsize=9)
        for extension in ('png','pdf','svg'):fig.savefig(output/('finite-group-coupling.'+extension),dpi=300,bbox_inches='tight')
        plt.close(fig)
    sections=[('Exact finite-group ket coefficients',sp.latex(cg.basis)),('Exchange parity',sp.latex(cg.transform(exchange))),
              ('Canonical multiplicity Hamiltonian',sp.latex(canonical)),('Constant complex basis change',sp.latex(change)),
              ('Recovered aligned basis',sp.latex(decomposition.basis)),('Reduced multiplicity Hamiltonian',sp.latex(reduced)),
              ('Protected-doublet energies',r'E_{\pm}=\pm\sqrt{\delta^2+t_r^2+t_i^2}')]
    document=r'\documentclass{article}'+'\n'+r'\usepackage{amsmath,amssymb}'+'\n'+r'\begin{document}'+'\n'
    markdown='# Finite-group coupling and equivalent doublets\n\n'
    for title,formula in sections:
        document+=r'\section*{'+title+'}\n'+r'\['+formula+r'\]'+'\n'
        markdown+='## '+title+'\n\n$$\n'+formula+'\n$$\n\n'
    (output/'coupling-report.tex').write_text(document+r'\end{document}'+'\n')
    (output/'coupling-report.md').write_text(markdown)
    np.savez(output/'coupling-data.npz',cg_basis=np.array(cg.basis,complex),aligned_basis=np.array(decomposition.basis,complex),
             basis_change=numeric_change,detunings=detunings,spectra=spectra,analytic_spectra=analytic,hopping=hopping)
    summary={'tensor_multiplicities':list(cg.multiplicities),'repeated_doublet_multiplicities':list(decomposition.multiplicities),
             'exact_symbolic_cg_and_schur_blocks':True,'exchange_parities':[1,-1,1,1],
             'max_energy_error':error,'max_doublet_splitting':splits,'hopping':[hopping.real,hopping.imag],
             'convention':'columns: irrep, copy, internal index; uncoupled tensor row a*dim(right)+b',
             'scope':'ordinary D3 group, no antiunitary or projective sewing; multiplicity gauge is not canonical',
             'latex_exported_not_document_compiled':True}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='work/finite-group-coupling')
    run(parser.parse_args().output)
