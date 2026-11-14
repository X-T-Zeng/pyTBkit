"""Automatically classify a spinless four-site tetrahedral tight-binding model."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from tbkit import permutation_group,MatrixRepresentation,CharacterTable


def cycle_type(permutation):
    remaining=set(range(len(permutation)));lengths=[]
    while remaining:
        node=min(remaining);length=0
        while node in remaining:
            remaining.remove(node);length+=1;node=permutation[node]
        lengths.append(length)
    return tuple(sorted(lengths,reverse=True))


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    group=permutation_group([(1,2,3,0),(1,0,2,3)])
    table=CharacterTable.from_group(group);irreps=table.irreps
    sites=MatrixRepresentation(group,[np.eye(4)[:,g] for g in group],name='sites')
    multiplicities=sites.decompose(irreps)
    assert sorted((r.dimension,m) for r,m in zip(irreps,multiplicities) if m)==[(1,1),(3,1)]
    triplet=next(r for r,m in zip(irreps,multiplicities) if r.dimension==3 and m)
    pa,p3=sites.projector(irreps[0]),sites.projector(triplet)
    errors={'symmetric_projector':float(np.max(np.abs(pa-np.ones((4,4))/4))),
            'triplet_projector':float(np.max(np.abs(p3-(np.eye(4)-np.ones((4,4))/4))))}
    assert max(errors.values())<1e-9
    epsilon,t=sp.symbols('epsilon t',real=True)
    h=epsilon*sp.eye(4)+t*(sp.ones(4)-sp.eye(4))
    q=sp.Matrix.hstack(sp.ones(4,1)/2,sp.Matrix([1,-1,0,0])/sp.sqrt(2),
                      sp.Matrix([1,1,-2,0])/sp.sqrt(6),sp.Matrix([1,1,1,-3])/sp.sqrt(12))
    resolved=(q.adjoint()*h*q).applyfunc(sp.simplify)
    assert resolved==sp.diag(epsilon+3*t,epsilon-t,epsilon-t,epsilon-t)
    values=np.array(resolved.subs({epsilon:sp.Rational(1,5),t:-sp.Rational(7,10)}),float).diagonal()
    numeric=np.array(h.subs({epsilon:sp.Rational(1,5),t:-sp.Rational(7,10)}),float)
    np.testing.assert_allclose(np.linalg.eigvalsh(numeric),np.sort(values),atol=1e-14)
    assert all(np.allclose(d@numeric,numeric@d,atol=1e-14) for d in sites.matrices)
    # Complex rotations within the degenerate triplet cannot change characters.
    rng=np.random.default_rng(53)
    rotation=np.linalg.qr(rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)))[0]
    subspace=np.array(q[:,1:],complex)@rotation
    errors['rotated_triplet_character']=float(np.max(np.abs(sites.on_subspace(subspace).characters()-triplet.characters())))
    assert errors['rotated_triplet_character']<1e-9
    kinds=[(1,1,1,1),(2,1,1),(2,2),(3,1),(4,)]
    representatives=[next(g for g in group if cycle_type(g)==kind) for kind in kinds]
    characters=np.array([[r.characters()[group.index(g)] for g in representatives] for r in irreps])
    # Round only a display copy after checking all displayed values are integers.
    display=np.rint(characters.real).astype(int)
    errors['integer_display']=float(np.max(np.abs(characters-display)))
    assert errors['integer_display']<1e-9
    decomposition=r'\Gamma_{\rm sites}='+r'\oplus'.join(r'\rho_{'+str(i)+'}' for i,m in enumerate(multiplicities) if m)
    style={'font.family':'DejaVu Sans','font.size':10,'mathtext.fontset':'stixsans',
           'axes.spines.top':False,'axes.spines.right':False,'axes.titlelocation':'left',
           'axes.titlesize':11,'axes.labelcolor':'#263442','text.color':'#263442',
           'figure.constrained_layout.use':True,'pdf.fonttype':42,'svg.fonttype':'none'}
    with plt.rc_context(style):
        fig,axes=plt.subplots(1,2,figsize=(9.6,3.9),gridspec_kw={'width_ratios':[1.35,1]})
        axes[0].axis('off');axes[0].set_title(r'(a)  Automatically generated $S_4$ characters',pad=18)
        graphic=axes[0].table(cellText=display.tolist(),
                             colLabels=[r'$e$ (1)',r'$(12)$ (6)',r'$(12)(34)$ (3)',r'$(123)$ (8)',r'$(1234)$ (6)'],
                             rowLabels=[rf'$\rho_{{{i}}}$' for i in range(len(irreps))],
                             loc='center',cellLoc='center',rowLoc='center')
        graphic.auto_set_font_size(False);graphic.set_fontsize(10);graphic.scale(.95,1.95)
        for (row,col),cell in graphic.get_celld().items():
            cell.set_edgecolor('#DCE2E9');cell.set_linewidth(.7)
            if row==0 or col==-1:cell.set_facecolor('#EDF3F8')
        axes[0].text(.5,.09,'$'+decomposition+'$',ha='center',transform=axes[0].transAxes,fontsize=12)
        axes[0].text(.5,.005,'Numerical irreps; displayed characters rounded. Parentheses: class size.',
                     ha='center',transform=axes[0].transAxes,fontsize=7.5,color='#647181')
        axes[1].set_title('(b)  Tetrahedral four-site cluster',pad=18)
        axes[1].hlines(values[0],-.1,.45,color='#0072B2',lw=2.4)
        axes[1].hlines(np.repeat(values[1],3),[.9,1.35,1.8],[1.2,1.65,2.1],color='#D55E00',lw=2.4)
        axes[1].text(.17,values[0]+.22,r'$\epsilon+3t$',ha='center',color='#0072B2')
        axes[1].text(1.5,values[1]+.22,r'$\epsilon-t$ (triplet)',ha='center',color='#B04E08')
        axes[1].set(xlim=(-.4,2.4),ylim=(-2.45,1.8),ylabel='Energy (model units)',xlabel='Irreducible sector')
        axes[1].set_xticks([.17,1.5],labels=['symmetric','three dimensional'])
        axes[1].text(.98,.03,r'$\epsilon=0.2,\ t=-0.7$',ha='right',transform=axes[1].transAxes,color='#647181',fontsize=9)
        for extension in ('png','pdf','svg'):fig.savefig(output/('automatic-irreps.'+extension),dpi=300,bbox_inches='tight')
        plt.close(fig)
    sections=[('Analytic spinless tetrahedron Hamiltonian',sp.latex(h)),('Analytic orthonormal basis',sp.latex(q)),
              ('Exact analytic block energies',sp.latex(resolved)),
              ('Numerical characters: rounded display only',sp.latex(sp.ImmutableMatrix(display))),
              ('Numerical sector identification',decomposition)]
    document=r'\documentclass{article}'+'\n'+r'\usepackage{amsmath,amssymb}'+'\n'+r'\begin{document}'+'\n'
    markdown='# Automatic ordinary irreps and a spinless tetrahedron\n\n'
    for title,formula in sections:
        document+=r'\section*{'+title+'}\n'+r'\['+formula+r'\]'+'\n'
        markdown+='## '+title+'\n\n$$\n'+formula+'\n$$\n\n'
    document+=r'\end{document}'+'\n'
    (output/'automatic-irrep-report.tex').write_text(document)
    (output/'automatic-irrep-report.md').write_text(markdown)
    summary={'group_order':group.order,'irrep_dimensions':[r.dimension for r in irreps],
             'site_multiplicities':list(multiplicities),'triplet_irrep':triplet.name,
             'spectrum':values.tolist(),'max_errors':errors,
             'fs_indicators':[r.frobenius_schur() for r in irreps],
             'numeric_irreps_not_exact':True,'analytic_hamiltonian_exact':True,
             'assumptions':'spinless scalar sites at regular tetrahedron vertices, equal real hopping between every pair; full Td permutation action is S4',
             'latex_exported_not_document_compiled':True}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez(output/'automatic-irrep-data.npz',cayley=group.table,characters=characters,
             symmetric_projector=pa,triplet_projector=p3,analytic_basis=np.array(q,complex),hamiltonian=numeric,
             spectrum=values,**{r.name:r.as_arrays() for r in irreps})
    print(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='work/automatic-irreps')
    run(parser.parse_args().output)
