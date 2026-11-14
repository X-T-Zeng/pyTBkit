"""D3/S3 characters -> symmetry-adapted three-site tight-binding states."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from tbkit import dihedral_group,MatrixRepresentation,CharacterTable,matrix_group


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    g=dihedral_group(3)
    rotation=sp.Matrix([[-sp.Rational(1,2),-sp.sqrt(3)/2],[sp.sqrt(3)/2,-sp.Rational(1,2)]])
    mirror=sp.diag(1,-1)
    a1=MatrixRepresentation(g,[sp.eye(1) for _ in g],name='A_1')
    a2=MatrixRepresentation(g,[sp.Matrix([[(-1)**b]]) for k,b in g],name='A_2')
    e=MatrixRepresentation(g,[rotation**k*mirror**b for k,b in g],name='E')
    matrices=[]
    for k,b in g:
        m=sp.zeros(3)
        for j in range(3):m[(k+(-1)**b*j)%3,j]=1
        matrices.append(m)
    vertices=MatrixRepresentation(g,matrices,name='sites')
    table=CharacterTable([a1,a2,e]);multiplicities=vertices.decompose([a1,a2,e])
    assert multiplicities==(1,0,1)
    pa,pe=vertices.projector(a1),vertices.projector(e)
    assert pa==sp.ones(3)/3 and pe==sp.eye(3)-pa
    epsilon,t=sp.symbols('epsilon t',real=True)
    h=epsilon*sp.eye(3)+t*(sp.ones(3)-sp.eye(3))
    assert all(h*d==d*h for d in vertices.matrices)
    q=sp.Matrix.hstack(sp.ones(3,1)/sp.sqrt(3),sp.Matrix([2,-1,-1])/sp.sqrt(6),sp.Matrix([0,1,-1])/sp.sqrt(2))
    resolved=(q.adjoint()*h*q).applyfunc(sp.simplify)
    assert resolved==sp.diag(epsilon+2*t,epsilon-t,epsilon-t)
    values=np.array(resolved.subs({epsilon:sp.Rational(1,5),t:-sp.Rational(7,10)}),float).diagonal()
    np.testing.assert_allclose(np.linalg.eigvalsh(np.array(h.subs({epsilon:sp.Rational(1,5),t:-sp.Rational(7,10)}),float)),np.sort(values),atol=1e-14)
    # Independent ordinary quaternionic example: keep the central -I in Q8.
    sx=sp.Matrix([[0,1],[1,0]]);sy=sp.Matrix([[0,-sp.I],[sp.I,0]])
    quaternionic=matrix_group([sp.I*sx,sp.I*sy],name='Q8')
    assert quaternionic.frobenius_schur()==-1
    sections=[('D3 character table',table.latex()),('Exact Cayley table',g.latex()),
              ('Symmetric-sector projector',sp.latex(pa)),('Doublet-sector projector',sp.latex(pe)),
              ('Three-site Hamiltonian',sp.latex(h)),('Symmetry-adapted basis columns',sp.latex(q)),
              ('Resolved Hamiltonian',sp.latex(resolved))]
    document=r'\documentclass{article}'+'\n'+r'\usepackage{amsmath,amssymb,graphicx}'+'\n'+r'\begin{document}'+'\n'
    markdown='# D3 symmetry and a spinless three-site model\n\n'
    for title,formula in sections:
        document+=r'\section*{'+title+'}\n'+r'\['+formula+r'\]'+'\n'
        markdown+='## '+title+'\n\n$$\n'+formula+'\n$$\n\n'
    document+=r'\end{document}'+'\n'
    (output/'group-report.tex').write_text(document)
    (output/'group-report.md').write_text(markdown)
    style={'font.family':'DejaVu Sans','font.size':10,'mathtext.fontset':'stixsans',
           'axes.spines.top':False,'axes.spines.right':False,'axes.titlelocation':'left',
           'axes.titlesize':11,'axes.labelcolor':'#263442','text.color':'#263442',
           'figure.constrained_layout.use':True,'pdf.fonttype':42,'svg.fonttype':'none'}
    with plt.rc_context(style):
        fig,axes=plt.subplots(1,2,figsize=(8.3,3.6),gridspec_kw={'width_ratios':[1.08,1]})
        axes[0].axis('off');axes[0].set_title(r'(a)  $D_3\cong S_3$ character table',pad=16)
        graphic=axes[0].table(cellText=[[f'${sp.latex(x)}$' for x in row] for row in table.values.tolist()],
                              colLabels=[r'$e$ (1)',r'$C_3$ (2)',r'$\sigma_v$ (3)'],
                              rowLabels=[r'$A_1$',r'$A_2$',r'$E$'],loc='center',cellLoc='center',rowLoc='center')
        graphic.auto_set_font_size(False);graphic.set_fontsize(11);graphic.scale(.95,1.95)
        for (row,col),cell in graphic.get_celld().items():
            cell.set_edgecolor('#DCE2E9');cell.set_linewidth(.7)
            if row==0 or col==-1:cell.set_facecolor('#EDF3F8')
        axes[0].text(.5,.1,r'$\Gamma_{\rm sites}=A_1\oplus E$',ha='center',transform=axes[0].transAxes,fontsize=12)
        axes[1].set_title('(b)  Symmetry-resolved trimer',pad=16)
        axes[1].hlines(values[0],-.12,.42,color='#0072B2',lw=2.4)
        axes[1].hlines(np.repeat(values[1],2),[.9,1.42],[1.25,1.77],color='#D55E00',lw=2.4)
        axes[1].text(.15,values[0]+.15,r'$\epsilon+2t$',ha='center',color='#0072B2')
        axes[1].text(1.33,values[1]+.15,r'$\epsilon-t$ (doublet)',ha='center',color='#B04E08')
        axes[1].set(xlim=(-.4,2),ylim=(-1.55,1.5),ylabel='Energy (model units)',xlabel='Irreducible sector')
        axes[1].set_xticks([.15,1.33],labels=[r'$A_1$',r'$E$'])
        axes[1].text(.98,.03,r'$\epsilon=0.2,\ t=-0.7$',ha='right',transform=axes[1].transAxes,color='#647181',fontsize=9)
        for extension in ('png','pdf','svg'):fig.savefig(output/('group-theory.'+extension),dpi=300,bbox_inches='tight')
        plt.close(fig)
    summary={'group_order':g.order,'class_sizes':[len(c) for c in table.classes],
             'irreducible_dimensions':[1,1,2],'site_multiplicities':list(multiplicities),
             'E_tensor_E':list(e.tensor(e).decompose([a1,a2,e])),
             'Q8_order':quaternionic.group.order,'Q8_frobenius_schur':quaternionic.frobenius_schur(),
             'exact_projectors_and_block_diagonalization':True,'spectrum':values.tolist(),
             'matlab_parity':'basic finite-group layer; not all original APIs or corepresentations',
             'latex_exported_not_pdf_compiled':True}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez(output/'group-data.npz',cayley=g.table,character_table=np.array(table.values,int),
             symmetric_projector=np.array(pa,float),doublet_projector=np.array(pe,float),basis=np.array(q,float),spectrum=values)
    print(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='work/group-theory')
    run(p.parse_args().output)
