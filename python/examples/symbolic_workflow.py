"""Exact model -> symmetry/formula report -> Numba bands and k.p comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from tbkit import SymbolicHamiltonian,Oper,Qnum,BasisFunc


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    x,y,m,v,b=sp.symbols('k_x k_y m v b',real=True)
    sx=sp.Matrix([[0,1],[1,0]]);sy=sp.Matrix([[0,-sp.I],[sp.I,0]]);sz=sp.diag(1,-1)
    h=SymbolicHamiltonian(v*sp.sin(x)*sx+v*sp.sin(y)*sy+
                          (m+b*(2-sp.cos(x)-sp.cos(y)))*sz,(x,y))
    pauli=h.pauli_decomposition()
    c4=Oper((sp.eye(2)-sp.I*sz)/sp.sqrt(2),sp.Matrix([[0,1],[-1,0]]))
    assert h.is_hermitian() is True
    assert h.symmetry_residual(c4)==sp.zeros(2)
    assert (pauli.reconstruct()-h.matrix).applyfunc(sp.simplify)==sp.zeros(2)
    kp=h.taylor(2)
    expected=v*x*sx+v*y*sy+(m+b*(x*x+y*y)/2)*sz
    assert (kp.matrix-expected).applyfunc(sp.simplify)==sp.zeros(2)
    binding={m:-sp.Rational(1,2),v:sp.S.One,b:sp.S.One}
    numeric=h.compile(binding)
    kx=np.linspace(-np.pi,np.pi,401);k=np.column_stack([kx,np.zeros_like(kx)])
    energies=numeric.bands(k,backend='numba')
    positive=np.sqrt(np.sin(kx)**2+(-.5+1-np.cos(kx))**2)
    error=float(np.max(np.abs(energies-np.column_stack([-positive,positive]))))
    assert error<1e-12
    near=np.linspace(0.01,.3,100);kn=np.column_stack([near,np.zeros_like(near)])
    local_error=np.max(np.abs(numeric.hamiltonian(kn)-kp.to_hk(binding).hamiltonian(kn)),axis=(1,2))
    # sin(k)-k is cubic; test convergence order rather than only matching backends.
    small=np.array([[.1,0],[.05,0]])
    trunc=np.max(np.abs(numeric.hamiltonian(small)-kp.to_hk(binding).hamiltonian(small)),axis=(1,2))
    assert 7.9<trunc[0]/trunc[1]<8.1
    state=BasisFunc({Qnum(1,0,sp.Rational(1,2),sp.Rational(1,2)):1/sp.sqrt(3),
                     Qnum(1,1,sp.Rational(1,2),-sp.Rational(1,2)):sp.sqrt(sp.Rational(2,3))})
    assert state.inner(state)==1
    # A shareable plot, with the same exact Pauli formula shown as in the report.
    fig,axes=plt.subplots(1,2,figsize=(10,4.8))
    axes[0].plot(kx,energies[:,0],color='#214c90',label='lower band')
    axes[0].plot(kx,energies[:,1],color='#b14932',label='upper band')
    axes[0].set(xlabel=r'$k_x$ ($k_y=0$)',ylabel='energy (model units)',title='Exact trigonometric model')
    axes[0].legend(frameon=False)
    axes[1].loglog(near,local_error,color='#226e5c',label='second-order Taylor error')
    axes[1].loglog(near,near**3/6,'--',color='#818181',label=r'$k_x^3/6$')
    axes[1].set(xlabel=r'$k_x$',ylabel=r'$\max_{ij}|H_{ij}-H^{(2)}_{ij}|$',title='Local expansion convergence')
    axes[1].legend(frameon=False)
    fig.suptitle(r'$H=v\sin k_x\,\sigma_x+v\sin k_y\,\sigma_y+[m+b(2-\cos k_x-\cos k_y)]\sigma_z$',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(output/'symbolic-bands.png',dpi=160);plt.close(fig)
    sections=[('Exact Hamiltonian',h.latex()),('Pauli tensor basis',pauli.latex()),
              ('Second-order total-degree expansion',kp.latex()),('Momentum derivative',h.derivative(0).latex()),
              ('C4 input-action convention',c4.latex()),('Normalized coupled-label superposition',state.latex())]
    document=r'\documentclass{article}'+'\n'+r'\usepackage{amsmath,amssymb,graphicx}'+'\n'+r'\begin{document}'+'\n'
    markdown='# Symbolic pyTBkit research example\n\n'
    for title,formula in sections:
        document+=r'\section*{'+title+'}\n'+r'\['+formula+r'\]'+'\n'
        markdown+='## '+title+'\n\n$$\n'+formula+'\n$$\n\n'
    document+=r'\includegraphics[width=\textwidth]{symbolic-bands.png}'+'\n'+r'\end{document}'+'\n'
    markdown+='![Bands and local Taylor convergence](symbolic-bands.png)\n'
    (output/'symbolic-report.tex').write_text(document)
    (output/'symbolic-report.md').write_text(markdown)
    np.savez(output/'symbolic-data.npz',k=k,energy=energies,near_k=near,taylor_error=local_error)
    summary={'hermitian':True,'exact_c4_residual_zero':True,'pauli_reconstruction_exact':True,
             'unbound_parameters_before_compilation':[str(p) for p in h.parameters],
             'max_analytic_energy_error':error,'taylor_error_halving_ratio':float(trunc[0]/trunc[1]),
             'numba_nopython':bool(numeric._h[1].nopython_signatures),
             'tex_exported_not_pdf_compiled':True}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='work/symbolic-example')
    run(parser.parse_args().output)
