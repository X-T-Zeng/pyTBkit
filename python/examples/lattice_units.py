"""Length-unit covariance and a missing-bond counterexample, with raw data."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tbkit import Structure
from tbkit.slater_koster import from_structure
from tbkit.plotting import plot_style,save_figure,PALETTE


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    q=np.linspace(-np.pi,np.pi,301);lengths=np.array([1e-150,1e-9,1.,1e150])
    energy=[];derivative=[];counts=[]
    for a in lengths:
        structure=Structure(a*np.diag([1.,4.,4.]),[[0,0,0]],['X'])
        model=from_structure(structure,[['s']],1.1*a,{'ss_sigma':1.},periodic=(True,False,False))
        k=np.column_stack((q/a,np.zeros((len(q),2))))
        e=model.bands(k)[:,0];d=model.gradient(k)[:,0,0,0].real/a
        np.testing.assert_allclose(e,2*np.cos(q),rtol=2e-14,atol=2e-15)
        np.testing.assert_allclose(d,-2*np.sin(q),rtol=2e-14,atol=2e-15)
        energy.append(e);derivative.append(d);counts.append(len(structure.neighbors(1.1*a,periodic=(True,False,False)).i))
    # Historical values are an executed main snapshot with recorded source hash.
    pristine=json.loads((Path(__file__).resolve().parents[1]/'audit/lattice-python-pristine.json').read_text())
    old=np.array(pristine['close_pair']['observed_energies'])
    pair=Structure(np.eye(3),[[0,0,0],[5e-11,0,0]],['A','B'])
    fixed=from_structure(pair,[['s'],['s']],8e-11,{'ss_sigma':1.},periodic=(False,False,False)).bands([0,0,0])
    np.testing.assert_allclose(fixed,[-1,1],rtol=2e-14,atol=0)
    with plot_style('paper',**{'font.size':9.5,'axes.titlesize':11,'axes.labelsize':10.5}):
        fig,axes=plt.subplots(1,3,figsize=(12.5,3.55),constrained_layout=True)
        for a,e,d,color,style in zip(lengths,energy,derivative,PALETTE,['-','--',':','-.']):
            label=f'$a=10^{{{int(np.log10(a))}}}$'
            axes[0].plot(q,e,color=color,ls=style,lw=1.7,label=label)
            axes[1].plot(q,d,color=color,ls=style,lw=1.7)
        axes[0].set(xlabel='$k_xa$',ylabel='$E/t$',title='(a) One chain, four length units')
        axes[1].set(xlabel='$k_xa$',ylabel='$(ta)^{-1}\\,\\partial E/\\partial k_x$',title='(b) Derivative covariance')
        axes[0].legend(frameon=False,loc='lower center',fontsize=8)
        for ax in axes[:2]:
            ax.set_xticks([-np.pi,0,np.pi],[r'$-\pi$','0',r'$\pi$']);ax.grid(alpha=.12)
        for center,levels,color in ((0,old,PALETTE[3]),(1,fixed,PALETTE[0])):
            for e in levels:axes[2].plot([center-.21,center+.21],[e,e],color=color,lw=2.5)
        axes[2].text(0,.12,'2 states',ha='center',fontsize=8)
        axes[2].text(.5,1.28,'$d=5\\times10^{-11}$, $r_c=8\\times10^{-11}$',ha='center',fontsize=8)
        axes[2].set(xticks=[0,1],xticklabels=['main snapshot','relative tolerance'],ylabel='$E/t$',
                    title='(c) Lost-bond counterexample',xlim=(-.5,1.5),ylim=(-1.4,1.55))
        axes[2].axhline(0,color='.75',lw=.6,ls='--',zorder=0)
        save_figure(fig,output/'lattice-units');plt.close(fig)
    np.savez_compressed(output/'lattice-units.npz',q=q,lengths=lengths,energy=energy,normalized_derivative=derivative,
                        pair_old=old,pair_fixed=fixed,bond_counts=counts)
    report={'length_scales':lengths.tolist(),'directed_bonds_per_chain_cell':counts,
            'max_absolute_energy_error':float(np.max(abs(np.array(energy)-2*np.cos(q)))),
            'max_absolute_normalized_derivative_error':float(np.max(abs(np.array(derivative)+2*np.sin(q)))),
            'pristine_python_commit':pristine['base_commit'],'pair_old':old.tolist(),'pair_fixed':fixed.tolist(),
            'units':'common arbitrary length units; t=1, no implicit Angstrom/SI conversion or hbar velocity factor',
            'assumption':'fixed s-s integral t; distance-dependent parameter callbacks must transform their length units consistently'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='work/lattice-units')
    print(json.dumps(run(parser.parse_args().output),indent=2))
