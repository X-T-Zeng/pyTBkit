"""Energy-unit covariance and explicit device absorption, with analytic checks."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tbkit.transport import PrincipalLayer,chain_transmission,transmission


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    energies=np.linspace(-1.5,1.5,301)
    scales=np.array([1e-20,1.,1e20])
    onsite=.15;gamma_l=.3;gamma_r=.6
    expected=gamma_l*gamma_r/((energies-onsite)**2+(gamma_l+gamma_r)**2/4)
    curves=np.array([[transmission(e*s,[[onsite*s]],[[-.5j*gamma_l*s]],[[-.5j*gamma_r*s]])
                      for e in energies] for s in scales])
    error=float(np.max(abs(curves-expected)))
    assert error<1e-12
    sweep=np.logspace(-20,20,81)
    perfect=np.array([chain_transmission(PrincipalLayer([[0]],[[s]]),0,eta=s*1e-6) for s in sweep])
    assert np.max(abs(perfect-1))<1e-12
    # Reproduce the *effect* of the previous fixed absorber by an independent
    # analytic formula. This is not a rerun of the old decimator at every unit.
    r=(np.sqrt(4+1e-12)-1e-6)/2
    fixed_absorber=(2*r*sweep/(2*r*sweep+1e-12))**2
    style={'font.family':'DejaVu Sans','font.size':9.5,'mathtext.fontset':'stixsans',
           'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,
           'axes.labelcolor':'#263442','text.color':'#263442','xtick.color':'#4B5563','ytick.color':'#4B5563',
           'axes.titlelocation':'left','axes.titlesize':10.5,'axes.titlepad':10,
           'legend.frameon':False,'legend.fontsize':8,'figure.constrained_layout.use':True,
           'pdf.fonttype':42,'svg.fonttype':'none'}
    with plt.rc_context(style):
        fig,axes=plt.subplots(1,2,figsize=(8.4,3.5))
        for curve,label,color,line in zip(curves,[r'$\lambda=10^{-20}$',r'$\lambda=1$',r'$\lambda=10^{20}$'],
                                          ['#0072B2','#009E73','#D55E00'],['-','--',':']):
            axes[0].plot(energies,curve,color=color,linestyle=line,linewidth=1.8,label=label)
        axes[0].plot(energies[::20],expected[::20],'o',color='#263442',ms=3,mfc='white',label='analytic',zorder=3)
        axes[0].set(title='(a)  Single-level transmission',xlabel=r'$E/\lambda$ (reference units)',
                    ylabel='Transmission',xlim=(-1.5,1.5),ylim=(0,1.05))
        axes[0].legend(loc='upper left')
        axes[1].semilogx(sweep,perfect,color='#0072B2',lw=1.8,label='no device absorber')
        axes[1].semilogx(sweep,fixed_absorber,'--',color='#D55E00',lw=1.6,label=r'fixed $\eta_D=10^{-12}$')
        axes[1].set(title='(b)  Perfect-chain unit covariance',xlabel=r'Energy rescaling factor $\lambda$',
                    ylabel='Transmission',xlim=(1e-20,1e20),ylim=(-.03,1.08))
        axes[1].set_xticks([1e-20,1e-10,1,1e10,1e20])
        axes[1].legend(loc='lower right')
        for ax in axes:
            ax.tick_params(direction='out');ax.spines[['bottom','left']].set_color('#4B5563')
        for ext in ('png','pdf','svg'):
            fig.savefig(output/('transport-scaling.'+ext),dpi=300,bbox_inches='tight')
        plt.close(fig)
    report={'max_single_level_analytic_error':error,'max_perfect_chain_unit_error':float(np.max(abs(perfect-1))),
            'energy_rescaling_range':[float(sweep[0]),float(sweep[-1])],
            'extra_device_eta_default':0,'lead_eta_over_hopping':1e-6,
            'fixed_absorber_curve':'independent analytic formula, not original solver output',
            'transmission_units':'dimensionless; no conductance/spin prefactor'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(output/'transport-scaling-data.npz',energies=energies,scales=scales,
                        curves=curves,analytic=expected,unit_sweep=sweep,perfect=perfect,fixed_absorber=fixed_absorber)
    print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='work/transport-scaling')
    run(p.parse_args().output)
