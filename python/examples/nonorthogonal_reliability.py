"""Unit-covariant derivatives and loss of information in dependent orbital bases."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from tbkit import HK, NonOrthogonal


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    sx=np.array([[0.,1.],[1.,0.]]);sz=np.diag([1.,-1.])
    k=np.linspace(-1.5,1.5,201)[:,None];expected=k[:,0]/np.sqrt(k[:,0]**2+.16)
    scales=[1e-100,1.,1e100];slopes=[]
    for scale in scales:
        model=NonOrthogonal(HK({(0,):scale*.4*sz,(1,):scale*sx}),HK({(0,):np.eye(2)}))
        slope=model.band_gradient(k)[:,0,1]/scale
        np.testing.assert_allclose(slope,expected,rtol=2e-14,atol=3e-16)
        slopes.append(slope)
    q,_=np.linalg.qr(np.random.default_rng(13).normal(size=(3,3)))
    deltas=np.logspace(0,-16,65);errors=[];accepted=[];conditions=[];residuals=[];metric_errors=[]
    for delta in deltas:
        b=np.diag([1.,1.,np.sqrt(delta)])@q
        h=b.T@np.diag([1.,2.,3.])@b;s=b.T@b
        # Unrestricted solve is an observed failure/sensitivity curve, NOT the
        # reference truth for the rounded pencil. Exact congruence gives 1,2,3.
        try:errors.append(float(np.max(abs(eigh(h,s,eigvals_only=True)-[1,2,3]))/3))
        except np.linalg.LinAlgError:errors.append(np.nan)
        model=NonOrthogonal(HK({(0,):h}),HK({(0,):s}))
        try:
            result=model.spectrum([0.]);accepted.append(True)
            conditions.append(result.overlap_condition[0]);residuals.append(result.residual.max());metric_errors.append(result.metric_error.max())
        except ValueError:
            accepted.append(False);conditions.append(np.nan);residuals.append(np.nan);metric_errors.append(np.nan)
    errors=np.array(errors);accepted=np.array(accepted);x=1/deltas
    if accepted[-1]:raise AssertionError('nearly dependent basis was accepted')
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180}):
        fig,axes=plt.subplots(1,2,figsize=(10.8,4.1),constrained_layout=True)
        for slope,scale,color,style in zip(slopes,scales,['#245b8e','#d77a32','#3b8a78'],['-','--',':']):
            axes[0].plot(k[:,0],slope,style,lw=2,color=color,label=f'$E_0=10^{{{int(np.log10(scale))}}}$')
        axes[0].set(xlabel='$k$ (dimensionless)',ylabel='$(\\partial E_+/\\partial k)/E_0$',title='Unit-covariant band derivative')
        axes[0].legend(frameon=False,loc='upper left');axes[0].grid(alpha=.16)
        # Floor is used ONLY for plotting exact floating zeros on a log axis.
        # Unmodified errors, rejection mask and diagnostic data go into NPZ.
        y=np.maximum(errors,np.finfo(float).eps/10)
        axes[1].axvspan(1e8,1e16,color='#f5ece6',zorder=0)
        axes[1].loglog(x,y,color='#939ca6',lw=1,label='Unrestricted rounded-input solve')
        axes[1].scatter(x[accepted],y[accepted],s=16,color='#3b8a78',label='Accepted after checks',zorder=3)
        rejected=~accepted & np.isfinite(y)
        axes[1].scatter(x[rejected],y[rejected],s=25,color='#b64d40',marker='x',label='Rejected by reliability gate',zorder=3)
        axes[1].axvline(1e8,color='#b64d40',ls='--',lw=1)
        axes[1].set(xlabel='Intended $\\kappa(S)=1/\\delta$',ylabel='$\\max_n|E_n-n|/3$',
                    title='Rounding amplified by a dependent basis',xlim=(1,1e16),ylim=(1e-17,3))
        axes[1].legend(frameon=False,loc='upper left',fontsize=8)
        axes[1].grid(alpha=.16,which='major')
        fig.savefig(output/'nonorthogonal-reliability.png');fig.savefig(output/'nonorthogonal-reliability.svg');plt.close(fig)
    np.savez_compressed(output/'nonorthogonal-reliability.npz',k=k,scales=scales,slopes=slopes,delta=deltas,
                        raw_energy_error=errors,accepted=accepted,estimated_condition=conditions,
                        residual=residuals,metric_error=metric_errors)
    report={'seed':13,'condition_limit':1e8,'scales':scales,
            'max_absolute_scaled_gradient_error':float(np.max(abs(np.array(slopes)-expected))),
            'basis_scan_points':len(deltas),'accepted':int(accepted.sum()),'rejected':int((~accepted).sum()),
            'max_accepted_residual':float(np.nanmax(residuals)),'max_accepted_metric_error':float(np.nanmax(metric_errors)),
            'mathematical_reference':'exact congruence B†diag(1,2,3)B, B†B gives energies 1,2,3',
            'scope':'small residual alone does not bound forward error of rounded ill-conditioned input; no automatic basis truncation'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='work/nonorthogonal-reliability')
    print(json.dumps(run(parser.parse_args().output),indent=2))
