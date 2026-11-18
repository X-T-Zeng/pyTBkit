"""Reproducible research graphics: bands, signed curvature and normalized DOS."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.integrate import trapezoid

from tbkit import models,kpath,kmesh,berry_curvature
from tbkit.plotting import PALETTE,plot_style,format_axes,bandplot,save_figure


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model=models.qi_wu_zhang(mass=-1)
    path,distance,junctions=kpath([[0,0],[np.pi,0],[np.pi,np.pi],[0,0]],100)
    bands=model.bands(path)
    axis=np.linspace(-np.pi,np.pi,121)
    xx,yy=np.meshgrid(axis,axis,indexing='xy')
    curvature=berry_curvature(model,np.column_stack([xx.ravel(),yy.ravel()]))[:,0].reshape(xx.shape)
    energy=np.linspace(-3.4,3.4,501)
    dos=model.dos(kmesh([91,91]),energy,width=.06)
    assert np.all(dos>=0)
    states=float(trapezoid(dos,energy))
    assert abs(states-2)<1e-5
    with plot_style():
        fig,axes=plt.subplots(1,3,figsize=(10.5,3.5),gridspec_kw={'width_ratios':[1.12,1,1]})
        bandplot(distance,bands,ticks=distance[junctions],labels=[r'$\Gamma$','X','M',r'$\Gamma$'],
                 ax=axes[0],reference=0,ylabel=r'$E/t$')
        axes[0].set_title('(a)  Band structure')
        axes[0].set_ylim(-3.3,3.3)
        axes[0].set_yticks([-3,-1.5,0,1.5,3])
        limit=float(np.max(abs(curvature)))
        field=axes[1].pcolormesh(xx,yy,curvature,cmap='RdBu_r',
                                norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),shading='nearest',rasterized=True)
        axes[1].set(xlabel=r'$k_x a$',ylabel=r'$k_y a$',aspect='equal',
                    xlim=(-np.pi,np.pi),ylim=(-np.pi,np.pi))
        axes[1].set_title('(b)  Occupied-band curvature')
        axes[1].set_xticks([-np.pi,0,np.pi],labels=[r'$-\pi$','0',r'$\pi$'])
        axes[1].set_yticks([-np.pi,0,np.pi],labels=[r'$-\pi$','0',r'$\pi$'])
        # A small colorbar with an explicit unit; both signs retain equal range.
        colorbar=fig.colorbar(field,ax=axes[1],fraction=.046,pad=.035)
        colorbar.set_label(r'$\Omega_{xy}/a^2$',labelpad=4)
        colorbar.set_ticks([-limit,0,limit])
        colorbar.outline.set_linewidth(.6)
        axes[2].plot(energy,dos,color=PALETTE[2])
        axes[2].set(xlabel=r'$E/t$',ylabel=r'DOS $\times\,t$ (states / cell)',xlim=(-3.4,3.4),ylim=(0,None))
        axes[2].set_title('(c)  Density of states')
        axes[2].text(.04,.95,r'$\sigma/t=0.06$',transform=axes[2].transAxes,va='top',fontsize=8.5,color='#647181')
        for ax in axes:format_axes(ax)
        fig.suptitle(r'Qi–Wu–Zhang model   $m/t=-1$  ·  $a=1$',x=.018,ha='left',fontsize=12,fontweight='medium')
        outputs=save_figure(fig,output/'research-gallery')
        plt.close(fig)
    np.savez_compressed(output/'research-gallery-data.npz',path=path,distance=distance,bands=bands,
                        k_axis=axis,curvature=curvature,energy=energy,dos=dos)
    report={'model':'QWZ','mass_over_t':-1,'dos_integrated_states':states,
            'curvature_color_limits':[-limit,limit],'curvature_clipped':False,
            'energy_reference_shift':0,'output_formats':list(outputs),'png_dpi':300}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='work/plot-gallery')
    run(p.parse_args().output)
