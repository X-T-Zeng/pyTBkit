"""Point-orbital geometry survives algebra: analytic Zak and QWZ references."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tbkit import HR,models,embed_orbitals,wilson_loop,prepare_geometry
from tbkit.plotting import plot_style,save_figure


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    old=json.loads((Path(__file__).resolve().parents[1]/'audit/gauge-python-pristine.json').read_text())
    base=models.ssh();centers=np.linspace(0,.8,17);k=(2*np.pi*np.arange(151)/151)[:,None]
    phases=[]
    for a in centers:
        model=2*embed_orbitals(HR(base.vectors,base.hopping,orbitals=[[0],[a]]))
        phases.append(wilson_loop(model,k,boundary_sewing=model.boundary_sewing([2*np.pi]))[0])
    phases=np.array(phases);reference=np.pi*(1+centers)
    phase_error=float(np.max(abs(np.exp(1j*phases)-np.exp(1j*reference))))
    assert phase_error<2e-13
    q=models.qi_wu_zhang();q=HR(q.vectors,q.hopping,orbitals=[[0,0],[.37,0]])
    ky=np.linspace(-np.pi,np.pi,181);probe=np.column_stack((np.full_like(ky,.31),ky))
    coefficient=prepare_geometry(q,probe,backend='numpy').berry[:,0,1,0]
    embedded=embed_orbitals(q)
    point=prepare_geometry(embedded,probe,backend='numpy').berry[:,0,1,0]
    scaled=prepare_geometry(2*embedded,probe,backend='numba',frame='point-orbital').berry[:,0,1,0]
    x,y=probe.T;z=-1+np.cos(x)+np.cos(y);r=np.sqrt(np.sin(x)**2+np.sin(y)**2+z*z)
    dz=-np.sin(y);dr=(np.sin(y)*np.cos(y)+z*dz)/r
    correction=-.37*.5*(dz/r-z*dr/r**2)
    np.testing.assert_allclose(point-coefficient,correction,atol=3e-14)
    np.testing.assert_allclose(scaled,point,atol=3e-14)
    # A flat intracell dimer: Im(dH_12/dk)/(t*a)=cos(k*a), a=.4.
    dimer=embed_orbitals(HR([[0]],[[[0,1.3],[1.3,0]]],orbitals=[[.2],[.6]]))
    path=np.linspace(-2*np.pi,2*np.pi,121)[:,None]
    derivative=(2*dimer).gradient(path,backend='numba')[:,0,0,1].imag/(2*1.3*.4)
    np.testing.assert_allclose(derivative,np.cos(.4*path[:,0]),atol=2e-14)
    report={'base_commit':old['base_commit'],'max_zak_phase_unit_circle_error':phase_error,
            'max_point_geometry_scaling_error':float(np.max(abs(scaled-point))),
            'max_qwz_position_correction_error':float(np.max(abs(point-coefficient-correction))),
            'old_scaled_unsewn_ssh_phase':old['scaled_unsewn_phase'][0],
            'correct_scaled_ssh_phase':float(phases[8]),
            'old_double_position_curvature_error':old['scaled_berry_difference_double_position'],
            'scope':'orthogonal point orbitals, hbar=1, a_cell=1; no full Wannier dipole or nonorthogonal connection'}
    with plot_style('paper'):
        fig,axes=plt.subplots(1,3,figsize=(12.2,3.8),layout='constrained')
        axes[0].plot(centers,reference/np.pi,color='#245b8e',label=r'$\gamma/\pi=1+\tau_2/a$')
        axes[0].plot(centers,(phases%(2*np.pi))/np.pi,'o',ms=4,mfc='white',color='#d77a32',label=r'$2H_{\rm emb}$, sewn')
        axes[0].plot([.4],[old['scaled_unsewn_phase'][0]/np.pi],'x',ms=8,color='#657181',label='Original main, unsewn')
        axes[0].set(xlabel=r'$\tau_2/a$',ylabel=r'Zak phase $\gamma/\pi$ (mod 2)',title='(a) SSH polarization')
        axes[0].legend(fontsize=8,loc='upper left')
        axes[1].plot(ky/np.pi,coefficient,label='Coefficient frame',color='#245b8e')
        axes[1].plot(ky/np.pi,point,label='Point orbitals',color='#d77a32')
        axes[1].plot(ky[::9]/np.pi,scaled[::9],'o',ms=4,mfc='white',color='#3b8a78',label=r'$2H_{\rm emb}$, explicit frame')
        axes[1].set(xlabel=r'$k_y a/\pi$',ylabel=r'$\Omega_{xy}/a^2$',title=r'(b) QWZ, $k_x a=0.31$')
        axes[1].margins(y=.5)
        axes[1].legend(fontsize=8,loc='upper center')
        axes[2].plot(path[:,0]/np.pi,np.cos(.4*path[:,0]),label=r'$\cos(k\Delta\tau)$',color='#245b8e')
        axes[2].plot(path[::6,0]/np.pi,derivative[::6],'o',ms=4,mfc='white',color='#d77a32',label='Embedded gradient')
        axes[2].set(xlabel=r'$ka/\pi$',ylabel=r'$\mathrm{Im}(\partial_k H_{12})/(t\Delta\tau)$',title='(c) Flat-dimer commutator')
        axes[2].legend(fontsize=8,loc='lower center')
        save_figure(fig,output/'orbital-gauge');plt.close(fig)
    np.savez(output/'orbital-gauge.npz',centers=centers,phase=phases,phase_reference=reference,
             ky=ky,coefficient=coefficient,point=point,scaled=scaled,correction=correction,
             dimer_k=path,normalized_derivative=derivative)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='work/orbital-gauge')
    run(p.parse_args().output)
