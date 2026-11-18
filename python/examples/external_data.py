"""Use actual TBkit repository DFT fixtures and retain explicit absolute energies."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tbkit.dft import read_eigenval,read_kpoints,read_procar
from tbkit.wannier import read_wout
from tbkit.plotting import plot_style,bandplot,save_figure


def run(output):
    project=Path(__file__).resolve().parents[2]
    config=json.loads((project/'python/upstream.json').read_text())
    root=project/config['checkout'];output=Path(output);output.mkdir(parents=True,exist_ok=True)
    folder=root/'example/example09-TopoIndices/MirrorChern'
    eigen=read_eigenval(folder/'EIGENVAL');path=read_kpoints(folder/'KPOINTS');wannier=read_wout(folder/'wannier90.wout')
    np.testing.assert_allclose(eigen.kpoints,path.expanded(),atol=6e-8)
    # This pinned line-mode fixture repeats each connected junction twice.
    # Mark the midpoint of the two indices; do not call index a k-space metric.
    segments=path.points.reshape(-1,2,3)
    np.testing.assert_allclose(segments[:-1,1],segments[1:,0])
    tick_positions=[0,*[i*path.count-.5 for i in range(1,len(segments))],len(eigen.kpoints)-1]
    endpoint_labels=[path.labels[0],*path.labels[1::2]]
    tick_labels=[r'$\Gamma$' if label.upper() in ('G','GAMMA') else label for label in endpoint_labels]
    with plot_style():
        fig,ax=plt.subplots(figsize=(5.6,3.5))
        bandplot(np.arange(len(eigen.kpoints)),eigen.energies[:,:,0],ax=ax,linewidth=.65,
                 ticks=tick_positions,labels=tick_labels,
                 ylabel='Absolute VASP eigenvalue (eV)')
        ax.set(xlabel='DFT path index',title='First-principles band structure')
        save_figure(fig,output/'dft_bands');plt.close(fig)
    np.savez_compressed(output/'dft_and_wannier.npz',k=eigen.kpoints,energy=eigen.energies,centers=wannier.centers,spreads=wannier.spreads)
    projections=read_procar(root/'example/example01-DFT_PostProcessing/VASP_pbandplot/Graphene/PROCAR')
    pz=projections.orbitals.index('pz')
    # Component 0 is charge, the remaining 3 are signed spin expectation tables.
    pz_weight=projections.projections[0,:,:,0,:,pz].sum(axis=-1)
    np.savez_compressed(output/'graphene_projection.npz',k=projections.kpoints,energy=projections.energies,pz=pz_weight)
    report={'reference_repository':config['repository'],'reference_commit':config['commit'],'dft_shape':eigen.energies.shape,'wannier_centers':len(wannier.centers),'procar_shape':projections.projections.shape,'projection_components':['charge','mx','my','mz']}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='work/python-external-data')
    print(json.dumps(run(parser.parse_args().output),indent=2))
