"""Headless end-to-end example. Run from python/ after installing the package."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tbkit import models, kpath, kmesh, chern_number, berry_curvature, surface_green, transmission
from tbkit.plotting import bandplot,plot_style,save_figure


def main(output='work/python-example'):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    model=models.qi_wu_zhang(mass=-1)
    k,distance,ticks=kpath([[0,0],[np.pi,0],[np.pi,np.pi],[0,0]],80)
    bands=model.bands(k,backend='numba')
    with plot_style():
        ax=bandplot(distance,bands,ticks=distance[ticks],labels=[r'$\Gamma$','X','M',r'$\Gamma$'],
                    reference=0,ylabel=r'$E/t$')
        ax.set_title('Qi–Wu–Zhang model')
        save_figure(ax.figure,output/'bands')
        plt.close(ax.figure)
    grid=kmesh([41,41]); curvature=berry_curvature(model,grid)
    np.savez_compressed(output/'bands_and_curvature.npz',k=k,energies=bands,mesh=grid,curvature=curvature)
    model.save(output/'model.h5')
    print('Chern number:',chern_number(model,(31,31)))
    lead=surface_green(.3,[[0]],[[1]],eta=1e-5)
    print('Perfect chain transmission:',transmission(.3,[[0]],lead,lead))

if __name__=='__main__': main()
