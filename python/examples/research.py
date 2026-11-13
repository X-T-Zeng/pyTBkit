"""Headless end-to-end example. Run from python/ after installing the package."""
from pathlib import Path
import numpy as np
from tbkit import models, kpath, kmesh, chern_number, berry_curvature, surface_green, transmission
from tbkit.plotting import bandplot


def main(output='work/python-example'):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    model=models.qi_wu_zhang(mass=-1)
    k,distance,ticks=kpath([[0,0],[np.pi,0],[np.pi,np.pi],[0,0]],80)
    bands=model.bands(k,backend='numba')
    ax=bandplot(distance,bands,ticks=distance[ticks],labels=['Gamma','X','M','Gamma'])
    ax.figure.savefig(output/'bands.png',dpi=150)
    grid=kmesh([41,41]); curvature=berry_curvature(model,grid)
    np.savez_compressed(output/'bands_and_curvature.npz',k=k,energies=bands,mesh=grid,curvature=curvature)
    model.save(output/'model.h5')
    print('Chern number:',chern_number(model,(31,31)))
    lead=surface_green(.3,[[0]],[[1]],eta=1e-5)
    print('Perfect chain transmission:',transmission(.3,[[0]],lead,lead))

if __name__=='__main__': main()
