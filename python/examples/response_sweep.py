"""Prepare once, scan tensors/mu/T; point-orbital frame is an explicit choice."""
import argparse
import json
from pathlib import Path
import numpy as np
from tbkit import (HR,models,kmesh,chern_number,prepare_spectral,prepare_geometry,
                   response_from_geometry)


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    base=models.qi_wu_zhang(-1)
    model=HR(base.vectors,base.hopping,orbitals=[[0,0],[.37,.11]])
    mesh=kmesh([41,41]);mu=np.linspace(-2.5,2.5,51);temperatures=np.array([50.,200.,500.])
    spectral=prepare_spectral(model,mesh,frame='point-orbital')
    geometry=prepare_geometry(spectral)
    result={}
    for kind,indices in [('berry',(0,1)),('soahc',(0,1,1)),('ncte',(0,1,1))]:
        result[kind]=np.array([response_from_geometry(geometry,mu,kind=kind,indices=indices,temperature=t).mean(axis=0) for t in temperatures])
    # Point-orbital embedding changes local curvature by a total derivative.
    # The full BZ occupied curvature integral still gives the coefficient Chern.
    estimate=float(response_from_geometry(geometry,[0.],kind='berry',indices=(0,1),temperature=0).mean()*2*np.pi)
    expected=chern_number(model,(31,31))
    if abs(estimate-expected)>1e-6:raise AssertionError('curvature integral failed convergence check')
    np.savez_compressed(output/'response_sweep.npz',mu=mu,temperature=temperatures,**result)
    report={'frame':geometry.frame,'nk':len(mesh),'chemical_potentials':len(mu),
            'temperatures_K':temperatures.tolist(),'k_batch_eigensystem_preparations':1,
            'response_evaluations':9,'spectral_bytes':spectral.nbytes,'geometry_bytes':geometry.nbytes,
            'chern_integral':estimate,'chern_reference':expected,
            'normalization':'BZ average of clean mathematical kernels; no SI prefactor or lifetime model'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='work/response-sweep')
    print(json.dumps(run(parser.parse_args().output),indent=2))
