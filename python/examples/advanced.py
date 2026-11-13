"""Advanced validation: nonsymmorphic sewing, no-inversion Z2, CG and leads."""
import argparse
import json
from pathlib import Path
import numpy as np
from tbkit import (Structure,from_structure,crystal_operations,spacegroup,crystal_residual,
                   models,z2_wilson,coupled_basis,HR,principal_layer,chain_transmission)


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    s=Structure(np.array([[0,.5,.5],[.5,0,.5],[.5,.5,0]])*5.43,[[0,0,0],[.25,.25,.25]],['Si','Si'])
    orbitals=[['s'],['s']];m=from_structure(s,orbitals,2.4,{'ss_sigma':-1.})
    errors=[crystal_residual(m,s,op,orbitals,[[.137,.247,.317]]) for op in crystal_operations(s)]
    assert max(errors)<1e-12
    m.save(output/'diamond_s_model.h5');s.to_poscar(output/'POSCAR-diamond')
    qsh=models.quantum_spin_hall(-1.,mixing=.17)
    tr=np.kron(1j*models.sigma_y,np.eye(2))
    z2=z2_wilson(qsh,tr,diagnostics=True)
    assert z2['z2']==1
    np.savez(output/'z2_flow.npz',ky=z2['ky_fractional'],phases=z2['phases'],reference=z2['reference_phase'])
    c,labels,_=coupled_basis(1,.5)
    (output/'clebsch_gordan.txt').write_text(str(c)+'\n'+str(labels)+'\n')
    chain=HR([[1],[-1],[2],[-2]],[[[1]],[[1]],[[.2]],[[.2]]])
    lead=principal_layer(chain)
    trans=chain_transmission(lead,.3,3);assert abs(trans-1)<1e-6
    # The optional compiled backend is required by the complete cloud install.
    k=np.array([[.2,.3,.4],[-.1,.4,.7]])
    ti=models.topological_insulator()
    err=float(np.max(abs(ti.hamiltonian(k)-ti.hamiltonian(k,backend='fortran'))))
    assert err<1e-13
    result={'diamond_spacegroup':spacegroup(s)['number'],'max_crystal_covariance_error':max(errors),
            'z2_without_inversion':z2['z2'],'cg_dimension':c.shape[0],
            'principal_layer_width':lead.width,'long_range_chain_transmission':trans,
            'numba_fortran_error':err}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='work/python-advanced-examples')
    print(json.dumps(run(parser.parse_args().output),indent=2))
