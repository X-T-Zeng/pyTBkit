"""Reproducible research examples; each checks an independent physical result.

Run from repository root: python python/examples/full_suite.py --output work/examples
"""
import argparse
import json
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose as close
from tbkit import models,HR,Oper
from tbkit.structure import Structure
from tbkit.slater_koster import from_structure
from tbkit.orbitals import angular_momentum
from tbkit.sparse import SparseHR
from tbkit.symmetry import invariant_kp
from tbkit.responses import integrate_bz,response_kernel
from tbkit.topology import nested_polarization,inversion_indicators,mirror_chern
from tbkit.construction import direct_sum
from tbkit.continuum import landau_hamiltonian
from tbkit.nonorthogonal import NonOrthogonal
from tbkit.spin import SpinModel
from tbkit.circuit import Component,Subckt


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result={}
    # 1. Structure -> neighbors -> Slater-Koster -> dense and sparse interpolation.
    chain=Structure(np.diag([1.,10.,10.]),[[0,0,0]],['C'])
    chain.to_poscar(output/'POSCAR')
    model=from_structure(Structure.from_poscar(output/'POSCAR'),[['s']],1.01,{'ss_sigma':-1.},onsite=[.2])
    k=np.column_stack([np.linspace(-np.pi,np.pi,101),np.zeros((101,2))])
    dense=model.bands(k);sparse=SparseHR.from_dense(model)
    close(dense,.2-2*np.cos(k[:,:1]),atol=1e-13)
    close(sparse.bands(k),dense,atol=1e-13)
    model.save(output/'sk_chain.h5');np.savez(output/'sk_chain.npz',k=k,energy=dense)
    result['structure_slater_koster']={'norb':model.norb,'max_error':float(np.max(abs(dense-(.2-2*np.cos(k[:,:1])))))}
    # 2. Nonorthogonal H,S generalized bands and derivatives.
    ssh=models.ssh();overlap=HR([[0],[1],[-1]],[2*np.eye(2),.1*np.eye(2),.1*np.eye(2)])
    no=NonOrthogonal(ssh,overlap);kk=np.linspace(0,np.pi,51)[:,None]
    close(no.bands(kk),ssh.bands(kk)/(2+.2*np.cos(kk)),atol=1e-13)
    np.savez(output/'nonorthogonal.npz',k=kk,energy=no.bands(kk),gradient=no.band_gradient(kk))
    result['nonorthogonal']={'passed':True}
    # 3. Original response equations -> streaming BZ integral -> Chern coefficient.
    qwz=models.qi_wu_zhang(-1)
    chern_est=float(integrate_bz(lambda k:response_kernel(qwz,k,[0.],temperature=0.),[51,51])[0]*2*np.pi)
    assert abs(chern_est-1)<1e-7
    nonlinear=integrate_bz(lambda k:response_kernel(qwz,k,[-.4,0,.4],kind='soahc',indices=(0,1,1),temperature=300.),[15,15])
    np.savez(output/'responses.npz',mu=[-.4,0,.4],soahc=nonlinear)
    result['streaming_responses']={'chern_integral':chern_est,'soahc':nonlinear.tolist()}
    # 4. Nested topology, mirror sectors, and inversion strong/weak TI indices.
    nested=nested_polarization(models.bbh(.5),(25,25));assert abs(nested-.5)<1e-9
    mirror=np.diag([1j,1j,-1j,-1j]);sector=mirror_chern(direct_sum(qwz,models.qi_wu_zhang(1)),mirror,shape=(21,21))
    ti=models.topological_insulator(2)
    indicator=inversion_indicators(ti,np.kron(models.sigma_z,np.eye(2)),np.kron(np.eye(2),1j*models.sigma_y),occupied=2)
    assert indicator['strong_z2']==1
    result['topology']={'nested_polarization':nested,'mirror_chern':sector,'strong_z2':indicator['strong_z2'],'weak_z2':indicator['weak_z2']}
    # 5. Symmetry-constrained k.p and Landau levels with basis truncation check.
    tr=Oper(1j*models.sigma_y,-np.eye(2),antiunitary=True)
    allowed=invariant_kp(2,2,1,[tr]);assert len(allowed)==7
    landau=np.linalg.eigvalsh(landau_hamiltonian(models.dirac(.3),20))
    # Paired n>=1 bulk levels; two truncation-sensitive +/- mass states also exist.
    for n in range(1,10):assert np.min(abs(landau-np.sqrt(.3**2+2*n)))<1e-12
    np.save(output/'landau.npy',landau)
    result['symmetry_landau']={'allowed_terms':len(allowed),'lowest_positive_bulk':float(np.sqrt(.3**2+2))}
    # 6. Classical spin damping and a finite current-biased circuit.
    spins=SpinModel(1,[],[],field=[0,0,1.])
    initial=np.array([[1.,0,0]])
    trajectory=spins.evolve(initial,.02,200,damping=.4)
    assert spins.energy_field(trajectory[-1])[0]<spins.energy_field(initial)[0]
    np.save(output/'spin_trajectory.npy',trajectory)
    circuit=Subckt(2,[(0,1,Component('R',1000)),(1,-1,Component('R',1000))])
    volts=circuit.voltages(0,[.001,0]);close(volts,[2,1],atol=1e-14)
    circuit.to_spice(output/'resistor_chain.cir')
    result['spin_circuit']={'final_spin_z':float(trajectory[-1,0,2]),'voltages':volts.real.tolist()}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='work/python-full-examples')
    print(json.dumps(run(parser.parse_args().output),indent=2))
