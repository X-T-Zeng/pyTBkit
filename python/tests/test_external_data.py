from pathlib import Path
from reference_source import source_root
import numpy as np
from numpy.testing import assert_allclose as close
from tbkit.dft import read_eigenval,read_kpoints,read_procar
from tbkit.wannier import read_win,read_wout,BOHR

ROOT=source_root()


def test_real_repository_eigenval_and_kpath():
    base=ROOT/'example/example09-TopoIndices/MirrorChern'
    e=read_eigenval(base/'EIGENVAL');k=read_kpoints(base/'KPOINTS')
    assert e.energies.shape==(153,152,1) and e.nelectrons==124
    close(e.energies[0,0,0],-18.864555,atol=1e-10)
    close(k.expanded(),e.kpoints,atol=6e-8)
    close(e.weights.sum(),1,atol=1e-7)
    assert k.labels[0]=='GAMMA' and k.mode=='line'


def test_spin_eigenval_and_automatic_mesh(tmp_path):
    path=tmp_path/'E';path.write_text('h\nh\nh\nh\nh\n2 1 2\n\n0 0 0 1\n1 -1 -2 .9 .8\n2 1 2 .1 .2\n')
    e=read_eigenval(path);close(e.energies,[[[-1,-2],[1,2]]])
    path.write_text('mesh\n0\nMonkhorst\n2 2 1\n0 0 0\n')
    k=read_kpoints(path);assert k.expanded().shape==(4,3)
    close(k.expanded()[0],[-.25,-.25,0])


def test_real_noncollinear_procar():
    p=read_procar(ROOT/'example/example01-DFT_PostProcessing/VASP_pbandplot/Graphene/PROCAR')
    assert p.energies.shape==(1,180,33)
    assert p.projections.shape==(1,180,33,4,2,10)
    close(p.energies[0,0,0],-19.55082147,atol=1e-10)
    close(p.projections[0,0,0,0,:,0],[.236,.236])
    assert p.orbitals[-1]=='tot'
    # Rounded print precision is 1e-3; each component's orbital sum is consistent.
    close(p.projections[...,0,:-1].sum(axis=-1),p.projections[...,0,-1],atol=.005)


def test_wannier_input_and_real_final_state(tmp_path):
    path=tmp_path/'seed.win'
    path.write_text('num_wann = 2\nspinors = true\nfermi_energy = 1.2d0\nbegin unit_cell_cart\nbohr\n2 0 0\n0 2 0\n0 0 2\nend unit_cell_cart\nbegin atoms_frac\nC 0 0 0\nend atoms_frac\nbegin projections\nC:pz\nend projections\n')
    w=read_win(path);assert w.parameters['spinors'] is True
    close(w.structure.lattice,2*BOHR*np.eye(3));assert w.blocks['projections']==('C:pz',)
    output=read_wout(ROOT/'example/example09-TopoIndices/MirrorChern/wannier90.wout')
    assert output.centers.shape==(40,3)
    close(output.centers[0],[.058867,3.687964,18.407288],atol=1e-8)
    close(output.spreads[0],22.07794542,atol=1e-10)
