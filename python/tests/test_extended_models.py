import numpy as np
import pytest
from numpy.testing import assert_allclose as close
from scipy import sparse
from tbkit import HR,models,kmesh,Oper
from tbkit.construction import supercell,slab,select_orbitals,direct_sum,_cosets,peierls_finite
from tbkit.sparse import SparseHR
from tbkit.nonorthogonal import NonOrthogonal
from tbkit.symmetry import finite_group,symmetrize,invariant_kp
from tbkit.io import apply_wsvec


@pytest.mark.parametrize('matrix',[[[2,1],[0,1]],[[1,1],[-1,2]],[[-2,1],[0,1]],[[1,4],[0,1]]])
def test_general_supercell_folding(matrix):
    m=models.qi_wu_zhang();matrix=np.array(matrix)
    sc=supercell(m,matrix);k=np.array([.23,-.41])
    _,duals,_=_cosets(matrix.T)
    expected=np.sort(np.concatenate([m.bands(k+2*np.pi*c@np.linalg.inv(matrix).T) for c in duals]))
    close(sc.bands(k),expected,atol=1e-12)
    close(sc.lattice,matrix@m.lattice)


def test_slab_select_and_direct_sum():
    m=models.qi_wu_zhang();s=slab(m,5,axis=0)
    finite=m.cut([5,1],periodic=[False,True])
    close(s.hamiltonian([.2,0]),finite.toarray(),atol=1e-14)
    selected=select_orbitals(m,[1])
    close(selected.hamiltonian([.2,.3]),m.hamiltonian([.2,.3])[1:,1:])
    total=direct_sum(m,selected)
    close(total.bands([.2,.3]),np.sort(np.r_[m.bands([.2,.3]),selected.bands([.2,.3])]))


def test_peierls_plaquette_and_gauge():
    m=HR([[1,0],[-1,0],[0,1],[0,-1]],[[[1]]]*4)
    flux=.23
    field=lambda p:np.array([0,flux*p[0]])
    h=peierls_finite(m,[2,2],field).toarray()
    close(h,h.conj().T)
    product=h[0,2]*h[2,3]*h[3,1]*h[1,0]
    close(product,np.exp(1j*flux),atol=1e-14)
    # A -> A + grad chi with chi=x+2y changes phases by a diagonal gauge.
    hh=peierls_finite(m,[2,2],lambda p:field(p)+[1,2]).toarray()
    g=np.exp(1j*np.array([0,2,1,3]))
    close(hh,g.conj()[:,None]*h*g[None,:],atol=1e-14)
    close(np.linalg.eigvalsh(h),np.linalg.eigvalsh(hh),atol=1e-14)


def test_sparse_storage_and_partial_spectrum():
    n=100;h=sparse.diags(np.arange(n,dtype=float)+.1)
    m=SparseHR({(0,):h,(1,):sparse.eye(n)*.2,(-1,):sparse.eye(n)*.2})
    assert m.values.nbytes<10000
    close(m.sparse_bands([.3],3,sigma=1),[.1+.4*np.cos(.3),1.1+.4*np.cos(.3),2.1+.4*np.cos(.3)],atol=1e-12)
    with pytest.raises(MemoryError):m.to_dense(max_bytes=10)
    dense=SparseHR.from_dense(models.qi_wu_zhang())
    k=kmesh([3,3]);close(dense.hamiltonian(k),models.qi_wu_zhang().hamiltonian(k),atol=1e-14)
    close(dense.gradient(k),dense.gradient(k,backend='numpy'),atol=1e-14)
    close(dense.to_dense().hopping,models.qi_wu_zhang().hopping)


def test_nonorthogonal_eigenvectors_and_gradient():
    h=models.ssh();s=HR([[0],[1],[-1]],[np.eye(2)*2,np.eye(2)*.1,np.eye(2)*.1])
    m=NonOrthogonal(h,s);k=np.array([[.3],[.7]])
    e,u=m.bands(k,eigenvectors=True)
    close(h.hamiltonian(k)@u,(s.hamiltonian(k)@u)*e[:,None,:],atol=1e-13)
    close(u.conj().swapaxes(-1,-2)@s.hamiltonian(k)@u,np.repeat(np.eye(2)[None],2,axis=0),atol=1e-13)
    close(m.band_gradient(k)[:,0],(m.bands(k+1e-5)-m.bands(k-1e-5))/2e-5,atol=1e-9)
    bad=NonOrthogonal(h,HR([[0]],[-np.eye(2)]))
    with pytest.raises(ValueError):bad.bands([0])


def test_symmetry_closure_and_projection():
    inversion=Oper(models.sigma_z,-np.eye(2))
    group=finite_group([inversion]);assert len(group)==2
    broken=models.dirac().to_sympy()
    from tbkit import HK
    h=HK({(0,0):models.sigma_x+models.sigma_z,(1,0):models.sigma_x})
    sym=symmetrize(h,[inversion]);k=np.array([[.2,.3],[-.3,.4]])
    assert inversion.residual(sym,k)<1e-13
    close(sym.hamiltonian(k),models.sigma_z+k[:,0,None,None]*models.sigma_x,atol=1e-14)
    tr=Oper(1j*models.sigma_y,-np.eye(2),antiunitary=True)
    assert len(finite_group([tr]))==2  # quotient by irrelevant global phase


def test_symmetry_symbolic_invariants():
    tr=Oper(1j*models.sigma_y,-np.eye(2),antiunitary=True)
    basis=invariant_kp(2,2,1,[tr])
    assert len(basis)==7 # scalar constant plus six odd spin-linear terms
    for term in basis:
        assert tr.residual(term,[[.2,.3],[-.1,.4]])<1e-13
    inversion=Oper(models.sigma_z,-np.eye(2))
    both=invariant_kp(2,2,1,[tr,inversion])
    assert len(both)==5


def test_wsvec_interpolation(tmp_path):
    m=HR([[0,0,0]],[[[2]]])
    p=tmp_path/'wsvec.dat';p.write_text('written by wannier90\n0 0 0 1 1\n2\n1 0 0\n-1 0 0\n')
    corrected=apply_wsvec(m,p)
    close(corrected.hamiltonian([.3,0,0]),[[2*np.cos(.3)]],atol=1e-14)
    close(corrected.gradient([.3,0,0])[0],[[-2*np.sin(.3)]],atol=1e-14)


def test_sparse_supercell_slab_cut_and_serialization(tmp_path):
    from tbkit import load_hdf5
    m=models.qi_wu_zhang();s=SparseHR.from_dense(m)
    matrix=[[2,1],[0,1]]
    a=s.supercell(matrix);b=m.supercell(matrix)
    close(a.hamiltonian([.2,.3]),b.hamiltonian([.2,.3]),atol=1e-13)
    close(s.cut([3,4],periodic=[False,True]).toarray(),m.cut([3,4],periodic=[False,True]).toarray(),atol=1e-13)
    close(s.slab(4).hamiltonian([.2,.3]),slab(m,4).hamiltonian([.2,.3]),atol=1e-13)
    a.save(tmp_path/'sparse.h5');loaded=load_hdf5(tmp_path/'sparse.h5')
    assert isinstance(loaded,SparseHR)
    close(loaded.hamiltonian([.2,.3]),a.hamiltonian([.2,.3]),atol=1e-13)


def test_model_algebra_and_all_model_schemas(tmp_path):
    from tbkit import load_hdf5
    a=models.ssh(.4);b=models.ssh(.8)
    c=2*a-b
    assert isinstance(c,HR)
    close(c.hamiltonian([.3]),2*a.hamiltonian([.3])-b.hamiltonian([.3]),atol=1e-14)
    for m in [models.dirac(),models.qi_wu_zhang().to_htrig()]:
        path=tmp_path/(type(m).__name__+'.h5');m.save(path);loaded=load_hdf5(path)
        close(loaded.hamiltonian([.2,.3]),m.hamiltonian([.2,.3]),atol=1e-14)
        close((m+m).gradient([.2,.3]),2*m.gradient([.2,.3]),atol=1e-14)


def test_single_layer_without_onsite_is_zero():
    m=HR([[1],[-1]],[[[1]],[[1]]])
    zero=slab(m,1)
    close(zero.hamiltonian([.2]),[[0.]])
    close(zero.hamiltonian([.2],backend='numpy'),[[0.]])
    close(SparseHR.from_dense(m).slab(1).hamiltonian([.2]),[[0.]])
