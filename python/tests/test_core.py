import numpy as np
import pytest
from numpy.testing import assert_allclose as close
from tbkit import HR, HK, Htrig, Oper, kmesh, kpath, models

@pytest.mark.parametrize('model',[models.ssh(),models.qi_wu_zhang(),models.dirac()])
def test_backends_and_gradients(model):
    rng=np.random.default_rng(12); k=rng.uniform(-1,1,(9,model.dim))
    close(model.hamiltonian(k),model.hamiltonian(k,backend='numpy'),atol=1e-14)
    close(model.gradient(k),model.gradient(k,backend='numpy'),atol=1e-14)
    for a in range(model.dim):
        dk=np.zeros(model.dim); dk[a]=1e-5
        close(model.gradient(k)[:,a],(model.hamiltonian(k+dk)-model.hamiltonian(k-dk))/(2e-5),atol=1e-9)
    close(model.bands(k,chunk_size=3),model.bands(k,backend='numpy'),atol=1e-13)
    e,u=model.bands(k,eigenvectors=True)
    close(model.hamiltonian(k)@u,u*e[:,None,:],atol=1e-13)


def test_ssh_analytic_and_finite_chain():
    m=models.ssh(); k=np.linspace(-np.pi,np.pi,21)[:,None]
    e=np.sqrt(.6**2+1+1.2*np.cos(k[:,0]))
    close(m.bands(k),np.stack([-e,e],axis=1),atol=1e-14)
    finite=m.cut([20]).toarray()
    close(finite,finite.conj().T)
    e=np.linalg.eigvalsh(finite)
    assert np.max(np.abs(e[19:21]))<1e-4
    ring=m.cut([5],periodic=[True]).toarray()
    close(np.linalg.eigvalsh(ring),np.sort(m.bands(2*np.pi*np.arange(5)[:,None]/5).ravel()),atol=1e-13)


def test_qwz_analytic_and_conversions():
    m=models.qi_wu_zhang(); k=np.array([[.1,.2],[-.7,.9]])
    e=np.sqrt(np.sin(k[:,0])**2+np.sin(k[:,1])**2+(-1+np.cos(k[:,0])+np.cos(k[:,1]))**2)
    close(m.bands(k),np.stack([-e,e],axis=1))
    trig=m.to_htrig()
    close(trig.hamiltonian(k),m.hamiltonian(k),atol=1e-14)
    close(trig.gradient(k),m.gradient(k),atol=1e-14)
    close(trig.to_hr().hamiltonian(k),m.hamiltonian(k),atol=1e-14)
    close(trig.hamiltonian(k),trig.hamiltonian(k,backend='numpy'),atol=1e-14)
    small=k*1e-3
    close(m.to_hk(2).hamiltonian(small),m.hamiltonian(small),atol=2e-10)


def test_supercell_zone_folding():
    m=models.ssh(); sc=m.supercell([3]); k=.17
    expected=np.sort(np.concatenate([m.bands([k+2*np.pi*j/3]) for j in range(3)]))
    close(sc.bands([k]),expected,atol=1e-13)


def test_coordinate_and_cache_isolation():
    m=HR([[0],[1],[-1]],[[[0]],[[1]],[[1]]],lattice=[[2]])
    close(m.hamiltonian([.25],coordinates='fractional'),m.hamiltonian([np.pi/4]),atol=1e-14)
    close(m.gradient([.25],coordinates='fractional'),[[[-4*np.pi]]],atol=1e-13)
    a=models.ssh(.4); b=models.ssh(.8)
    assert not np.allclose(a.hamiltonian([0]),b.hamiltonian([0]))
    c=a.with_hopping([0],np.zeros((2,2)))
    assert not np.allclose(c.bands([0]),a.bands([0]))
    with pytest.raises(ValueError): a.hopping[0,0,0]=3


def test_nonhermitian_rejected_and_repaired():
    m=HR([[1]],[[[1]]])
    with pytest.raises(ValueError): m.bands([.4])
    close(m.hermitize().bands([.4]),[np.cos(.4)])


def test_symbolic_and_regularization():
    import sympy as s
    x,y=s.symbols('x y',real=True)
    m=HK.from_sympy(s.Matrix([[2+x*x,x-s.I*y],[x+s.I*y,-2-x*x]]),(x,y))
    close(np.asarray(m.to_sympy((x,y)).subs({x:.2,y:.3})).astype(complex),m.hamiltonian([.2,.3]))
    close(m.kp2tb().hamiltonian([1e-4,2e-4]),m.hamiltonian([1e-4,2e-4]),atol=2e-12)
    with pytest.raises(ValueError): HK.from_sympy(s.eye(2)*s.Symbol('m'),(x,y))


def test_geometry_and_operator():
    assert kmesh([3,4]).shape==(12,2)
    k,d,t=kpath([[0,0],[1,0],[1,1]],3)
    close(d[t],[0,1,2])
    m=models.dirac()
    # inversion sigma_z H(-k) sigma_z = H(k)
    op=Oper(models.sigma_z,-np.eye(2))
    assert op.residual(m,k)<1e-13
    close(op.apply(m).gradient(k),m.gradient(k),atol=1e-14)
    tr=Oper(np.eye(2),-np.eye(1),antiunitary=True)
    assert tr.residual(models.ssh(),[[.2],[.8]])<1e-13


def test_dos_normalization():
    m=models.ssh(); grid=np.linspace(-3,3,2001)
    dos=m.dos(kmesh([30]),grid,width=.05)
    assert abs(np.trapezoid(dos,grid)-2)<1e-8


def test_sparse():
    h=np.diag(np.arange(6.)+.1)
    m=HR([[0]],[h])
    close(m.sparse_hamiltonian([0]).toarray(),h)
    close(m.sparse_bands([0],2,sigma=1),[.1,1.1],atol=1e-12)


def test_user_numba_pipeline():
    from numba import njit
    from tbkit import kernels
    @njit
    def occupied_energy(k,r,h):
        matrices=kernels.fourier(k,r,h)
        total=0.0
        for q in range(len(k)):
            total+=np.linalg.eigvalsh(matrices[q])[0]
        return total/len(k)
    model=models.qi_wu_zhang()
    k=kmesh([5,5])
    close(occupied_energy(k,model.cartesian_vectors,model.hopping),model.bands(k)[:,0].mean())
