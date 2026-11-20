"""Independent reciprocal duality, Bloch phases, graph and measure references."""
import itertools
import warnings
import numpy as np
import pytest
from numba import njit
from numpy.testing import assert_allclose as close
from tbkit import HR,SparseHR,Structure,NonOrthogonal,kmesh,models,chern_number
from tbkit.lattice import checked_lattice,reciprocal_lattice,same_lattice
from tbkit.construction import direct_sum
from tbkit.responses import integrate_bz
from tbkit.slater_koster import from_structure,slater_koster
from tbkit import backends


L=np.array([[1.3,.2,0.],[.1,1.1,.3],[0.,.1,1.6]])
SCALES=(1e-200,1e-9,1.,1e9,1e200)


def cross_reciprocal(a):
    volume=np.dot(a[0],np.cross(a[1],a[2]))
    return 2*np.pi*np.array([np.cross(a[1],a[2]),np.cross(a[2],a[0]),np.cross(a[0],a[1])])/volume


@pytest.mark.parametrize('scale',SCALES)
def test_reciprocal_duality_and_cartesian_mesh(scale):
    a=scale*L;b=reciprocal_lattice(a)
    close(b*scale,cross_reciprocal(L),rtol=2e-14,atol=2e-15)
    close(a@b.T,2*np.pi*np.eye(3),rtol=0,atol=4e-15)
    q=np.array(list(itertools.product(np.arange(3)/3,np.arange(4)/4,np.arange(2)/2)))
    close(kmesh([3,4,2],lattice=a)*scale,q@cross_reciprocal(L),rtol=2e-14,atol=2e-15)
    # A reciprocal shift leaves exp(+ikR) unchanged for every integer R.
    r=np.array([[1,2,-1],[-2,0,3]])@a;k=q@b
    close(np.exp(1j*((k+b[1])@r.T)),np.exp(1j*(k@r.T)),atol=2e-14)


@pytest.mark.parametrize('scale',SCALES)
@pytest.mark.parametrize('backend',('numpy','numba','fortran'))
def test_dense_sparse_trig_analytic_chain_and_gradient(scale,backend):
    # Three-dimensional storage, one periodic hopping direction.
    a=scale*np.diag([1.,4.,4.]);q=np.array([.2,.7,1.1]);k=np.column_stack((q/scale,np.zeros((3,2))))
    model=HR([[-1,0,0],[1,0,0]],[[[1.]],[[1.]]],lattice=a)
    close(model.bands(k,backend=backend)[:,0],2*np.cos(q),rtol=2e-14,atol=0)
    close(model.gradient(k,backend=backend)[:,0,0,0]/scale,-2*np.sin(q),rtol=2e-14,atol=0)
    if backend!='fortran':
        for other in (SparseHR.from_dense(model),model.to_htrig(),model.to_htrig().to_hr()):
            close(other.hamiltonian(k,backend=backend)[:,0,0],2*np.cos(q),rtol=2e-14,atol=0)
            close(other.lattice/scale,a/scale,rtol=0,atol=0)
    if backend=='numpy':
        from tbkit.circuit import Hckt,Component
        circuit=Hckt(1,3,[(0,0,(1,0,0),Component('R',1.))],lattice=a)
        close(circuit.laplacian(k,0.)[:,0,0],2-2*np.cos(q),rtol=2e-14,atol=2e-16)


@pytest.mark.parametrize('scale',SCALES)
def test_cube_neighbors_keep_original_images_distances_and_sk_bands(scale):
    structure=Structure(scale*np.eye(3),[[0.,0.,0.]],['X'])
    n=structure.neighbors(1.1*scale)
    expected={(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)}
    assert set(map(tuple,n.cells))==expected
    close(n.distance/scale,1.,rtol=2e-14,atol=0)
    close(n.displacement/scale,n.cells,rtol=0,atol=0)
    model=from_structure(structure,[['s']],1.1*scale,{'ss_sigma':1.})
    fractional=np.array([[.13,.27,.31],[.2,.1,.4]])
    k=2*np.pi*fractional/scale
    close(model.bands(k)[:,0],2*np.cos(2*np.pi*fractional).sum(axis=1),rtol=2e-14,atol=1e-14)


@pytest.mark.parametrize('scale',(1e-200,1e-9,1.,1e200))
def test_sk_direction_is_scale_free_and_reversal_parity(scale):
    d=np.array([1.,2.,-3.]);parameters=dict(ss_sigma=1,sp_sigma=.7,sd_sigma=.3,pp_sigma=2,pp_pi=-.2,pd_sigma=.8,pd_pi=.4,dd_sigma=1.1,dd_pi=.6,dd_delta=.2)
    reference=slater_koster(d,parameters)
    close(slater_koster(scale*d,parameters),reference,rtol=2e-14,atol=5e-16)
    close(slater_koster(-scale*d,parameters),reference.T,rtol=2e-14,atol=5e-16)


def test_fixed_length_threshold_counterexample_changes_physical_spectrum():
    structure=Structure(np.eye(3),[[0,0,0],[5e-11,0,0]],['A','B'])
    neighbors=structure.neighbors(8e-11,periodic=(False,False,False))
    assert len(neighbors.i)==2
    close(neighbors.distance,[5e-11,5e-11],rtol=2e-14,atol=0)
    model=from_structure(structure,[['s'],['s']],8e-11,{'ss_sigma':1.},periodic=(False,False,False))
    close(model.bands([0.,0.,0.]),[-1.,1.],rtol=2e-14,atol=0)
    # Explicit tol keeps its absolute-length meaning and can intentionally omit.
    assert len(structure.neighbors(8e-11,tol=1e-10,periodic=(False,False,False)).i)==0


@pytest.mark.parametrize('scale',(1e-100,1.,1e100))
def test_skew_unwrapped_geometry_against_independent_image_enumeration(scale):
    positions=np.array([[1.1,0.,.2],[-.2,.3,.1]])
    cutoff=.9;reference={}
    # Independent finite R enumeration. Chosen cell/cutoff makes [-3,3]^3 ample.
    for i,j in itertools.product(range(2),repeat=2):
        for r in itertools.product(range(-3,4),repeat=3):
            d=(positions[j]+r-positions[i])@L;distance=np.sqrt(np.dot(d,d))
            if 0.<distance<=cutoff:reference[(i,j,*r)]=(d,distance)
    structure=Structure(scale*L,positions,['X','Y']);n=structure.neighbors(scale*cutoff,tol=0.)
    cartesian=Structure(scale*L,positions@(scale*L),['X','Y'],cartesian=True)
    close(cartesian.positions,positions,rtol=2e-14,atol=2e-16)
    actual={(int(i),int(j),*r):(d/scale,length/scale) for i,j,r,d,length in zip(n.i,n.j,n.cells,n.displacement,n.distance)}
    assert actual.keys()==reference.keys()
    for key in actual:
        close(actual[key][0],reference[key][0],rtol=3e-14,atol=1e-15)
        close(actual[key][1],reference[key][1],rtol=3e-14,atol=0)


@pytest.mark.parametrize('scale',SCALES)
def test_lattice_compatibility_does_not_silently_merge_distinct_bloch_bases(scale):
    a=HR([[-1],[1]],[[[1.]],[[1.]]],lattice=[[scale]])
    b=HR([[-1],[1]],[[[1.]],[[1.]]],lattice=[[scale*(1+1e-7)]])
    for operation in (lambda:a+b,lambda:direct_sum(a,b),lambda:NonOrthogonal(a,b)):
        with pytest.raises(ValueError,match='lattice'):operation()
    assert same_lattice([[scale]],[[scale*(1+1e-14)]])
    assert not same_lattice([[scale]],[[scale*(1+1e-7)]])
    assert not same_lattice([[scale]],[[np.nextafter(scale,np.inf)]],rtol=0)


@pytest.mark.parametrize('scale',(1e-200,1e-30,1.,1e30,1e200))
def test_bz_average_and_lazy_physical_measure(scale):
    a=scale*L
    def integrand(k):return 3+np.cos(k@a[0])
    with warnings.catch_warnings():
        warnings.simplefilter('error',RuntimeWarning)
        close(integrate_bz(integrand,[5,4,3],lattice=a,normalization='average'),3.,rtol=2e-14,atol=0)
    if 1e-100<scale<1e100:
        volume=abs(np.linalg.det(L))*scale**3
        close(integrate_bz(integrand,[5,4,3],lattice=a,normalization='states')*volume,3.,rtol=2e-13,atol=0)
        close(integrate_bz(integrand,[5,4,3],lattice=a,normalization='integral')*volume/(2*np.pi)**3,3.,rtol=2e-13,atol=0)
    else:
        with pytest.raises(ValueError,match='measure not representable'):
            integrate_bz(integrand,[3,3,3],lattice=a,normalization='states')


@pytest.mark.parametrize('scale',(1e-60,1.,1e60))
def test_chern_number_under_length_unit_change(scale):
    base=models.qi_wu_zhang()
    model=HR(base.vectors,base.hopping,lattice=scale*np.array([[1.7,.2],[0.,1.3]]))
    close(chern_number(model,(17,17)),1.,atol=1e-12)


def test_real_rank_orientation_and_representability_domains():
    for lattice in (np.zeros((3,3)),np.ones((3,3)),np.diag([1.,1.,1e-18]),np.full((3,3),np.nan),np.ones((2,3))):
        with pytest.raises(ValueError):checked_lattice(lattice)
    with pytest.raises(ValueError,match='real'):checked_lattice(np.eye(3,dtype=complex))
    with pytest.raises(ValueError,match='real'):slater_koster([1j,0,0],{})
    left=L.copy();left[0]*=-1
    close(left@reciprocal_lattice(left).T,2*np.pi*np.eye(3),atol=3e-15)
    s=Structure(1e-200*np.eye(3),[[0,0,0]],['X'])
    close(s.log_volume,3*np.log(1e-200),rtol=0,atol=1e-12)
    with pytest.raises(ValueError,match='not representable'):print(s.volume)
    with pytest.raises(MemoryError,match='max_images'):s.neighbors(2e-200,max_images=10)
    for cutoff in (0.,-1.,np.inf,np.nan):
        with pytest.raises(ValueError):s.neighbors(cutoff)
    for tol in (-1.,np.inf,np.nan):
        with pytest.raises(ValueError):s.neighbors(1e-200,tol=tol)


def test_nopython_fourier_arrays_retain_physical_units():
    @njit
    def pipeline(k,r,h):return backends.fourier(k,r,h)
    h=np.array([[[.5]],[[.5]]],complex)
    for scale in (1e-200,1e200):
        r=np.array([[-scale],[scale]]);k=np.array([[.2/scale],[.7/scale]])
        close(pipeline(k,r,h)[:,0,0],np.cos([.2,.7]),rtol=2e-14,atol=0)
    assert pipeline.nopython_signatures
