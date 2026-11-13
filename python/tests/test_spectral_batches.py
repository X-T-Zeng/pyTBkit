"""Reuse, independence, degeneracy policy and frame physics for spectral batches."""
import numpy as np
import pytest
from numpy.testing import assert_allclose as close
from numba import njit
from tbkit import (HR,models,prepare_spectral,prepare_geometry,response_from_geometry,
                   geometric_tensors,response_kernel,embed_orbitals,kernels)
from tbkit.base import TBkit
from tbkit.spectral import SpectralBatch


class Counted(TBkit):
    def __init__(self,model):self.model=model;self.dim=model.dim;self.norb=model.norb;self.solves=0
    def hamiltonian(self,k,*,backend='numba'):return self.model.hamiltonian(k,backend=backend)
    def gradient(self,k,*,backend='numba'):return self.model.gradient(k,backend=backend)
    def bands(self,*args,**kwargs):self.solves+=1;return self.model.bands(*args,**kwargs)


@pytest.mark.parametrize('backend',['numpy','numba'])
def test_reuse_without_repeated_eigensolve_or_model_dependency(backend):
    counted=Counted(models.dirac(.4));k=np.array([[.1,.3],[.5,-.2]])
    spectral=prepare_spectral(counted,k,backend=backend)
    geometry=prepare_geometry(spectral,backend=backend)
    assert counted.solves==1
    # Mutating/replacing the source object must not alter the snapshot.
    counted.model=models.dirac(3.)
    for temperature in (30.,100.,300.):
        for kind,indices in [('berry',(0,1)),('ncte',(0,1,1)),('soahc',(0,1,1))]:
            a=response_from_geometry(geometry,[-.3,0,.3],kind=kind,indices=indices,temperature=temperature)
            b=response_kernel(models.dirac(.4),k,[-.3,0,.3],kind=kind,indices=indices,temperature=temperature,backend='numpy')
            close(a,b,atol=1e-12)
    assert counted.solves==1
    assert spectral.nbytes>0 and geometry.nbytes>0
    with pytest.raises(ValueError):spectral.energy[0,0]=10
    with pytest.raises(ValueError):geometry.berry[0,0,1,0]=10


def test_independent_analytic_dirac_metric_and_frame_invariance():
    model=models.dirac(.4);k=np.array([[.1,.3],[.5,-.2]])
    result=prepare_geometry(prepare_spectral(model,k))
    r=np.sqrt((k*k).sum(axis=1)+.4**2)
    # g_ab=(delta_ab/r^2-k_a*k_b/r^4)/4 for either two-band eigenstate.
    for a in range(2):
        for b in range(2):
            expected=((a==b)/r**2-k[:,a]*k[:,b]/r**4)/4
            close(result.metric[:,a,b,0],expected,atol=1e-13)
            close(result.metric[:,a,b,1],expected,atol=1e-13)
    close(result.berry[:,0,1,0],.4/(2*r**3),atol=1e-13)
    batch=prepare_spectral(model,k,backend='numpy')
    phase=np.exp(1j*np.random.default_rng(9).normal(size=batch.energy.shape))
    u=batch.eigenvectors*phase[:,None,:]
    velocity=phase.conj()[:,None,:,None]*batch.velocity_matrices*phase[:,None,None,:]
    transformed=SpectralBatch(batch.k,batch.energy,u,velocity)
    rotated=prepare_geometry(transformed)
    for key,value in result.as_dict().items():close(value,rotated.as_dict()[key],atol=1e-13)


def test_point_orbital_frame_is_explicit_and_not_double_embedded():
    base=models.qi_wu_zhang()
    model=HR(base.vectors,base.hopping,orbitals=[[0,0],[.37,.11]])
    k=np.array([[.2,.3],[.4,.5]])
    point=prepare_geometry(model,k,frame='point-orbital')
    explicit=geometric_tensors(embed_orbitals(model),k,backend='numpy')
    assert point.frame=='point-orbital'
    for key,value in point.as_dict().items():close(value,explicit[key],atol=1e-13)
    no_double=prepare_geometry(embed_orbitals(model),k,frame='point-orbital')
    close(no_double.berry,point.berry,atol=1e-13)
    assert np.max(abs(point.berry-prepare_geometry(model,k).berry))>1e-3


def test_degenerate_policy_backend_parity_and_undefined_band_quantities():
    model=models.topological_insulator();k=[[.2,.3,.4],[.5,.7,.1]]
    batch=prepare_spectral(model,k)
    with pytest.raises(ValueError,match='degeneracy'):prepare_geometry(batch)
    a=prepare_geometry(batch,degeneracy='omit')
    b=prepare_geometry(batch,degeneracy='omit',backend='numpy')
    assert a.degenerate.all()
    for key,value in a.as_dict().items():close(value,b.as_dict()[key],atol=1e-13)


def test_single_point_shapes_chunking_and_validation():
    model=models.dirac(.4)
    single=prepare_geometry(model,[.2,.3]);batched=prepare_geometry(model,[[.2,.3]])
    assert single.energy.shape==batched.energy.shape==(1,2)
    assert response_from_geometry(single,0).shape==(1,)
    assert response_from_geometry(batched,0).shape==(1,1)
    k=np.random.default_rng(3).normal(size=(11,2))
    a=prepare_spectral(model,k,chunk_size=3);b=prepare_spectral(model,k,chunk_size=32)
    close(a.energy,b.energy,atol=1e-13);close(a.velocity_matrices,b.velocity_matrices,atol=1e-13)
    with pytest.raises(ValueError,match='already fixes'):prepare_geometry(a,k)
    with pytest.raises(ValueError,match='frame'):prepare_spectral(model,k,frame='unknown')
    with pytest.raises(ValueError):prepare_geometry(a,gap_tol=np.nan)
    with pytest.raises(ValueError):response_from_geometry(single,1j)
    with pytest.raises(ValueError):response_from_geometry(single,[],kind='berry')


def test_snapshot_array_kernels_are_callable_inside_user_njit():
    @njit
    def pipeline(k,r,h):
        matrices=kernels.fourier(k,r,h)
        gradient=kernels.fourier_gradient(k,r,h)
        nk,n,_=matrices.shape;e=np.empty((nk,n));u=np.empty_like(matrices)
        for q in range(nk):e[q],u[q]=np.linalg.eigh(matrices[q])
        v=kernels.band_velocities(u,gradient)
        return kernels.geometry_from_velocities(e,v,1e-8)[1]
    model=models.qi_wu_zhang();k=np.array([[.1,.3],[.4,.7]])
    result=pipeline(k,model.cartesian_vectors,model.hopping)
    close(result,prepare_geometry(model,k).berry,atol=1e-13)
