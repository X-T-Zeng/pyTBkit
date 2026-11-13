"""Integer supercells, slabs, and explicit basis operations."""
from collections import deque
import numpy as np
from .hr import HR


def _cosets(matrix):
    m=np.asarray(matrix)
    if m.ndim!=2 or m.shape[0]!=m.shape[1] or not np.allclose(m,np.rint(m)):
        raise ValueError("square integer supercell matrix required")
    m=np.rint(m).astype(int); det=int(round(np.linalg.det(m)))
    if det==0: raise ValueError("singular supercell transformation")
    adj=np.rint(np.linalg.inv(m)*det).astype(int); size=abs(det)
    def key(c): return tuple((np.asarray(c)@adj)%size)
    inv=np.linalg.inv(m); found={}; queue=deque([np.zeros(len(m),int)])
    while queue:
        c=queue.popleft(); residue=key(c)
        if residue in found: continue
        c=c-np.floor(c@inv+1e-10).astype(int)@m
        found[residue]=c
        if len(found)==size: break
        for step in np.eye(len(m),dtype=int): queue.append(c+step)
    return m,list(found.values()),key


def supercell(model,matrix):
    """General nonsingular integer supercell, exact coset decomposition."""
    m,cells,key=_cosets(matrix)
    if len(m)!=model.dim: raise ValueError("supercell dimension mismatch")
    lookup={key(c):i for i,c in enumerate(cells)}
    inverse=np.linalg.inv(m); n=len(cells)*model.norb; terms={}
    for ci,c in enumerate(cells):
        for r,h in zip(model.vectors,model.hopping):
            dest=c+r; cj=lookup[key(dest)]
            shift=tuple(np.rint((dest-cells[cj])@inverse).astype(int))
            if shift not in terms: terms[shift]=np.zeros((n,n),complex)
            terms[shift][ci*model.norb:(ci+1)*model.norb,cj*model.norb:(cj+1)*model.norb]+=h
    orbitals=np.concatenate([(model.orbitals+c)@inverse for c in cells])
    return HR(list(terms),list(terms.values()),lattice=m@model.lattice,orbitals=orbitals)


def slab(model,layers,axis=0):
    """HR slab with one open axis; retain dimension and only k-independent R_axis=0.

    Other directions remain periodic. The open-axis lattice vector is enlarged
    and metadata centers scaled; this avoids silently redefining coordinates.
    """
    if not 0<=axis<model.dim or not isinstance(layers,int) or layers<1:
        raise ValueError("valid axis and positive layer count required")
    repeats=np.ones(model.dim,int); repeats[axis]=layers
    full=model.supercell(repeats)
    keep=full.vectors[:,axis]==0
    return HR(full.vectors[keep],full.hopping[keep],lattice=full.lattice,orbitals=full.orbitals)


def select_orbitals(model,indices):
    indices=np.asarray(indices)
    if indices.ndim!=1 or len(indices)==0 or not np.issubdtype(indices.dtype,np.integer) or len(set(indices))!=len(indices) or np.any(indices<0) or np.any(indices>=model.norb):
        raise ValueError("unique valid orbital indices required")
    return HR(model.vectors,model.hopping[:,indices][:,:,indices],lattice=model.lattice,orbitals=model.orbitals[indices])


def direct_sum(*models):
    from scipy.linalg import block_diag
    if not models: raise ValueError("one or more models required")
    first=models[0]
    if any(m.dim!=first.dim or not np.allclose(m.lattice,first.lattice) for m in models):
        raise ValueError("direct sum models need the same lattice")
    keys=sorted(set(tuple(r) for m in models for r in m.vectors))
    mappings=[{tuple(r):h for r,h in zip(m.vectors,m.hopping)} for m in models]
    blocks=[block_diag(*(mapping.get(r,np.zeros((m.norb,m.norb))) for m,mapping in zip(models,mappings))) for r in keys]
    return HR(keys,blocks,lattice=first.lattice,orbitals=np.concatenate([m.orbitals for m in models]))


def peierls_finite(model,cells,vector_potential,*,charge_over_hbar=1.,periodic=None):
    """Finite CSR with midpoint Peierls phase exp(i q/hbar int_i^j A.dl).

    A is a callable of Cartesian position, assumed affine along a bond. Use
    open boundaries for a uniform field; periodic magnetic cells/gauge patches
    require separate flux quantization and are deliberately rejected here.
    """
    import itertools
    from scipy import sparse
    if periodic is not None and np.any(periodic): raise ValueError("Peierls routine currently requires open boundaries")
    cells=np.asarray(cells)
    if cells.shape!=(model.dim,) or np.any(cells<1) or not np.all(cells==np.rint(cells)):
        raise ValueError("positive integer cell counts required")
    cell_list=list(itertools.product(*(range(int(n)) for n in cells)))
    lookup={c:i for i,c in enumerate(cell_list)}; rows=[]; cols=[]; values=[]
    for ci,c in enumerate(cell_list):
        for r,h in zip(model.vectors,model.hopping):
            dest=tuple(np.asarray(c)+r)
            if dest not in lookup: continue
            cj=lookup[dest]
            ii,jj=np.nonzero(h)
            for i,j in zip(ii,jj):
                left=(np.asarray(c)+model.orbitals[i])@model.lattice
                right=(np.asarray(dest)+model.orbitals[j])@model.lattice
                a=np.asarray(vector_potential((left+right)/2),float)
                if a.shape!=(model.dim,) or not np.isfinite(a).all(): raise ValueError("invalid vector potential")
                phase=np.exp(1j*charge_over_hbar*np.dot(a,right-left))
                rows.append(ci*model.norb+i); cols.append(cj*model.norb+j); values.append(h[i,j]*phase)
    n=len(cell_list)*model.norb
    return sparse.coo_matrix((values,(rows,cols)),shape=(n,n)).tocsr()


def sparse_supercell(model,matrix):
    """Integer-cell transformation assembled directly as sparse COO blocks."""
    from scipy import sparse
    from .sparse import SparseHR
    matrix=np.asarray(matrix)
    if matrix.ndim==1:
        if matrix.shape!=(model.dim,) or np.any(matrix<1) or not np.all(matrix==np.rint(matrix)):raise ValueError("positive integer repeats required")
        matrix=np.diag(matrix)
    m,cells,key=_cosets(matrix)
    if len(m)!=model.dim:raise ValueError("supercell dimension mismatch")
    lookup={key(c):i for i,c in enumerate(cells)};inverse=np.linalg.inv(m);terms={}
    n=len(cells)*model.norb
    for ci,c in enumerate(cells):
        for r,h in zip(model.vectors,model.hopping):
            dest=c+r;cj=lookup[key(dest)];shift=tuple(np.rint((dest-cells[cj])@inverse).astype(int))
            row,col,value=terms.setdefault(shift,([],[],[]));coo=h.tocoo()
            row.extend(coo.row+ci*model.norb);col.extend(coo.col+cj*model.norb);value.extend(coo.data)
    blocks={r:sparse.coo_matrix((v,(i,j)),shape=(n,n)).tocsr() for r,(i,j,v) in terms.items()}
    orbitals=np.concatenate([(model.orbitals+c)@inverse for c in cells])
    return SparseHR(blocks,lattice=m@model.lattice,orbitals=orbitals)


def sparse_cut(model,cells,*,periodic=None):
    """Finite/periodic CSR assembly without ever densifying hopping blocks."""
    import itertools
    from scipy import sparse
    cells=np.asarray(cells)
    if cells.shape!=(model.dim,) or np.any(cells<1) or not np.all(cells==np.rint(cells)):raise ValueError("positive integer cell counts required")
    cells=cells.astype(int);periodic=np.zeros(model.dim,bool) if periodic is None else np.asarray(periodic,bool)
    if periodic.shape!=(model.dim,):raise ValueError("periodic flags dimension mismatch")
    positions=list(itertools.product(*(range(n) for n in cells)));index={c:i for i,c in enumerate(positions)}
    rows=[];cols=[];values=[]
    for ci,c in enumerate(positions):
        for r,h in zip(model.vectors,model.hopping):
            dest=np.asarray(c)+r
            if np.any(((dest<0)|(dest>=cells))&~periodic):continue
            cj=index[tuple(dest%cells)];coo=h.tocoo()
            rows.extend(coo.row+ci*model.norb);cols.extend(coo.col+cj*model.norb);values.extend(coo.data)
    n=len(positions)*model.norb
    return sparse.coo_matrix((values,(rows,cols)),shape=(n,n)).tocsr()
