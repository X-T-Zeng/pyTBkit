"""Wannier90 hr.dat (including degeneracies) and versioned HDF5."""
import numpy as np
import h5py
from .hr import HR


def read_wannier90(path,*,lattice=None,orbitals=None):
    with open(path) as f:
        next(f)
        n=int(next(f)); nr=int(next(f))
        if n<1 or nr<1: raise ValueError("invalid Wannier90 header")
        degeneracy=[]
        while len(degeneracy)<nr: degeneracy.extend(map(int,next(f).split()))
        if len(degeneracy)!=nr or min(degeneracy)<1: raise ValueError("invalid degeneracy list")
        vectors=[]; hopping=[]; seen=set()
        for rindex in range(nr):
            block=np.zeros((n,n),complex); entries=set(); vector=None
            for _ in range(n*n):
                fields=next(f).split()
                if len(fields)!=7: raise ValueError("malformed Wannier90 record")
                r=tuple(map(int,fields[:3])); i,j=map(int,fields[3:5])
                if vector is not None and vector!=r: raise ValueError("inconsistent R block")
                if not 1<=i<=n or not 1<=j<=n or (i,j) in entries: raise ValueError("invalid/duplicate matrix index")
                vector=r; entries.add((i,j))
                block[i-1,j-1]=complex(float(fields[5].replace('D','E')),float(fields[6].replace('D','E')))/degeneracy[rindex]
            if vector in seen: raise ValueError("duplicate Wannier90 R block")
            seen.add(vector); vectors.append(vector); hopping.append(block)
        if any(line.strip() for line in f): raise ValueError("unexpected records after hr.dat")
    return HR(vectors,hopping,lattice=lattice,orbitals=orbitals)


def write_wannier90(model,path):
    if model.dim!=3: raise ValueError("Wannier90 export requires a 3D model (pad lower dimensions explicitly)")
    with open(path,'w') as f:
        f.write(f'TBkit Python normalized hopping; degeneracies absorbed\n{model.norb}\n{len(model.vectors)}\n')
        for start in range(0,len(model.vectors),15): f.write(' '.join(['1']*min(15,len(model.vectors)-start))+'\n')
        for r,h in zip(model.vectors,model.hopping):
            for j in range(model.norb):
                for i in range(model.norb):
                    f.write(f'{r[0]} {r[1]} {r[2]} {i+1} {j+1} {h[i,j].real:.17e} {h[i,j].imag:.17e}\n')


def save_hdf5(model,path):
    from .sparse import SparseHR
    from .symbolic import HK,Htrig
    with h5py.File(path,'w') as f:
        if isinstance(model,SparseHR):
            f.attrs['schema']='tbkit.sparsehr.v1'
            for name in ('vectors','lattice','orbitals'):f.create_dataset(name,data=getattr(model,name))
            group=f.create_group('hopping')
            for i,h in enumerate(model.hopping):
                g=group.create_group(str(i));g.attrs['shape']=h.shape
                for key in ('data','indices','indptr'):g.create_dataset(key,data=getattr(h,key),compression='gzip')
        elif isinstance(model,HR):
            f.attrs['schema']='tbkit.hr.v1'
            for name in ('vectors','hopping','lattice','orbitals'):f.create_dataset(name,data=getattr(model,name),compression='gzip',shuffle=True)
        elif isinstance(model,HK):
            f.attrs['schema']='tbkit.hk.v1'
            f.create_dataset('powers',data=model.powers);f.create_dataset('coefficients',data=model.coefficients)
        elif isinstance(model,Htrig):
            f.attrs['schema']='tbkit.htrig.v1'
            for name in ('frequencies','cosine','sine','lattice','orbitals'):f.create_dataset(name,data=getattr(model,name))
        else:raise TypeError('materialize transformed/composite model before saving')


def load_hdf5(path):
    from .sparse import SparseHR
    from .symbolic import HK,Htrig
    from scipy import sparse
    with h5py.File(path) as f:
        schema=f.attrs.get('schema')
        if schema=='tbkit.hr.v1':return HR(f['vectors'][:],f['hopping'][:],lattice=f['lattice'][:],orbitals=f['orbitals'][:])
        if schema=='tbkit.sparsehr.v1':
            terms={}
            for i,r in enumerate(f['vectors'][:]):
                group=f['hopping'][str(i)]
                terms[tuple(r)]=sparse.csr_matrix((group['data'][:],group['indices'][:],group['indptr'][:]),shape=tuple(group.attrs['shape']))
            return SparseHR(terms,lattice=f['lattice'][:],orbitals=f['orbitals'][:])
        if schema=='tbkit.hk.v1':return HK({tuple(p):c for p,c in zip(f['powers'][:],f['coefficients'][:])})
        if schema=='tbkit.htrig.v1':return Htrig(f['frequencies'][:],f['cosine'][:],f['sine'][:],lattice=f['lattice'][:] if 'lattice' in f else None,orbitals=f['orbitals'][:] if 'orbitals' in f else None)
        raise ValueError('unsupported HDF5 schema')


def apply_wsvec(model,path):
    """Expand Wannier90 wsvec.dat orbital-pair displacement corrections.

    Input HR has hr.dat degeneracies already absorbed. For each (R,i,j),
    replace H_ij(R) by equally weighted H_ij(R+T) over its wsvec list.
    All nonzero entries must be represented; incomplete files raise.
    """
    if model.dim!=3: raise ValueError("wsvec requires a 3D HR model")
    with open(path) as f:
        lines=[line.split() for line in f if line.strip()]
    if not lines: raise ValueError("empty wsvec file")
    # Standard header is free text. A headerless numeric file is also accepted.
    start=0
    try: [int(x) for x in lines[0]]
    except ValueError: start=1
    cursor=start; correction={}
    while cursor<len(lines):
        fields=lines[cursor];cursor+=1
        if len(fields)!=5: raise ValueError("wsvec expects R1 R2 R3 i j")
        vals=list(map(int,fields));r=tuple(vals[:3]);i,j=vals[3]-1,vals[4]-1
        if not 0<=i<model.norb or not 0<=j<model.norb: raise ValueError("wsvec orbital index outside model")
        if cursor>=len(lines) or len(lines[cursor])!=1: raise ValueError("wsvec vector count missing")
        count=int(lines[cursor][0]);cursor+=1
        if count<1 or cursor+count>len(lines): raise ValueError("invalid wsvec count")
        vectors=np.asarray([list(map(int,row)) for row in lines[cursor:cursor+count]])
        if vectors.shape!=(count,3): raise ValueError("wsvec vectors must have three integers")
        cursor+=count;key=(r,i,j)
        if key in correction: raise ValueError("duplicate wsvec entry")
        correction[key]=vectors
    terms={}; original={tuple(r):h for r,h in zip(model.vectors,model.hopping)}
    for key in correction:
        if key[0] not in original: raise ValueError("wsvec contains an R absent from hr.dat")
    for r,h in original.items():
        for i,j in zip(*np.nonzero(h)):
            key=(r,i,j)
            if key not in correction: raise ValueError(f'missing wsvec correction for {key}')
            shifts=correction[key]
            for shift in shifts:
                target=tuple(np.asarray(r)+shift)
                if target not in terms: terms[target]=np.zeros_like(h)
                terms[target][i,j]+=h[i,j]/len(shifts)
    if not terms: terms={(0,0,0):np.zeros((model.norb,model.norb))}
    return HR(list(terms),list(terms.values()),lattice=model.lattice,orbitals=model.orbitals)
