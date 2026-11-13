import numpy as np


def kmesh(shape,*,lattice=None):
    """Endpoint-free Cartesian reciprocal mesh; lattice vectors are rows."""
    shape=np.asarray(shape)
    if shape.ndim!=1 or np.any(shape<1) or not np.all(shape==np.rint(shape)):
        raise ValueError("positive integer grid sizes required")
    dim=len(shape)
    lattice=np.eye(dim) if lattice is None else np.asarray(lattice,float)
    reciprocal=2*np.pi*np.linalg.inv(lattice).T
    grid=np.stack(np.meshgrid(*(np.arange(int(n))/n for n in shape),indexing="ij"),axis=-1).reshape(-1,dim)
    return grid@reciprocal


def kpath(nodes,points_per_segment=40):
    """Cartesian linear path, cumulative distance, and node indices."""
    nodes=np.asarray(nodes,float)
    if nodes.ndim!=2 or len(nodes)<2 or not isinstance(points_per_segment,int) or points_per_segment<1:
        raise ValueError("two or more nodes and positive integer segment resolution required")
    k=np.concatenate([np.linspace(a,b,points_per_segment,endpoint=False) for a,b in zip(nodes[:-1],nodes[1:])]+[nodes[-1:]])
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(k,axis=0),axis=1))]
    return k,distance,np.arange(len(nodes))*points_per_segment
