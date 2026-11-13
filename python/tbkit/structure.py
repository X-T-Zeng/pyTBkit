"""Crystal I/O and periodic neighbor lists. Lattice vectors are rows."""
from dataclasses import dataclass
from pathlib import Path
import itertools
import numpy as np
from scipy.spatial import cKDTree
from .hr import immutable


@dataclass(frozen=True)
class Neighbors:
    i: np.ndarray
    j: np.ndarray
    cells: np.ndarray
    displacement: np.ndarray
    distance: np.ndarray


class Structure:
    def __init__(self,lattice,positions,species,*,cartesian=False,selective=None):
        lattice=np.asarray(lattice,float); positions=np.asarray(positions,float)
        if lattice.shape!=(3,3) or abs(np.linalg.det(lattice))<1e-12:
            raise ValueError("invertible 3D lattice required")
        if positions.ndim!=2 or positions.shape[1]!=3 or len(species)!=len(positions):
            raise ValueError("positions (natoms,3), one species per atom required")
        self.lattice=immutable(lattice,float)
        self.positions=immutable(positions@np.linalg.inv(lattice) if cartesian else positions,float)
        self.species=tuple(map(str,species))
        if selective is not None and np.shape(selective)!=(len(positions),3):
            raise ValueError("selective flags shape must match positions")
        self.selective=None if selective is None else immutable(selective,bool)

    @property
    def cartesian(self): return self.positions@self.lattice

    @property
    def volume(self): return abs(np.linalg.det(self.lattice))

    @classmethod
    def from_poscar(cls,path):
        lines=Path(path).read_text().splitlines()
        if len(lines)<8: raise ValueError("truncated POSCAR")
        scale=np.array(list(map(float,lines[1].split())))
        lattice=np.array([list(map(float,line.split()[:3])) for line in lines[2:5]])
        if len(scale)==1:
            if scale[0]==0: raise ValueError("zero POSCAR scale")
            factor=(-scale[0]/abs(np.linalg.det(lattice)))**(1/3) if scale[0]<0 else scale[0]
        elif len(scale)==3 and np.all(scale>0): factor=scale
        else: raise ValueError("invalid POSCAR scale")
        lattice=lattice*factor
        cursor=5
        try:
            counts=list(map(int,lines[cursor].split()))
            names=[f'X{i+1}' for i in range(len(counts))]
        except ValueError:
            names=lines[cursor].split(); cursor+=1
            counts=list(map(int,lines[cursor].split()))
        if len(names)!=len(counts) or min(counts,default=-1)<0: raise ValueError("invalid atom counts")
        cursor+=1
        selective=lines[cursor].strip().lower().startswith('s')
        if selective: cursor+=1
        mode=lines[cursor].strip().lower(); cursor+=1
        if not mode or mode[0] not in 'dck': raise ValueError("Direct or Cartesian coordinates required")
        n=sum(counts); records=[line.split() for line in lines[cursor:cursor+n]]
        if len(records)!=n or any(len(line)<(6 if selective else 3) for line in records):
            raise ValueError("truncated atom positions")
        positions=np.array([[float(x) for x in row[:3]] for row in records]).reshape(n,3)
        flags=None
        if selective:
            if any(x.upper() not in ('T','F') for row in records for x in row[3:6]): raise ValueError("invalid selective flag")
            flags=np.array([[x.upper()=='T' for x in row[3:6]] for row in records])
        if mode[0] in 'ck': positions=positions*factor
        species=[name for name,count in zip(names,counts) for _ in range(count)]
        return cls(lattice,positions,species,cartesian=mode[0] in 'ck',selective=flags)

    def to_poscar(self,path):
        names=list(dict.fromkeys(self.species))
        order=[i for name in names for i,s in enumerate(self.species) if s==name]
        rows=['TBkit Python','1.0']+[' '.join(f'{x:.16g}' for x in row) for row in self.lattice]
        rows+=[' '.join(names),' '.join(str(self.species.count(n)) for n in names)]
        if self.selective is not None: rows+=['Selective dynamics']
        rows+=['Direct']
        for i in order:
            row=' '.join(f'{x:.16g}' for x in self.positions[i])
            if self.selective is not None: row+=' '+' '.join('T' if v else 'F' for v in self.selective[i])
            rows.append(row)
        Path(path).write_text('\n'.join(rows)+'\n')

    def neighbors(self,cutoff,*,periodic=(True,True,True),tol=1e-10):
        """All directed i -> j+R bonds within cutoff, excluding self at R=0.

        Reciprocal bounds remain valid for skew cells. Positions are not
        silently wrapped, so reported R refers to the original coordinates.
        """
        if cutoff<=0 or tol<0: raise ValueError("positive cutoff and nonnegative tolerance required")
        periodic=np.asarray(periodic,bool)
        if periodic.shape!=(3,): raise ValueError("three periodic flags required")
        if not len(self.positions):
            return Neighbors(np.empty(0,int),np.empty(0,int),np.empty((0,3),int),np.empty((0,3)),np.empty(0))
        span=np.ptp(self.positions,axis=0)
        bounds=np.ceil(cutoff*np.linalg.norm(np.linalg.inv(self.lattice),axis=0)+span).astype(int)
        cells=np.array(list(itertools.product(*(range(-b,b+1) if periodic[a] else [0] for a,b in enumerate(bounds)))))
        replicas=(self.cartesian[None,:,:]+(cells@self.lattice)[:,None,:]).reshape(-1,3)
        tree=cKDTree(replicas); ii=[]; jj=[]; rr=[]; dd=[]
        for i,indices in enumerate(tree.query_ball_point(self.cartesian,cutoff+tol)):
            for index in sorted(indices):
                ci,j=divmod(index,len(self.positions)); delta=replicas[index]-self.cartesian[i]
                if np.linalg.norm(delta)>tol:
                    ii.append(i); jj.append(j); rr.append(cells[ci]); dd.append(delta)
        delta=np.asarray(dd,float).reshape(-1,3)
        return Neighbors(np.asarray(ii,int),np.asarray(jj,int),np.asarray(rr,int).reshape(-1,3),delta,np.linalg.norm(delta,axis=1))
