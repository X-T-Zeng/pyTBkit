"""Crystal I/O and periodic neighbor lists. Lattice vectors are rows."""
from dataclasses import dataclass
from pathlib import Path
import itertools
import numpy as np
from scipy.spatial import cKDTree
from .hr import immutable
from .lattice import checked_lattice,inverse_lattice,log_cell_volume,finite_measure


@dataclass(frozen=True)
class Neighbors:
    i: np.ndarray
    j: np.ndarray
    cells: np.ndarray
    displacement: np.ndarray
    distance: np.ndarray


class Structure:
    def __init__(self,lattice,positions,species,*,cartesian=False,selective=None):
        lattice=checked_lattice(lattice,3)
        if np.iscomplexobj(positions):raise ValueError('positions must be real')
        positions=np.asarray(positions,float)
        if positions.ndim!=2 or positions.shape[1]!=3 or len(species)!=len(positions):
            raise ValueError("positions (natoms,3), one species per atom required")
        self.lattice=immutable(lattice,float)
        self.positions=immutable(positions@inverse_lattice(lattice) if cartesian else positions,float)
        self.species=tuple(map(str,species))
        if selective is not None and np.shape(selective)!=(len(positions),3):
            raise ValueError("selective flags shape must match positions")
        self.selective=None if selective is None else immutable(selective,bool)

    @property
    def cartesian(self): return self.positions@self.lattice

    @property
    def volume(self): return finite_measure(self.log_volume)

    @property
    def log_volume(self): return log_cell_volume(self.lattice)

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

    def neighbors(self,cutoff,*,periodic=(True,True,True),tol=None,max_images=2_000_000):
        """Directed i -> j+R bonds with tol < distance <= cutoff+tol.

        Reciprocal bounds remain valid for skew cells. Positions are not
        silently wrapped, so reported R refers to the original coordinates.
        Default tol is 1e-10 times cutoff; explicit tol has length units.
        Search geometry uses cutoff units, avoiding norm over/underflow when
        length units change. max_images bounds replicated atoms before allocation.
        """
        if (isinstance(cutoff,(bool,np.bool_)) or np.ndim(cutoff)!=0 or np.iscomplexobj(cutoff)
                or not np.isfinite(cutoff) or cutoff<=0):
            raise ValueError("finite positive cutoff required")
        cutoff=float(cutoff)
        if tol is not None and (np.ndim(tol)!=0 or np.iscomplexobj(tol) or not np.isfinite(tol) or tol<0):
            raise ValueError('finite nonnegative length tolerance required')
        with np.errstate(over='ignore'):
            tolerance=1e-10 if tol is None else float(tol)/cutoff
        radius=1.+tolerance
        if not np.isfinite(radius):raise ValueError('neighbor tolerance not representable in cutoff units')
        if isinstance(max_images,(bool,np.bool_)) or not isinstance(max_images,(int,np.integer)) or max_images<1:
            raise ValueError('positive integer max_images required')
        periodic=np.asarray(periodic,bool)
        if periodic.shape!=(3,): raise ValueError("three periodic flags required")
        if not len(self.positions):
            return Neighbors(np.empty(0,int),np.empty(0,int),np.empty((0,3),int),np.empty((0,3)),np.empty(0))
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            cell=self.lattice/cutoff
            # A common origin reduces cancellation without changing image labels.
            coordinates=(self.positions-self.positions[0])@cell
        if not np.isfinite(cell).all() or not np.any(cell) or not np.isfinite(coordinates).all():
            raise ValueError('neighbor geometry not representable in cutoff units')
        bounds=np.zeros(3,dtype=np.int64)
        if np.any(periodic):
            inverse=inverse_lattice(cell)
            span=np.ptp(self.positions,axis=0)
            limit=np.ceil(radius*np.hypot.reduce(inverse,axis=0)+span)
            # Python integer arithmetic prevents an overflowing image count.
            count=len(self.positions)
            for axis in range(3):
                if periodic[axis]:
                    if (not np.isfinite(limit[axis]) or limit[axis]>max_images
                            or limit[axis]>np.iinfo(np.int64).max//2-1):
                        raise MemoryError('neighbor replicated atoms exceed max_images')
                    bounds[axis]=int(limit[axis]);count*=2*int(bounds[axis])+1
            if count>max_images:raise MemoryError('neighbor replicated atoms exceed max_images')
        elif len(self.positions)>max_images:
            raise MemoryError('neighbor replicated atoms exceed max_images')
        cells=np.array(list(itertools.product(*(range(-b,b+1) if periodic[a] else [0] for a,b in enumerate(bounds)))))
        replicas=(coordinates[None,:,:]+(cells@cell)[:,None,:]).reshape(-1,3)
        if not np.isfinite(replicas).all():raise ValueError('nonfinite neighbor replicas')
        tree=cKDTree(replicas); ii=[]; jj=[]; rr=[]; dd=[]; distances=[]
        for i,indices in enumerate(tree.query_ball_point(coordinates,np.nextafter(radius,np.inf))):
            for index in sorted(indices):
                ci,j=divmod(index,len(self.positions))
                delta=(self.positions[j]-self.positions[i]+cells[ci])@cell
                distance=float(np.hypot.reduce(delta))
                if tolerance<distance<=radius:
                    ii.append(i); jj.append(j); rr.append(cells[ci]); dd.append(delta*cutoff)
                    distances.append(distance*cutoff)
        delta=np.asarray(dd,float).reshape(-1,3)
        distances=np.asarray(distances,float)
        if not np.isfinite(delta).all() or not np.isfinite(distances).all() or np.any(distances==0.):
            raise ValueError('neighbor displacement/distance not representable')
        return Neighbors(np.asarray(ii,int),np.asarray(jj,int),np.asarray(rr,int).reshape(-1,3),delta,distances)
