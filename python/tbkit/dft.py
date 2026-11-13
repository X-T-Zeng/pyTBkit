"""VASP band/projection/path data with explicit coordinates and spin layouts."""
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np


def _number(s):return float(s.replace('D','E').replace('d','e'))


def _next_nonempty(iterator):
    for line in iterator:
        if line.strip():return line.split()
    raise ValueError('truncated input file')


@dataclass(frozen=True)
class Eigenval:
    nelectrons: float
    kpoints: np.ndarray
    weights: np.ndarray
    energies: np.ndarray # nk, nb, spin channel (1 or 2)
    occupations: np.ndarray


def read_eigenval(path):
    with open(path) as f:
        for _ in range(5):
            if not f.readline():raise ValueError('truncated EIGENVAL header')
        fields=f.readline().split()
        if len(fields)!=3:raise ValueError('invalid EIGENVAL dimensions')
        nelectrons=_number(fields[0]);nk,nb=map(int,fields[1:])
        if min(nk,nb)<1:raise ValueError('nonpositive EIGENVAL dimensions')
        k=[];weights=[];energy=[];occupation=[];nspin=None
        for _ in range(nk):
            fields=_next_nonempty(f)
            if len(fields)!=4:raise ValueError('invalid k point record')
            k.append(list(map(_number,fields[:3])));weights.append(_number(fields[3]))
            bands=[];occ=[]
            for b in range(nb):
                fields=_next_nonempty(f)
                if int(fields[0])!=b+1 or len(fields) not in (3,5):raise ValueError('invalid band index/column count')
                spin=(len(fields)-1)//2
                if nspin is None:nspin=spin
                if spin!=nspin:raise ValueError('inconsistent spin columns')
                bands.append(list(map(_number,fields[1:1+spin])));occ.append(list(map(_number,fields[1+spin:])))
            energy.append(bands);occupation.append(occ)
    return Eigenval(nelectrons,np.array(k),np.array(weights),np.array(energy),np.array(occupation))


@dataclass(frozen=True)
class Kpoints:
    mode: str
    coordinates: str
    points: np.ndarray
    weights: np.ndarray
    labels: tuple
    count: int
    mesh: tuple=()
    shift: tuple=()

    def expanded(self):
        if self.mode=='line':return np.concatenate([np.linspace(a,b,self.count) for a,b in self.points.reshape(-1,2,3)])
        if self.mode=='explicit':return self.points.copy()
        axes=[]
        for n,shift in zip(self.mesh,self.shift):
            a=np.arange(n,dtype=float)
            axes.append((a+shift)/n if self.mode=='gamma' else (2*a+1-n)/(2*n)+shift/n)
        return np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)

    def cartesian(self,lattice,*,cartesian_scale=None):
        k=self.expanded()
        if self.coordinates=='reciprocal':return k@(2*np.pi*np.linalg.inv(lattice).T)
        if cartesian_scale is None:raise ValueError('VASP Cartesian KPOINTS require explicit 2pi/a scale')
        return k*cartesian_scale


def read_kpoints(path):
    lines=Path(path).read_text().splitlines()
    if len(lines)<4:raise ValueError('truncated KPOINTS')
    count=int(lines[1].split()[0]);mode=lines[2].strip().lower()
    if count==0:
        if mode[0] not in 'gm':raise ValueError('only Gamma/Monkhorst automatic meshes supported')
        mesh=tuple(map(int,lines[3].split()[:3]))
        shift=tuple(map(float,lines[4].split()[:3])) if len(lines)>4 and lines[4].strip() else (0.,0.,0.)
        if len(mesh)!=3 or min(mesh)<1 or len(shift)!=3:raise ValueError('invalid mesh or shift')
        return Kpoints('gamma' if mode[0]=='g' else 'monkhorst','reciprocal',np.empty((0,3)),np.empty(0),(),0,mesh,shift)
    if count<1:raise ValueError('positive point count required')
    line_mode=mode.startswith('l');coordinate=lines[3].strip().lower() if line_mode else mode
    if not coordinate or coordinate[0] not in 'rck':raise ValueError('invalid reciprocal/Cartesian mode')
    data=[];weights=[];labels=[]
    for line in lines[4 if line_mode else 3:]:
        if not line.strip():continue
        fields=line.replace('!',' ').replace('#',' ').split()
        if len(fields)<3:raise ValueError('invalid coordinate record')
        data.append([_number(x) for x in fields[:3]])
        if line_mode:labels.append(' '.join(fields[3:]));weights.append(0.)
        else:
            if len(fields)<4:raise ValueError('explicit k point weight required')
            weights.append(_number(fields[3]));labels.append(' '.join(fields[4:]))
            if len(data)==count:break
    if (line_mode and (len(data)%2 or not data)) or (not line_mode and len(data)!=count):raise ValueError('incomplete line segment/point list')
    return Kpoints('line' if line_mode else 'explicit','reciprocal' if coordinate[0]=='r' else 'cartesian',np.array(data),np.array(weights),tuple(labels),count)


@dataclass(frozen=True)
class Procar:
    kpoints: np.ndarray # spin,nk,3
    weights: np.ndarray # spin,nk
    energies: np.ndarray # spin,nk,nb
    occupations: np.ndarray
    orbitals: tuple # includes tot as last column
    projections: np.ndarray # spin,nk,nb,component(1 or 4),ion,orbital


def read_procar(path):
    lines=Path(path).read_text().splitlines();dims=None
    pattern=r'# of k-points:\s*(\d+)\s*# of bands:\s*(\d+)\s*# of ions:\s*(\d+)'
    for line in lines[:10]:
        match=re.search(pattern,line)
        if match:dims=tuple(map(int,match.groups()));break
    if dims is None or min(dims)<1:raise ValueError('PROCAR dimensions missing')
    nk,nb,ni=dims;channel=-1;ki=None;bi=None;records={};kdata={};labels=None;cursor=0
    float_pattern=r'([-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?)'
    kpattern=r'k-point\s+(\d+)\s*:\s*'+r'\s*'.join([float_pattern]*3)+r'\s+weight\s*=\s*'+float_pattern
    bpattern=r'band\s+(\d+)\s*# energy\s*'+float_pattern+r'\s*# occ\.\s*'+float_pattern
    while cursor<len(lines):
        line=lines[cursor].strip();cursor+=1
        if not line:continue
        km=re.match(kpattern,line)
        if km:
            ki=int(km[1])-1
            if ki==0:channel+=1
            if not 0<=ki<nk or channel<0:raise ValueError('invalid PROCAR k block')
            kdata[channel,ki]=(list(map(_number,km.groups()[1:4])),_number(km[5]));bi=None;continue
        bm=re.match(bpattern,line)
        if bm:
            bi=int(bm[1])-1
            if ki is None or not 0<=bi<nb:raise ValueError('invalid PROCAR band')
            key=(channel,ki,bi)
            if key in records:raise ValueError('duplicate PROCAR band')
            records[key]=[_number(bm[2]),_number(bm[3]),[]];continue
        if line.startswith('ion'):
            current=tuple(line.split()[1:])
            if labels is None:labels=current
            if current!=labels or labels[-1]!='tot':raise ValueError('inconsistent projection orbitals')
            continue
        fields=line.split()
        if bi is not None and fields[0].isdigit():
            if labels is None:raise ValueError('projection header missing')
            def projection_row(raw):
                parts=raw.strip().split(maxsplit=1)
                if len(parts)!=2:raise ValueError('invalid projection row')
                return [parts[0]]+re.findall(float_pattern,parts[1])
            table=[projection_row(line)]
            if cursor+ni>len(lines):raise ValueError('truncated projection table')
            table.extend(projection_row(row) for row in lines[cursor:cursor+ni]);cursor+=ni
            if [row[0] for row in table[:-1]]!=[str(i+1) for i in range(ni)] or table[-1][0]!='tot':raise ValueError('invalid ion projection table')
            if any(len(row)!=len(labels)+1 for row in table):raise ValueError('unsupported PROCAR projection/phase columns')
            records[channel,ki,bi][2].append([[ _number(x) for x in row[1:]] for row in table[:-1]])
    spins=channel+1
    if not labels or len(records)!=spins*nk*nb or len(kdata)!=spins*nk:raise ValueError('incomplete PROCAR data')
    nc=len(next(iter(records.values()))[2])
    if nc not in (1,4) or any(len(record[2])!=nc for record in records.values()):raise ValueError('only charge or charge+xyz spin projection tables supported')
    energy=np.empty((spins,nk,nb));occ=np.empty_like(energy);projections=np.empty((spins,nk,nb,nc,ni,len(labels)))
    kp=np.empty((spins,nk,3));weight=np.empty((spins,nk))
    for key,(e,o,p) in records.items():energy[key]=e;occ[key]=o;projections[key]=p
    for key,(k,w) in kdata.items():kp[key]=k;weight[key]=w
    return Procar(kp,weight,energy,occ,labels,projections)
