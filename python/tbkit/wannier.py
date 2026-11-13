"""Wannier input metadata and final localization output; never evaluate input."""
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
from .structure import Structure

BOHR=0.529177210903
_FLOAT=r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?'


def _float(s):return float(s.replace('D','E').replace('d','e'))


@dataclass(frozen=True)
class WannierInput:
    parameters: dict
    blocks: dict
    structure: object


def read_win(path):
    parameters={};blocks={};active=None
    for raw in Path(path).read_text().splitlines():
        line=re.split(r'[!#]',raw,1)[0].strip()
        if not line:continue
        lower=line.lower()
        if lower.startswith('begin '):
            if active is not None:raise ValueError('nested Wannier blocks')
            active=lower.split()[1]
            if active in blocks:raise ValueError('duplicate Wannier block')
            blocks[active]=[];continue
        if lower.startswith('end '):
            if active!=lower.split()[1]:raise ValueError('unmatched Wannier block end')
            active=None;continue
        if active is not None:blocks[active].append(line);continue
        match=re.match(r'(\w+)\s*(?:=|:)\s*(.*)',line)
        if not match:raise ValueError(f'unrecognized Wannier parameter: {line}')
        key,value=match[1].lower(),match[2].strip()
        if key in parameters:raise ValueError('duplicate Wannier parameter')
        if value.lower() in ('true','t','.true.'):value=True
        elif value.lower() in ('false','f','.false.'):value=False
        elif re.fullmatch(r'[-+]?\d+',value):value=int(value)
        elif re.fullmatch(_FLOAT,value):value=_float(value)
        parameters[key]=value
    if active is not None:raise ValueError('unterminated Wannier block')
    structure=None
    if 'unit_cell_cart' in blocks:
        records=blocks['unit_cell_cart'].copy();factor=1.
        if records and records[0].lower() in ('bohr','ang','angstrom','angstroms'):
            factor=BOHR if records.pop(0).lower()=='bohr' else 1.
        lattice=np.array([[ _float(x) for x in row.split()] for row in records])*factor
        if lattice.shape!=(3,3):raise ValueError('invalid Wannier unit cell')
        for block in ('atoms_frac','atoms_cart'):
            if block in blocks:
                records=blocks[block].copy();atomfactor=1.
                if block=='atoms_cart' and records and records[0].lower() in ('bohr','ang','angstrom','angstroms'):
                    atomfactor=BOHR if records.pop(0).lower()=='bohr' else 1.
                fields=[row.split() for row in records]
                if any(len(row)!=4 for row in fields):raise ValueError('invalid atom record')
                positions=np.array([[ _float(x) for x in row[1:]] for row in fields])*atomfactor
                structure=Structure(lattice,positions,[row[0] for row in fields],cartesian=block=='atoms_cart')
                break
    return WannierInput(parameters,{k:tuple(v) for k,v in blocks.items()},structure)


@dataclass(frozen=True)
class WannierOutput:
    lattice: np.ndarray
    centers: np.ndarray # Angstrom
    spreads: np.ndarray # Angstrom^2


def read_wout(path):
    text=Path(path).read_text()
    lattice=[]
    for a in range(1,4):
        matches=re.findall(r'^\s*a_'+str(a)+r'\s+('+_FLOAT+r')\s+('+_FLOAT+r')\s+('+_FLOAT+r')',text,flags=re.M)
        if not matches:raise ValueError('Wannier real-space lattice missing')
        lattice.append(list(map(_float,matches[-1])))
    final=text.rsplit('Final State',1)
    if len(final)!=2:raise ValueError('no converged/final Wannier state found')
    pattern=r'WF centre and spread\s+(\d+)\s*\(\s*('+_FLOAT+r')\s*,\s*('+_FLOAT+r')\s*,\s*('+_FLOAT+r')\s*\)\s*('+_FLOAT+r')'
    records=re.findall(pattern,final[-1])
    if not records or [int(r[0]) for r in records]!=list(range(1,len(records)+1)):raise ValueError('invalid final Wannier centers')
    values=np.array([[ _float(x) for x in row[1:]] for row in records])
    return WannierOutput(np.array(lattice),values[:,:3],values[:,3])
