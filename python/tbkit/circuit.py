"""Linear periodic RLC circuit Laplacians, finite solves and SPICE export.

This is the native numerical part of the Hckt/Subckt family. Commercial
HSPICE control decks and proprietary file parsers are not emulated.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from .base import points


@dataclass(frozen=True)
class Component:
    kind: str
    value: float

    def __post_init__(self):
        if self.kind not in ('R','L','C') or not np.isfinite(self.value) or self.value<=0:
            raise ValueError("positive finite R/L/C component required")

    def admittance(self,frequency):
        if not np.isfinite(frequency) or frequency<0:raise ValueError("nonnegative finite frequency in Hz required")
        omega=2*np.pi*frequency
        if self.kind=='R':return 1/self.value
        if self.kind=='C':return 1j*omega*self.value
        if omega==0:raise ValueError("ideal inductor at DC requires a constrained short-circuit solve")
        return 1/(1j*omega*self.value)


class Subckt:
    def __init__(self,nnodes,components):
        """components: (node_i,node_j,Component), node -1 denotes ground."""
        if not isinstance(nnodes,int) or nnodes<1:raise ValueError("positive node count required")
        self.nnodes=nnodes;self.components=tuple(components)
        for i,j,c in self.components:
            if not -1<=i<nnodes or not -1<=j<nnodes or i==j or not isinstance(c,Component):raise ValueError("invalid component terminals")

    def laplacian(self,frequency):
        y=np.zeros((self.nnodes,self.nnodes),complex)
        for i,j,c in self.components:
            value=c.admittance(frequency)
            if i>=0:y[i,i]+=value
            if j>=0:y[j,j]+=value
            if i>=0 and j>=0:y[i,j]-=value;y[j,i]-=value
        return y

    def voltages(self,frequency,injection):
        injection=np.asarray(injection,complex)
        if injection.shape!=(self.nnodes,):raise ValueError("one injected current per node required")
        return solve(self.laplacian(frequency),injection)

    def to_spice(self,path,*,title='TBkit RLC circuit'):
        lines=[title]
        for index,(i,j,c) in enumerate(self.components):
            lines.append(f'{c.kind}{index+1} {i+1} {j+1} {c.value:.17g}')
        Path(path).write_text('\n'.join(lines+['.end'])+'\n')


class Hckt:
    def __init__(self,nnodes,dim,bonds,*,ground=(),lattice=None):
        """bonds: (i,j,integer R,Component), each undirected physical bond ONCE.

        ground: (node,Component). Includes no automatic duplicate bond pruning.
        """
        if not isinstance(nnodes,int) or nnodes<1 or not isinstance(dim,int) or not 1<=dim<=3:raise ValueError("invalid circuit dimensions")
        self.nnodes,self.dim=nnodes,dim;self.bonds=tuple(bonds);self.ground=tuple(ground)
        self.lattice=np.eye(dim) if lattice is None else np.asarray(lattice,float)
        if self.lattice.shape!=(dim,dim) or not np.isfinite(self.lattice).all() or abs(np.linalg.det(self.lattice))<1e-12:raise ValueError("invertible lattice required")
        for i,j,r,c in self.bonds:
            if not 0<=i<nnodes or not 0<=j<nnodes or np.shape(r)!=(dim,) or not np.all(np.asarray(r)==np.rint(r)) or not isinstance(c,Component) or (i==j and not np.any(r)):
                raise ValueError("invalid circuit bond")
        for i,c in self.ground:
            if not 0<=i<nnodes or not isinstance(c,Component):raise ValueError("invalid grounding component")

    def laplacian(self,k,frequency):
        k,single=points(k,self.dim);out=np.zeros((len(k),self.nnodes,self.nnodes),complex)
        for i,j,r,c in self.bonds:
            y=c.admittance(frequency);phase=np.exp(1j*(k@(np.asarray(r)@self.lattice)))
            out[:,i,i]+=y;out[:,j,j]+=y
            out[:,i,j]-=y*phase;out[:,j,i]-=y*phase.conj()
        for i,c in self.ground:out[:,i,i]+=c.admittance(frequency)
        return out[0] if single else out
