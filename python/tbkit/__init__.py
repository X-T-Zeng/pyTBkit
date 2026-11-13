"""TBkit Python: explicit models, Numba kernels, and shared research tools."""
from .base import TBkit
from .hr import HR
from .symbolic import HK, Htrig
from .operators import Oper
from .geometry import kmesh, kpath
from .topology import berry_curvature, chern_number, wilson_loop, fermi
from .transport import surface_green, transmission
from .io import read_wannier90, write_wannier90, load_hdf5
from .fitting import fit_bands
from . import models
from . import backends as kernels

__version__ = "0.2.1"
__all__ = ["TBkit","HR","HK","Htrig","Oper","kmesh","kpath","berry_curvature",
           "chern_number","wilson_loop","fermi","surface_green","transmission",
           "read_wannier90","write_wannier90","load_hdf5","fit_bands","models","kernels"]

from .structure import Structure
from .sparse import SparseHR
from .nonorthogonal import NonOrthogonal
from .construction import supercell,slab,select_orbitals,direct_sum,peierls_finite
from .orbitals import angular_momentum,spin_rotation,spin_orbit_coupling,orbital_rotation
from .slater_koster import slater_koster,from_structure
from .symmetry import finite_group,symmetrize,invariant_kp
from .responses import geometric_tensors,response_kernel,integrate_bz,fermi_derivative
from .topology import mirror_chern,wilson_flow,nested_polarization,inversion_indicators
from .continuum import landau_hamiltonian,plane_wave_hamiltonian
from .spin import SpinModel,heisenberg_quantum
from .circuit import Component,Subckt,Hckt
__all__ += ["Structure","SparseHR","NonOrthogonal","supercell","slab","select_orbitals","direct_sum",
            "peierls_finite","angular_momentum","spin_rotation","spin_orbit_coupling","orbital_rotation",
            "slater_koster","from_structure","finite_group","symmetrize","invariant_kp","geometric_tensors",
            "response_kernel","integrate_bz","fermi_derivative","mirror_chern","wilson_flow",
            "nested_polarization","inversion_indicators","landau_hamiltonian","plane_wave_hamiltonian",
            "SpinModel","heisenberg_quantum","Component","Subckt","Hckt"]

from .dft import read_eigenval,read_kpoints,read_procar
from .wannier import read_win,read_wout
from .transport import PrincipalLayer,principal_layer,chain_transmission
__all__ += ["read_eigenval","read_kpoints","read_procar","read_win","read_wout","PrincipalLayer","principal_layer","chain_transmission"]

from .quantum import Qnum,BasisFunc,spherical_harmonic,coupled_basis
from .crystallography import spacegroup,crystal_operations,standardize,sewing_matrix,crystal_residual,symmetrize_hr
__all__ += ["Qnum","BasisFunc","spherical_harmonic","coupled_basis","spacegroup","crystal_operations",
            "standardize","sewing_matrix","crystal_residual","symmetrize_hr"]

from .topology import z2_wilson
__all__ += ["z2_wilson"]

from .gauge import OrbitalEmbedding,embed_orbitals
from .topology import wilson_transport
__all__ += ["OrbitalEmbedding","embed_orbitals","wilson_transport"]

from .spectral import SpectralBatch,prepare_spectral
from .responses import GeometryBatch,prepare_geometry,response_from_geometry
__all__ += ["SpectralBatch","prepare_spectral","GeometryBatch","prepare_geometry","response_from_geometry"]

from .symbolic_model import SymbolicHamiltonian,CompiledSymbolic,PauliDecomposition
__all__ += ["SymbolicHamiltonian","CompiledSymbolic","PauliDecomposition"]
