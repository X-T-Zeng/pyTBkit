# Native Python port coverage (0.2)

The repository contains 880 MATLAB source files. This implementation covers the
major numerical model families and multiple end-to-end research paths; it is not
file-by-file API parity with every MATLAB method. No success-returning placeholder
implementations represent pending capabilities. All MATLAB sources are retained.

| Original area | Native Python | Validation and current boundary |
| --- | --- | --- |
| TBkit, HR, HK, Htrig | common bands/DOS, HR Fourier, polynomial/sin-cos, symbolic extraction | analytic spectra, finite differences, exact HR/fft.m numerical body via Octave |
| sparse HR, EIGENCAR_gen_sparse | CSR storage, COO Numba kernels, sparse shift-invert | dense/sparse numerical parity and real memory benchmark |
| hrdat_read, from_wannier90 | indexed complex data, degeneracies, wsvec expansion | round trips, wsvec phases/gradients; hr.dat requires external lattice |
| HR/Htrig/HK conversion | exact commensurate Fourier/trig, Taylor expansion, quadratic lattice regularization | analytic examples and derivative tests |
| supercell, cut_piece, slab, reseq | arbitrary integer cosets, CSR finite cuts, periodic slabs, basis selection/direct sum | zone folding including skew/negative-determinant supercells, SSH edges |
| POSCAR, neighbor/SK families | POSCAR scales/flags/species, periodic cKDTree, real s/p/d SK | skew-cell brute force, covariance and reverse Hermiticity, structure-to-band example |
| spin/orbital algebra | angular momentum, real orbital rotations, lambda L.S | commutators, Casimirs, multiplet spectra and unitarity |
| overlap/generalized eigensolve | NonOrthogonal, S-normalized eigensystem and dE/dk | generalized equation residuals, finite differences, nonpositive S rejection |
| Oper/group, model constraints | unitary/antiunitary action, group closure up to phase, finite-group projection, exact symbolic invariant k.p basis | TR/inversion, seven spin-linear invariants, covariance |
| spglib wrappers | optional spglib space group, standard cells, atom permutations/cell shifts | simple cubic and diamond nonsymmorphic mappings; s/p/d orbital+spin sewing and HR projection, including nonsymmorphic phases |
| Berry/BCP/NCTE/SOAHC | Numba geometric kernels and streaming BZ quadrature | exact original BCP_k/NCTE_k/SOAHC_int_k via Octave, analytic Berry/metric and derivatives |
| Chern/Wilson/BP_1D | gauge-invariant occupied links, Wilson flow | SSH phase, +/-/zero QWZ Chern, gap/link checks |
| nested/mirror Wilson, Z2/Z4 | isolated Wannier-window nested polarization, constant mirror sectors, inversion indicators and no-inversion 2D Wilson Z2 | BBH topological/trivial phases; mirror opposite-Chern blocks; strong/weak/trivial 3D TI |
| Landau/Peierls/plane waves | Weyl-ordered truncated oscillator HK, finite affine-A Peierls phases, supplied reciprocal-harmonic expansion | harmonic oscillator/Dirac levels, gauge/plaquette, free/coupled plane waves |
| Green/surf/transport | principal-layer decimation, residual refinement, Caroli transmission | analytic chain roots inside/outside band/at center, unit transmission |
| fitting/plotting | model-builder least squares, headless Matplotlib | fitted SSH parameters, generated QWZ plot/data |
| SpinModel | classical tensor exchange/anisotropy/field, JIT energy/field, deterministic LLG, small quantum Heisenberg clusters | energy derivatives, damping/norms, singlet/triplet |
| Hckt/Subckt | native RLC admittance, finite current solve, periodic circuit Laplacian, basic SPICE export | resistor voltages, analytic RLC admittances/periodic dispersion |
| group base/ordinary representation theory | AbstractFiniteGroup, validated Cayley groups, cyclic/dihedral/permutation and strict matrix closure, subgroup/coset/quotient, ordinary characters and isotypic projectors | D3/S3/D4 known subgroup and character data, exact trimer blocks, FS real/complex/quaternionic examples, nonassociative-loop rejection; GROUPS.zh-CN.md lists remaining corepresentation/irrep-generation work |

## Important boundaries

- The original GUI, plotting collection, MATLAB-specific operator overloads,
  property conventions and helper functions are not all exposed as aliases.
- Qnum, linear symbolic BasisFunc, spherical harmonics and Clebsch-Gordan
  transforms are implemented. General radial wavefunctions, every original
  symbolic operation, higher-l hopping and general irrep projectors remain pending.
- Space groups include atom maps, nonsymmorphic Bloch sewing and HR projection
  for closed real s/p/d orbital sets with optional spin. Arbitrary higher-l
  representations and magnetic space groups remain pending.
- Inversion Z2 requires spinful T^2=-1, constant inversion and a periodic orbital
  gauge. Separately, z2_wilson handles 2D inversion-broken spinful insulators;
  it requires converged Wilson meshes, constant T and periodic orbital gauge.
  General 3D no-inversion plane assembly remains pending.
- Nested topology currently returns a 2D isolated phase-sector polarization,
  not every nested flow geometry, mirror plane and higher-dimensional invariant.
- SOAHC/NCTE/BCP kernels match the source formulas. Material-specific SI units,
  scattering models, all nonlinear response variants and arbitrary adaptive BZ
  integration are not automatically supplied.
- Magnetic Peierls construction is finite and affine-A only. Periodic magnetic
  cells/flux quantization and general path quadrature require further work.
- Landau matrices are truncated: top oscillator states can produce spurious
  eigenvalues; convergence in basis size is the caller's responsibility.
- Spin and circuit numerical cores are implemented; Vampire export/import,
  commercial HSPICE decks/results, all device/circuit mesh helpers, and the
  full original exchange-label interface are not claimed.
- EIGENVAL, KPOINTS, charge/noncollinear PROCAR, Wannier win/wout metadata
  parsing is implemented and tested with real repository data. Complete DFT
  ecosystems (vasprun.xml and every format variant), arbitrary MATLAB .mat,
  GPU/distributed execution remain pending. Optional model-level Fortran/OpenMP
  Fourier dispatch is built and compared directly with Numba and NumPy.

## Numerical design

Numba kernels accept arrays and work inside user @njit functions. Parallelism
is over k points, not hopping reduction order, and fastmath is disabled. Shared
persistent caches do not hold model state. Native arrays are copied/read-only;
HR edits return new objects. Non-Hermitian inputs fail unless explicitly
projected. Degenerate single-band quantities are undefined; the default is to
raise, with a documented cutoff/omit mode only for comparison to the source.
Tests compare degenerate-subspace traces rather than gauge-dependent individual
curvatures. See VALIDATION.zh-CN.md for current execution evidence.


MATLAB reference: BUAA-TCMP/TBkit @ `48e5217593598c8e1d634cdee3976e1dc6984516`; see `upstream.json`.
