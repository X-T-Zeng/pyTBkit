These smoke tests validate the installed execution environment, not TBkit's full MATLAB implementation.

- `kernel.f90`: positive energy of H(k) = sin(k) sigma_x + (m + cos(k)) sigma_z, double precision, OpenMP, callable from Python through f2py.
- `smoke.f90`: LAPACK eigenvalues and a two-thread OpenMP reduction.
- `test_numeric.py`: independent analytic/matrix comparison, numerical/symbolic integration, Numba, HDF5 and plotting.
- `smoke.m`: Octave complex Hermitian eigenvalues, quadrature and FFT round trip.
- `CMakeLists.txt`, `fft.c`, `check.cpp`: native C/FFTW and C++ builds with CTest.

Run all tests from the repository root using `bash codex-env/test.sh`.
