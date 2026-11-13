# Contributing

The long-term objective is a useful, multifunctional and efficient native Python
toolkit informed by official BUAA-TCMP/TBkit. Python is the development mainline;
MATLAB is also audited, but behavior is not copied when it contradicts physics.

1. Start a branch from current main. Keep independent changes reviewable and avoid
   rewriting other contributors' commits or generated references.
2. State the mathematical convention: basis/orbital gauge, k coordinates,
   Fourier sign, units, band/occupied-subspace assumptions and normalization.
3. For a physics bug, include an analytic derivation, counterexample or independent
   reference test. A round trip or agreement with the same erroneous formula is
   insufficient. Distinguish a source discrepancy from a confirmed physical bug.
4. For symmetry construction test allowed-space completeness, independence,
   group action and antiunitary signs, as well as residuals of generated terms.
5. For degeneracies use subspace/gauge-invariant quantities or reject an undefined
   single-band observable. Do not silently Hermitize incorrect input.
6. Run relevant tests and complete the validation suite before publishing a release.
   Record skips and external-tool limitations; commercial MATLAB is not required.
7. Benchmark warm execution separately from JIT compilation with fixed seeds,
   thread counts and numerical error. Keep a reference NumPy implementation.
8. Never commit virtual environments, downloaded MATLAB repositories, scientific
   input with secrets, caches or machine-specific compiled extensions.

Setup, pinned upstream retrieval and full validation commands are in README.md.
Current source provenance is python/upstream.json; changes to that pin require a
reviewed diff and revalidation, not an automatic version bump.
