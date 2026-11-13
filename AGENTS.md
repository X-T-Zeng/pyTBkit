# pyTBkit working instructions

- Develop the Python toolkit as the main product. Use BUAA-TCMP/TBkit at the
  reviewed commit in python/upstream.json as the only canonical MATLAB source.
- Read python/LONG_TASK.zh-CN.md, python/migration-status.json and the current
  physics audit before choosing the next independent task.
- Preserve other contributors' work; use a feature/fix branch and a reviewable PR.
  Never force-push main or claim all original MATLAB behavior is implemented.
- Prioritize physical correctness (Hermiticity, units, gauges, group/antiunitary
  actions, degeneracies and convergence) before performance or more APIs.
- Treat exact symbolic construction, derivation and scientific presentation as
  core product features. Preserve symbolic parameters until explicit numerical
  lowering. Read python/SYMBOLIC.zh-CN.md and its pending migration checklist;
  do not drop original symbolic/display workflows when optimizing numeric code.
- Include a runnable example and independent regression for each substantive
  numerical change. Explain assumptions and unsupported cases explicitly.
- Keep Numba nopython kernels array-based, with no hidden cross-model mutable
  cache. Validate changes against a distinct reference and measure actual loads.
- Do not treat tests of patched official files as tests of pristine official
  behavior. Reference downloads stay under work/, not in commits.
- Keep scientific audit findings, source revision and validation records current.
  Full validation: python python/tools/validate.py --output work/python-validation
  --benchmarks (after environment setup, upstream retrieval and native build).
