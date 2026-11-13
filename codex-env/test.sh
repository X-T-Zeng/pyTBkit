#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/codex-env/activate.sh"
export NUMBA_CACHE_DIR="$ROOT/work/numba-cache"
export MPLCONFIGDIR="$ROOT/work/matplotlib"
export XDG_CONFIG_HOME="$ROOT/work/config"
export XDG_DATA_HOME="$ROOT/work/data"
mkdir -p "$ROOT/work/build"
cd "$ROOT/work/build"
"$FC" -O2 -fopenmp -fcheck=all "$ROOT/codex-env/tests/smoke.f90" -L"$ROOT/.toolchain/lib" -Wl,-rpath,"$ROOT/.toolchain/lib" -llapack -lblas -o smoke
OMP_NUM_THREADS=2 ./smoke
cmake -S "$ROOT/codex-env/tests" -B native -G Ninja
cmake --build native
ctest --test-dir native --output-on-failure
python -m numpy.f2py -c "$ROOT/codex-env/tests/kernel.f90" -m tbkernel --backend meson --f90flags=-fopenmp -lgomp
OMP_NUM_THREADS=2 PYTHONPATH="$PWD" python -m pytest -q "$ROOT/codex-env/tests/test_numeric.py"
octave --no-gui --quiet "$ROOT/codex-env/tests/smoke.m"
