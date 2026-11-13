#!/usr/bin/env bash
# Complete reusable cloud setup for the Python port, including native validation.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
bash codex-env/install.sh
source codex-env/activate.sh
python -m pip install -r python/requirements-tested.txt
python -m pip install --no-deps -e ./python
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-2}
if [[ ${BUILD_FORTRAN:-1} == 1 ]]; then
    python python/tools/build_fortran.py
fi
python python/tools/validate.py --output work/python-validation --benchmarks
