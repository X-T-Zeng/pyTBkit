# Source from bash; no shell startup files are modified.
CODEX_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export XDG_CACHE_HOME="$CODEX_REPO/work/cache"
export MAMBA_ROOT_PREFIX="$CODEX_REPO/work/mamba"
eval "$("$CODEX_REPO/work/micromamba" shell hook -s bash)"
micromamba activate "$CODEX_REPO/.toolchain"
source "$CODEX_REPO/.venv/bin/activate"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0 MPLBACKEND=Agg
export FC=${FC:-x86_64-conda-linux-gnu-gfortran}
export CC=${CC:-x86_64-conda-linux-gnu-cc}
export CXX=${CXX:-x86_64-conda-linux-gnu-c++}
export PIP_CACHE_DIR="$CODEX_REPO/work/pip-cache"
export NUMBA_CACHE_DIR="$CODEX_REPO/work/numba-cache"
export MPLCONFIGDIR="$CODEX_REPO/work/matplotlib"
