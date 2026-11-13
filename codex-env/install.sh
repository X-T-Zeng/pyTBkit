#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
mkdir -p work
export XDG_CACHE_HOME="$ROOT/work/cache"
export PIP_CACHE_DIR="$ROOT/work/pip-cache"
export CONDA_OVERRIDE_ARCHSPEC=x86_64
export MAMBA_ROOT_PREFIX="$ROOT/work/mamba"
if [[ ! -x work/micromamba ]]; then
  curl -fL --retry 2 https://github.com/mamba-org/micromamba-releases/releases/download/2.10.0-0/micromamba-linux-64 -o work/micromamba
  chmod +x work/micromamba
fi
echo '701f39c06f8e6a670a0e186c8479b9e5c0502a2f0cda6cef65d83bdd2875e8bd  work/micromamba' | sha256sum -c -
if [[ ! -f .toolchain/.codex-complete ]]; then
  # Initialize an empty prefix without registering it in ~/.conda.
  mkdir -p .toolchain/conda-meta
  touch .toolchain/conda-meta/history
  if [[ -f codex-env/toolchain-linux-64.lock ]]; then
    work/micromamba install -y -p "$ROOT/.toolchain" -f codex-env/toolchain-linux-64.lock
  else
    work/micromamba install -y -p "$ROOT/.toolchain" -c conda-forge --strict-channel-priority '_x86_64-microarch-level=1=5_x86_64' python=3.12 gcc_linux-64 gxx_linux-64 gfortran_linux-64 cmake ninja pkg-config openblas libblas liblapack fftw octave make git pip
  fi
  touch .toolchain/.codex-complete
fi
.toolchain/bin/python -m venv .venv
REQ=codex-env/requirements.in
[[ ! -f codex-env/requirements.lock ]] || REQ=codex-env/requirements.lock
.venv/bin/python -m pip install -r "$REQ"
# Keep the reviewed environment lock stable on reruns; never capture editable
# repository paths or unrelated research packages added later to this venv.
if [[ ! -f codex-env/requirements.lock ]]; then
  .venv/bin/python -m pip freeze | sed '/^#/d; /^-e /d; /^tbkit-python==/d' > codex-env/requirements.lock
fi
work/micromamba env export -p "$ROOT/.toolchain" --explicit > codex-env/toolchain-linux-64.lock
bash codex-env/test.sh
