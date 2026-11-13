"""Pinned official MATLAB source, explicitly separated from proposed fixes."""
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2]
CONFIG=json.loads((PROJECT/'python/upstream.json').read_text())


def source_root(*,patched=False):
    folder=PROJECT/CONFIG['patched_checkout' if patched else 'checkout']
    if not (folder/'src').exists():
        raise RuntimeError('prepare the pinned official reference: python python/tools/prepare_upstream.py')
    return folder
