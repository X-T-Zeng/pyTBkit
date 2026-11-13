"""Optional F2PY model dispatch. No automatic compilation during imports."""
import numpy as np


def evaluate(kind,k,vectors,hopping):
    try:from . import _tbkit_fortran
    except ImportError as exc:
        raise ImportError('build optional backend: python python/tools/build_fortran.py') from exc
    function=getattr(_tbkit_fortran,kind)
    return np.ascontiguousarray(function(np.asfortranarray(k),np.asfortranarray(vectors),np.asfortranarray(hopping)))
