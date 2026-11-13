import numpy as np
from scipy.optimize import least_squares


def fit_bands(builder,initial,k,target,*,bounds=(-np.inf,np.inf),backend="numba",**kwargs):
    """Fit sorted eigenvalues via a model builder(parameters)->TBkit.

    This does not track bands across degeneracies or resolve gauge/nonunique
    parameters. Inspect the returned scipy OptimizeResult and fit validation.
    """
    target=np.asarray(target,float)
    def residual(p):
        result=builder(p).bands(k,backend=backend)
        if result.shape!=target.shape: raise ValueError("target shape must match bands")
        return (result-target).ravel()
    return least_squares(residual,np.asarray(initial,float),bounds=bounds,**kwargs)
