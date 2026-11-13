import numpy as np


def bandplot(distance,energies,*,ticks=None,labels=None,ax=None,**kwargs):
    """Return a Matplotlib axes; no interactive display side effects."""
    import matplotlib.pyplot as plt
    if ax is None: _,ax=plt.subplots()
    ax.plot(distance,np.asarray(energies),**kwargs)
    if ticks is not None: ax.set_xticks(ticks,labels=labels)
    ax.set_xlabel('k path'); ax.set_ylabel('Energy')
    return ax
