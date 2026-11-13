"""Consistent scientific figures, with lazy Matplotlib and no global styling.

Plotting preserves supplied data: no smoothing, energy shifts, clipping or band
reconnection. Styling is scoped, and caller-provided axes remain supported.
"""
from pathlib import Path

import numpy as np


# Okabe-Ito colors, excluding the low-contrast yellow on a white background.
PALETTE = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9')


def plot_style(preset='paper', **overrides):
    """Return a Matplotlib rc_context for paper or presentation figures.

    Use ``with plot_style():`` around figure creation. Bundled fonts and MathText
    avoid external font/LaTeX requirements. Import and context exit do not change
    the user's global defaults. overrides are ordinary rcParams names, supplied
    through a dictionary when they contain dots.
    """
    import matplotlib as mpl
    from cycler import cycler
    if preset not in ('paper', 'presentation'):
        raise ValueError('preset must be paper or presentation')
    talk = preset == 'presentation'
    settings = {
        'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 14 if talk else 9.5, 'mathtext.fontset': 'stixsans',
        'text.usetex': False,
        'figure.figsize': (9.6, 5.8) if talk else (4.2, 3.2),
        'figure.dpi': 140, 'figure.facecolor': 'white',
        'figure.constrained_layout.use': True,
        'axes.facecolor': 'white', 'axes.edgecolor': '#4B5563',
        'axes.labelcolor': '#263442', 'text.color': '#263442',
        'axes.labelsize': 15 if talk else 10,
        'axes.titlesize': 16 if talk else 10.5,
        'axes.titleweight': 'medium', 'axes.titlelocation': 'left',
        'axes.titlepad': 10, 'axes.linewidth': .8,
        'axes.prop_cycle': cycler(color=PALETTE),
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.axisbelow': True, 'axes.grid': False,
        'xtick.color': '#4B5563', 'ytick.color': '#4B5563',
        'xtick.labelsize': 12 if talk else 8.5,
        'ytick.labelsize': 12 if talk else 8.5,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
        'xtick.major.width': .8, 'ytick.major.width': .8,
        'grid.color': '#E5E9EE', 'grid.linewidth': .6,
        'lines.linewidth': 2.2 if talk else 1.65,
        'lines.markersize': 5 if talk else 3.5,
        'legend.frameon': False, 'legend.fontsize': 12 if talk else 8,
        'legend.handlelength': 1.8, 'legend.borderaxespad': .5,
        'savefig.dpi': 300, 'savefig.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    }
    settings.update(overrides)
    return mpl.rc_context(settings)


def format_axes(ax, *, grid=False):
    """Apply light, open axes; labels, limits and scientific scales are retained."""
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#4B5563')
    ax.spines[['left', 'bottom']].set_linewidth(.8)
    ax.tick_params(direction='out', width=.8, length=3.5)
    ax.set_axisbelow(True)
    ax.grid(grid, axis='y')
    return ax


def bandplot(distance, energies, *, ticks=None, labels=None, ax=None,
             reference=None, ylabel='Energy (model units)', decorate=True, **kwargs):
    """Plot raw bands and return axes, without displaying or changing energies.

    NaNs are retained as visible breaks. A reference line is opt-in and never
    subtracts from the data. Dense bands use a coherent single color by default;
    explicit line properties override defaults. ticks mark path junctions.
    Caller axes may opt out of decoration with decorate=False.
    """
    import matplotlib.pyplot as plt
    if np.iscomplexobj(distance) or np.iscomplexobj(energies):
        raise ValueError('plot real momenta/energies explicitly; complex data are not silently discarded')
    distance = np.asarray(distance, dtype=float)
    energy = np.ma.asarray(energies, dtype=float).filled(np.nan)
    if energy.ndim == 1:
        energy = energy[:, None]
    if (distance.ndim != 1 or len(distance) < 2 or not np.isfinite(distance).all()
            or np.any(np.diff(distance) < 0) or distance[-1] <= distance[0]):
        raise ValueError('finite nondecreasing path distance with nonzero span required')
    if (energy.ndim != 2 or energy.shape[0] != len(distance) or energy.shape[1] < 1
            or np.isinf(energy).any() or not np.isfinite(energy).any()):
        raise ValueError('energies must be (nk,) or (nk,nband); NaN gaps are allowed')
    if ticks is not None:
        if np.iscomplexobj(ticks):
            raise ValueError('real path ticks required')
        ticks = np.asarray(ticks, dtype=float)
        if (ticks.ndim != 1 or not np.isfinite(ticks).all()
                or np.any(np.diff(ticks) < 0) or np.any(ticks < distance[0])
                or np.any(ticks > distance[-1])):
            raise ValueError('ticks must be ordered finite positions within the path')
        if labels is not None and len(labels) != len(ticks):
            raise ValueError('one label per path tick required')
    elif labels is not None:
        raise ValueError('labels require ticks')
    if reference is not None and (not np.isscalar(reference) or np.iscomplexobj(reference) or not np.isfinite(reference)):
        raise ValueError('finite real energy reference required')
    if ax is None:
        with plot_style():
            _, ax = plt.subplots()
            default_linewidth = plt.rcParams['lines.linewidth']
    else:
        default_linewidth = plt.rcParams['lines.linewidth']
    properties = dict(kwargs)
    if not {'color', 'c'} & properties.keys():
        properties['color'] = PALETTE[0]
    if not {'linewidth', 'lw'} & properties.keys():
        properties['linewidth'] = default_linewidth
    properties.setdefault('zorder', 2)
    ax.plot(distance, energy, **properties)
    if ticks is not None:
        ax.set_xticks(ticks, labels=labels)
        for tick in np.unique(ticks[(ticks > distance[0]) & (ticks < distance[-1])]):
            ax.axvline(tick, color='#DCE2E9', linewidth=.7, zorder=0)
    if reference is not None:
        ax.axhline(reference, color='#8C96A3', linewidth=.8, linestyle='--', zorder=1)
    ax.set(xlim=(distance[0], distance[-1]), xlabel='Wave-vector path', ylabel=ylabel)
    if decorate:
        format_axes(ax)
    return ax


def save_figure(figure, path, *, formats=('png', 'pdf', 'svg'), dpi=300, transparent=False):
    """Export a figure stem to PNG plus editable vector PDF/SVG by default.

    A supported suffix in path is treated as part of the requested stem, so
    'bands.png' and 'bands' both give bands.<format>. Returns paths by format.
    This does not close or display the figure. Parents are created as needed.
    """
    import matplotlib as mpl
    formats = tuple(formats)
    if not formats or len(set(formats)) != len(formats) or any(f not in ('png', 'pdf', 'svg') for f in formats):
        raise ValueError('choose unique formats from png, pdf, svg')
    if not np.isscalar(dpi) or not np.isfinite(dpi) or dpi <= 0:
        raise ValueError('positive finite export dpi required')
    stem = Path(path)
    if stem.suffix.lower() in ('.png', '.pdf', '.svg'):
        stem = stem.with_suffix('')
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    with mpl.rc_context({'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        for extension in formats:
            target = stem.with_name(stem.name+'.'+extension)
            metadata = ({'Date': None} if extension == 'svg' else
                        {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else None)
            figure.savefig(target, dpi=dpi, bbox_inches='tight', pad_inches=.06,
                           transparent=transparent, metadata=metadata)
            outputs[extension] = target
    return outputs
