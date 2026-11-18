"""Two-band spectra: energy units and positive orbital Gram normalization.

Run with PYTHONPATH=python; results are dimensionless. No SI velocity or
nonorthogonal Berry connection is inferred from the supplied H and S.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tbkit import HK, NonOrthogonal


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    k = np.linspace(-2., 2., 301)[:, None]
    sx = np.array([[0., 1.], [1., 0.]])
    sz = np.diag([1., -1.])
    analytic = np.sqrt(k[:, 0]**2+.4**2)[:, None]*[-1., 1.]
    colors = ['#245b8e', '#d77a32', '#3b8a78']
    report = {'energy_scales': [1e-200, 1., 1e200], 'rejected_nonhermitian': 0,
              'rejected_overlap': 0, 'max_relative_spectral_error': 0.,
              'scope': 'Hermitian two-band H; positive orbital Gram S; no indefinite BdG metric'}
    with plt.rc_context({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelsize': 12, 'savefig.dpi': 180}):
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.), constrained_layout=True)
        for scale, color, style in zip(report['energy_scales'], colors, ['-', '--', ':']):
            h = HK({(0,): scale*.4*sz, (1,): scale*sx})
            e = h.bands(k)/scale
            error = float(np.max(abs(e-analytic)/abs(analytic)))
            assert error < 2e-14
            report['max_relative_spectral_error'] = max(error, report['max_relative_spectral_error'])
            for band in range(2):
                axes[0].plot(k[:, 0], e[:, band], style, color=color, linewidth=2.,
                             label=f'$E_0=10^{{{int(np.log10(scale))}}}$' if band == 0 else None)
            bad = scale*np.array([[0, 1j], [1j, 0]])
            try:
                HK({(0,): bad}).bands([0.])
            except ValueError:
                report['rejected_nonhermitian'] += 1
            else:
                raise AssertionError('non-Hermitian H accepted')
            # Common rescaling H,S is a normalization change of the orbital basis.
            h0 = HK({(0,): scale*.4*sz, (1,): scale*sx})
            s = HK({(0,): scale*np.diag([1., 2.])})
            e = NonOrthogonal(h0, s).bands(k)
            # det(H-E S)=2E²-.4E-(.16+k²).
            generalized = np.column_stack(((.4-np.sqrt(.16+8*(.16+k[:, 0]**2)))/4,
                                           (.4+np.sqrt(.16+8*(.16+k[:, 0]**2)))/4))
            np.testing.assert_allclose(e, generalized, rtol=2e-14, atol=0)
            for band in range(2):
                axes[1].plot(k[:, 0], e[:, band], style, color=color, linewidth=2.)
            invalid_s = HK({(0,): scale*np.array([[1, .4j], [0, 1]])})
            try:
                NonOrthogonal(h0, invalid_s).bands([0.])
            except ValueError:
                report['rejected_overlap'] += 1
            else:
                raise AssertionError('non-Hermitian S accepted')
        for ax, title, ylabel in zip(axes, ['Energy-unit invariance', 'Orbital-normalization invariance'],
                                     ['$E/E_0$', '$E$ (fixed physical units)']):
            ax.set(xlabel='$k$ (dimensionless)', ylabel=ylabel, title=title)
            ax.axhline(0, color='#d0d4da', linewidth=.7, zorder=0)
            ax.grid(alpha=.16)
        axes[0].legend(frameon=False, fontsize=9)
        axes[1].text(.38, .95, '$S=\\mathrm{diag}(1,2)$\nCommon scaling of $H$ and $S$',
                     transform=axes[1].transAxes, va='top', fontsize=9)
        fig.savefig(output/'hermiticity-units.png')
        fig.savefig(output/'hermiticity-units.svg')
        plt.close(fig)
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='work/hermiticity-units')
    print(json.dumps(run(parser.parse_args().output), indent=2))
