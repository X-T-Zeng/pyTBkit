"""Plotting must preserve scientific data and caller state across export."""
import xml.etree.ElementTree as ET
import numpy as np
import pytest

matplotlib=pytest.importorskip('matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tbkit.plotting import bandplot,plot_style,save_figure


def test_bandplot_preserves_raw_energies_gaps_and_explicit_line_aliases():
    distance=np.array([0.,.4,.9,1.4])
    energies=np.array([[-2.,2.],[-1.,np.nan],[-1.5,1.5],[-2.,2.]])
    original=energies.copy()
    fig,ax=plt.subplots()
    result=bandplot(distance,energies,ticks=[0,.9,1.4],labels=['G','X','G'],ax=ax,
                    reference=.25,c='#333333',lw=.9)
    assert result is ax
    for band,line in enumerate(ax.lines[:2]):
        np.testing.assert_array_equal(line.get_xdata(),distance)
        np.testing.assert_array_equal(line.get_ydata(),original[:,band])
    np.testing.assert_array_equal(energies,original)
    # NaN breaks and the reference line remain distinct from actual bands.
    assert np.isnan(ax.lines[1].get_ydata()[1])
    assert np.all(np.asarray(ax.lines[-1].get_ydata())==.25)
    plt.close(fig)
    with pytest.raises(ValueError):bandplot(distance,energies+0j)
    with pytest.raises(ValueError):bandplot(distance,energies,ticks=[0,2])


def test_style_context_restores_user_configuration_even_on_error():
    original={k:matplotlib.rcParams[k] for k in ('font.size','axes.spines.top','axes.prop_cycle')}
    with pytest.raises(RuntimeError):
        with plot_style('presentation',**{'font.size':18}):
            assert matplotlib.rcParams['font.size']==18
            raise RuntimeError('user computation fails inside plotting context')
    for key,value in original.items():
        assert matplotlib.rcParams[key]==value


def test_masked_bands_and_vector_exports_are_editable(tmp_path):
    energies=np.ma.array([1.,2.,3.],mask=[False,True,False])
    with plot_style():
        ax=bandplot([0,.5,1],energies)
        assert np.isnan(ax.lines[0].get_ydata()[1])
        ax.set_title('Editable spectrum')
        fonttype=matplotlib.rcParams['svg.fonttype']
        files=save_figure(ax.figure,tmp_path/'nested'/'spectrum.png')
        assert matplotlib.rcParams['svg.fonttype']==fonttype
    assert all(p.stat().st_size>500 for p in files.values())
    assert files['pdf'].read_bytes().startswith(b'%PDF')
    document=ET.parse(files['svg']).getroot()
    labels=[''.join(node.itertext()) for node in document.iter('{http://www.w3.org/2000/svg}text')]
    assert 'Editable spectrum' in labels
    assert plt.fignum_exists(ax.figure.number)  # exporting leaves caller's figure open
    plt.close(ax.figure)
