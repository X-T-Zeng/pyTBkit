"""Operator-domain checks independent of a solver's selected input triangle."""
import numpy as np
import pytest
from numba import njit
from scipy import sparse
from numpy.testing import assert_allclose as close
from tbkit import HR, HK, NonOrthogonal, prepare_spectral
from tbkit.base import hermitian, hermitian_sparse
from tbkit.validation_kernels import hermiticity_metrics


SCALES = (1e-200, 1e-20, 1., 1e20, 1e200)
BAD = np.array([[0, 1j], [1j, 0]])  # eigenvalues +/-i, not +/-1
GOOD = np.array([[2, 1j], [-1j, -2]])  # eigenvalues +/-sqrt(5)


@pytest.mark.parametrize('scale', SCALES)
@pytest.mark.parametrize('backend', ('numpy', 'numba'))
def test_dense_analytic_spectrum_and_complex_counterexample(scale, backend):
    close(HR([[0]], [scale*GOOD]).bands([0.], backend=backend)/scale,
          [-np.sqrt(5), np.sqrt(5)], rtol=2e-14, atol=0)
    with pytest.raises(ValueError, match='Hermitian'):
        HR([[0]], [scale*BAD]).bands([0.], backend=backend)


@pytest.mark.parametrize('scale', SCALES)
def test_sparse_rejects_before_hermitian_arpack(scale, monkeypatch):
    import tbkit.base as base
    def forbidden(*args, **kwargs):
        pytest.fail('non-Hermitian matrix reached eigsh')
    monkeypatch.setattr(base, 'eigsh', forbidden)
    h = sparse.block_diag((BAD, np.diag([3., 5.]))).toarray()
    with pytest.raises(ValueError, match='Hermitian'):
        HR([[0]], [scale*h]).sparse_bands([0.], 1)


@pytest.mark.parametrize('scale', SCALES)
def test_positive_gram_metric_domain_is_basis_normalization_invariant(scale):
    # Multiplying every basis ket by sqrt(scale) scales BOTH H and S.
    # Generalized energies stay unchanged, and an invalid S stays invalid.
    h = HK({(0,): scale*np.diag([1., 3.])})
    s = HK({(0,): scale*np.diag([2., 4.])})
    close(NonOrthogonal(h, s).bands([0.]), [.5, .75], rtol=2e-14)
    for bad in (np.array([[1, .4j], [0, 1]]), np.diag([1., -1.])):
        with pytest.raises(ValueError):
            NonOrthogonal(h, HK({(0,): scale*bad})).bands([0.])


@pytest.mark.parametrize('scale', SCALES)
def test_accidentally_hermitian_point_does_not_certify_derivative(scale):
    h = HK({(0,): scale*np.diag([1., 3.]), (1,): scale*BAD})
    # prepare_spectral checks dH BEFORE the change of eigenbasis.
    with pytest.raises(ValueError, match='Hermitian'):
        prepare_spectral(h, [0.])
    # NonOrthogonal's derivative shares the same physical-domain validator.
    with pytest.raises(ValueError, match='Hermitian'):
        NonOrthogonal(h, HK({(0,): np.eye(2)})).band_gradient([0.], gap_tol=scale*.1)


def test_frobenius_basis_and_batch_invariance():
    rng = np.random.default_rng(20261117)
    u, _ = np.linalg.qr(rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)))
    for matrix, accepted in ((GOOD, True), (GOOD+1e-7*BAD, False)):
        for a in (matrix, u@matrix@u.conj().T):
            if accepted:
                hermitian(a)
            else:
                with pytest.raises(ValueError):
                    hermitian(a)
    # A huge valid point must never set the tolerance of a tiny invalid one.
    with pytest.raises(ValueError):
        hermitian(np.array([1e200*GOOD, 1e-200*BAD]))
    h = HK({(0,): GOOD, (1,): 1j*np.eye(2)})
    for chunk in (1, 2, 100):
        with pytest.raises(ValueError):
            h.bands([[0.], [1e-8]], chunk_size=chunk)


@pytest.mark.parametrize('validator,convert', ((hermitian, np.asarray), (hermitian_sparse, sparse.csr_matrix)))
def test_overflow_subnormal_exact_and_explicit_absolute_tolerance(validator, convert):
    # abs(1.7e308+1.7e308j) overflows, but its components are finite.
    h = np.array([[1.7e308, 1.7e308*(1+1j)], [1.7e308*(1-1j), -1.7e308]])
    validator(convert(h))
    with pytest.raises(ValueError):
        validator(convert(1.7e308*BAD))
    tiny = np.nextafter(0., 1.)
    validator(convert(tiny*GOOD))
    with pytest.raises(ValueError):
        validator(convert(tiny*BAD), rtol=0)
    # Exact and absolute-only checks must not lose this defect on scaling H.
    h = np.diag([1e300, 1.]).astype(complex)
    h[0, 1] = 1e-300
    for atol in (0., 1e-301):
        with pytest.raises(ValueError):
            validator(convert(h), atol=atol, rtol=0)
    validator(convert(h), atol=2e-300, rtol=0)
    validator(convert(1e-20*BAD), 1e-19)  # positional atol retained, explicit units
    validator(convert(np.zeros((2, 2))), rtol=0)


@pytest.mark.parametrize('invalid', (-1, np.inf, np.nan, True, 1j, '1e-10', [1e-10]))
def test_tolerance_domain(invalid):
    for function, h in ((hermitian, GOOD), (hermitian_sparse, sparse.csr_matrix(GOOD))):
        for keyword in ('rtol', 'atol'):
            with pytest.raises(ValueError, match='tolerances'):
                function(h, **{keyword: invalid})


def test_sparse_duplicates_inputs_and_large_nondense_storage(monkeypatch):
    # Duplicate entries are legitimate sparse storage, not separate amplitudes.
    h = sparse.coo_matrix(([.4j, .6j, -1j], ([0, 0, 1], [1, 1, 0])), shape=(2, 2))
    before = h.data.copy()
    hermitian_sparse(h)
    close(h.data, before, atol=0, rtol=0)
    def forbidden(*args, **kwargs):
        pytest.fail('sparse validation must not densify')
    monkeypatch.setattr(sparse.csr_matrix, 'toarray', forbidden)
    hermitian_sparse(sparse.eye(100000, format='csr', dtype=complex))


def test_metrics_numpy_reference_and_nopython_composition():
    rng = np.random.default_rng(8)
    h = rng.normal(size=(31, 4, 4))+1j*rng.normal(size=(31, 4, 4))
    @njit
    def composed(a):
        scale, norm, defect, exact = hermiticity_metrics(a)
        return defect/norm, exact
    relative, exact = composed(h)
    expected = np.linalg.norm(h-h.conj().swapaxes(-1, -2), axis=(-2, -1))/np.linalg.norm(h, axis=(-2, -1))
    close(relative, expected, rtol=8e-16, atol=0)
    assert not exact.any()
    assert composed.nopython_signatures
    hermitian(np.empty((0, 2, 2)))
