"""Positive-Gram orbital spectra H c = E S c, with explicit reliability gates.

No indefinite-metric/BdG solver, automatic basis truncation or Berry connection.
"""
from dataclasses import dataclass
from numbers import Real
import numpy as np
from scipy.linalg import eigh, eigvalsh
from .base import points,hermitian
from .hr import immutable
from .generalized_kernels import generalized_diagnostics


def _real_tolerance(value, name, *, positive=False, minimum=0.):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real scalar')
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f'{name} must be finite') from None
    if not np.isfinite(result) or result < minimum or (positive and result == 0.):
        raise ValueError(f'invalid {name}')
    return result


def _component_scale(a):
    return max(np.max(abs(a.real)), np.max(abs(a.imag)))


def _normalize(a, scale):
    # Real-component divisions preserve subnormals without complex reciprocals.
    return np.ascontiguousarray(a.real/scale + 1j*(a.imag/scale))


def _energy_units(values, hscale, sscale):
    # Multiplying hscale/sscale first can overflow even for finite final E.
    mantissa, exponent = np.frexp(values)
    hm, he = np.frexp(hscale)
    sm, se = np.frexp(sscale)
    with np.errstate(over='ignore', under='ignore'):
        out = np.ldexp(mantissa*(hm/sm), exponent+he-se)
    if not np.isfinite(out).all() or np.any((values != 0.) & (out == 0.)):
        raise ValueError('generalized energies not representable in float64; change energy units')
    return out


def generalized_diagnostics_numpy(h, s, e, u):
    """Independent vectorized reference, same validated normalized array domain."""
    hc, sc = h@u, s@u
    numerator = np.linalg.norm(hc-sc*e[:, None, :], axis=1)
    denominator = ((np.linalg.norm(h, axis=(-2, -1))[:, None]
                    + abs(e)*np.linalg.norm(s, axis=(-2, -1))[:, None])
                   * np.linalg.norm(u, axis=1))
    residual = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0.)
    residual[(denominator == 0.) & (numerator != 0.)] = np.inf
    metric = np.linalg.norm(u.conj().swapaxes(-1, -2)@sc-np.eye(u.shape[-1]), axis=(-2, -1))
    return residual, metric


@dataclass(frozen=True)
class GeneralizedSpectrum:
    """Caller-owned read-only batch; eigenvectors are columns in the INPUT basis.

    k (nk,dim), energy (nk,n), eigenvectors (nk,n,n), overlap_condition (nk,),
    residual (nk,n), metric_error (nk,). Residuals were checked in separately
    scaled H/S units, metric_error is ||C† S C-I||_F. No forward-error guarantee.
    """
    k: np.ndarray
    energy: np.ndarray
    eigenvectors: np.ndarray
    overlap_condition: np.ndarray
    residual: np.ndarray
    metric_error: np.ndarray
    single: bool = False

    def __post_init__(self):
        if any(np.iscomplexobj(getattr(self,name)) for name in
               ('k','energy','overlap_condition','residual','metric_error')):
            raise ValueError('real generalized spectral coordinates and diagnostics required')
        for name in ('k', 'energy', 'eigenvectors', 'overlap_condition', 'residual', 'metric_error'):
            value = immutable(getattr(self, name), complex if name == 'eigenvectors' else float)
            object.__setattr__(self, name, value)
        if self.k.ndim!=2 or not len(self.k) or self.energy.ndim!=2 or not self.energy.shape[1]:
            raise ValueError('nonempty batched generalized spectra required')
        nk,n=self.energy.shape
        if (len(self.k)!=nk or self.eigenvectors.shape!=(nk,n,n) or self.residual.shape!=(nk,n)
                or self.overlap_condition.shape!=(nk,) or self.metric_error.shape!=(nk,)):
            raise ValueError('generalized spectral shapes differ')
        if np.any(self.overlap_condition<1.) or np.any(self.residual<0.) or np.any(self.metric_error<0.):
            raise ValueError('invalid generalized spectral diagnostics')
        if self.single and nk!=1:raise ValueError('single-point spectrum must have one k point')


class NonOrthogonal:
    def __init__(self,hamiltonian,overlap):
        if hamiltonian.dim!=overlap.dim or hamiltonian.norb!=overlap.norb:
            raise ValueError("H/S dimensions differ")
        if hasattr(hamiltonian,'lattice') and hasattr(overlap,'lattice') and not np.allclose(hamiltonian.lattice,overlap.lattice):
            raise ValueError("H/S lattices differ")
        self.h,self.s=hamiltonian,overlap
        self.dim,self.norb=hamiltonian.dim,hamiltonian.norb

    def spectrum(self,k,*,backend='numba',chunk_size=256,condition_limit=1e8,check_tol=1e-8):
        """Solve and certify a positive-Gram spectrum, with no implicit truncation.

        Reject lambda_max(S)/lambda_min(S) > condition_limit before solving H/S.
        check_tol bounds per-band normwise residual and Frobenius frame error.
        Both bounds are dimensionless. Raising limits is an explicit accuracy
        tradeoff, not a way to recover information already lost to rounding.
        """
        limit = _real_tolerance(condition_limit, 'condition_limit', minimum=1.)
        tolerance = _real_tolerance(check_tol, 'check_tol', positive=True)
        k,single=points(k,self.dim)
        if isinstance(chunk_size, (bool, np.bool_)) or not isinstance(chunk_size, (int, np.integer)) or chunk_size<1 or not len(k):
            raise ValueError("nonempty k and positive integer chunk size required")
        energies=[]; vectors=[]; conditions=[]; residuals=[]; metric_errors=[]
        for start in range(0,len(k),chunk_size):
            kk=k[start:start+chunk_size]
            h=self.h.hamiltonian(kk,backend=backend); s=self.s.hamiltonian(kk,backend=backend)
            # Explicit zero absolute tolerance also works on main's older
            # elementwise helper. PR #10 supplies the shared Frobenius policy.
            hermitian(h,atol=0.); hermitian(s,atol=0.)
            hh=np.empty_like(h,dtype=complex); ss=np.empty_like(s,dtype=complex)
            ee=np.empty((len(kk),self.norb)); uu=np.empty_like(h,dtype=complex)
            hscales=[]; sscales=[]
            for q,(hi,si) in enumerate(zip(h,s)):
                hs=_component_scale(hi) or 1.; sc=_component_scale(si)
                if sc == 0.: raise ValueError("overlap must be positive definite")
                hh[q]=_normalize(hi,hs); ss[q]=_normalize(si,sc)
                se=eigvalsh(ss[q],check_finite=False)
                if se[0]<=0.: raise ValueError("overlap must be positive definite")
                if se[0]<se[-1]/limit:
                    raise ValueError(f'ill-conditioned overlap at k index {start+q}: '
                                     f'condition estimate {se[-1]/se[0]:.6g} exceeds condition_limit={limit:.6g}')
                conditions.append(se[-1]/se[0])
                ee[q],uu[q]=eigh(hh[q],ss[q],check_finite=False)
                hscales.append(hs); sscales.append(sc)
            diagnostic=generalized_diagnostics if backend=='numba' else generalized_diagnostics_numpy
            residual,metric_error=diagnostic(hh,ss,ee,uu)
            if (not np.isfinite(residual).all() or not np.isfinite(metric_error).all()
                    or np.any(residual>tolerance) or np.any(metric_error>tolerance)):
                raise ValueError('generalized eigensystem failed residual/metric checks; examine basis conditioning')
            for q in range(len(kk)):
                energies.append(_energy_units(ee[q],hscales[q],sscales[q]))
                vectors.append(uu[q]/np.sqrt(sscales[q]))
            residuals.extend(residual); metric_errors.extend(metric_error)
        return GeneralizedSpectrum(k,np.array(energies),np.array(vectors),np.array(conditions),
                                   np.array(residuals),np.array(metric_errors),single)

    def bands(self,k,*,eigenvectors=False,backend='numba',chunk_size=256,condition_limit=1e8,check_tol=1e-8):
        result=self.spectrum(k,backend=backend,chunk_size=chunk_size,condition_limit=condition_limit,check_tol=check_tol)
        e,u=result.energy,result.eigenvectors
        if result.single:e=e[0];u=u[0]
        # Keep historical writable bands results; spectrum is the immutable API.
        return (e.copy(),u.copy()) if eigenvectors else e.copy()

    def band_gradient(self,k,*,backend='numba',gap_tol=None,gap_rtol=None,condition_limit=1e8,check_tol=1e-8):
        """HF c†(dH-E dS)c in the supplied differentiable orbital basis.

        Default degeneracy threshold is 1e-9 of the local total bandwidth,
        invariant to energy units and offsets up to roundoff. Explicit gap_tol
        retains absolute-energy semantics and disables default relative cutoff;
        gap_rtol explicitly adds a relative cutoff. Exact degeneracy is rejected
        even with gap_rtol=0. This is not an orthogonal Berry/velocity API.
        """
        absolute=0. if gap_tol is None else _real_tolerance(gap_tol,'gap_tol',positive=True)
        relative=(1e-9 if gap_tol is None else 0.) if gap_rtol is None else _real_tolerance(gap_rtol,'gap_rtol')
        e,u=self.bands(k,eigenvectors=True,backend=backend,condition_limit=condition_limit,check_tol=check_tol)
        scale=np.max(abs(e),axis=-1,keepdims=True)
        scale=np.where(scale==0.,1.,scale)
        normalized=e/scale
        with np.errstate(over='ignore',invalid='ignore'):
            raw=e[..., :,None]-e[...,None,:]
            gaps=np.where(np.isfinite(raw),abs(raw)/scale[...,None],
                          abs(normalized[..., :,None]-normalized[...,None,:]))
            absolute_scaled=absolute/scale[...,None]
        bandwidth=np.max(gaps,axis=(-2,-1),keepdims=True)
        threshold=np.maximum(absolute_scaled,relative*bandwidth)
        if np.any(((gaps==0.)|(gaps<threshold))&~np.eye(self.norb,dtype=bool)):
            raise ValueError("band gradient at a degeneracy requires directional degenerate perturbation theory")
        dh=self.h.gradient(k,backend=backend); ds=self.s.gradient(k,backend=backend)
        hermitian(dh,atol=0.);hermitian(ds,atol=0.)
        with np.errstate(over='ignore',invalid='ignore'):
            diagonal_h=np.einsum('...in,...aij,...jn->...an',u.conj(),dh,u)
            diagonal_s=np.einsum('...in,...aij,...jn->...an',u.conj(),ds,u)
            gradient=(diagonal_h-e[...,None,:]*diagonal_s).real
        if not np.isfinite(gradient).all():raise ValueError('nonfinite generalized band gradient; examine scales and conditioning')
        return gradient
