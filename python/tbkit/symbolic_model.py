"""Exact research models and presentation, with an explicit numerical boundary.

The symbolic model is not a TBkit numerical model: parameters remain symbolic
until to_hk/compile is requested. No automatic rational recovery is performed.
"""
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType

import numpy as np
import sympy as sp
from numba import njit, prange

from .base import TBkit, points


def _coordinates(coordinates):
    coordinates = tuple(coordinates)
    if (not coordinates or len(set(coordinates)) != len(coordinates)
            or any(not isinstance(k, sp.Symbol) or k.is_real is not True
                   for k in coordinates)):
        raise ValueError('distinct, explicitly real momentum Symbols required')
    return coordinates


def _matrix(matrix):
    matrix = sp.ImmutableMatrix(matrix)
    if matrix.rows == 0 or matrix.rows != matrix.cols:
        raise ValueError('nonempty square symbolic matrix required')
    if any(e.has(sp.nan, sp.oo, -sp.oo, sp.zoo) for e in matrix):
        raise ValueError('nonfinite symbolic matrix')
    return matrix


def _zero_status(matrix):
    statuses = [sp.simplify(e).is_zero for e in matrix]
    if all(s is True for s in statuses):
        return True
    if any(s is False for s in statuses):
        return False
    return None


@dataclass(frozen=True, init=False)
class SymbolicHamiltonian:
    """Immutable exact matrix with declared real Cartesian momentum variables.

    Real/complex parameter assumptions belong to the supplied SymPy Symbols.
    Floats stay floats. Scalar energy shifts must be written as scalar*eye(n).
    This class does not infer a lattice, orbital embedding, units or periodicity.
    """
    matrix: sp.ImmutableMatrix
    coordinates: tuple

    def __init__(self, matrix, coordinates):
        object.__setattr__(self, 'matrix', _matrix(matrix))
        object.__setattr__(self, 'coordinates', _coordinates(coordinates))

    @property
    def dim(self):
        return len(self.coordinates)

    @property
    def norb(self):
        return self.matrix.rows

    @property
    def parameters(self):
        return tuple(sorted(self.matrix.free_symbols-set(self.coordinates), key=sp.default_sort_key))

    def to_sympy(self):
        return self.matrix

    def __repr__(self):
        return f'SymbolicHamiltonian(norb={self.norb}, dim={self.dim}, parameters={self.parameters})'

    def pretty(self, *, use_unicode=True):
        return sp.pretty(self.matrix, use_unicode=use_unicode)

    def latex(self, **settings):
        return sp.latex(self.matrix, **settings)

    def _repr_latex_(self):
        return '$\\displaystyle ' + self.latex() + '$'

    def subs(self, substitutions, *, simultaneous=True):
        """Return a new model; simultaneous substitution also permits k swaps."""
        return type(self)(self.matrix.subs(substitutions, simultaneous=simultaneous), self.coordinates)

    def simplify(self):
        return type(self)(self.matrix.applyfunc(sp.simplify), self.coordinates)

    def expand(self):
        return type(self)(self.matrix.applyfunc(sp.expand), self.coordinates)

    def collect(self):
        """Collect polynomial momentum powers; leave parameter coefficients exact."""
        return type(self)(self.matrix.applyfunc(lambda e: sp.collect(sp.expand(e), self.coordinates)), self.coordinates)

    def rewrite(self, target):
        """Explicit SymPy rewrite, for example target=sin or exp."""
        return type(self)(self.matrix.applyfunc(lambda e: e.rewrite(target)), self.coordinates)

    def derivative(self, axis, order=1):
        if not isinstance(axis, (int, np.integer)) or not 0 <= axis < self.dim:
            raise ValueError('invalid derivative axis')
        if not isinstance(order, (int, np.integer)) or order < 0:
            raise ValueError('nonnegative integer derivative order required')
        return type(self)(self.matrix.diff(self.coordinates[axis], int(order)), self.coordinates)

    def dagger(self):
        return type(self)(self.matrix.adjoint(), self.coordinates)

    def __add__(self, other):
        if not isinstance(other, SymbolicHamiltonian):
            return NotImplemented
        if self.coordinates != other.coordinates or self.norb != other.norb:
            raise ValueError('symbolic model coordinates/shape differ')
        return type(self)(self.matrix+other.matrix, self.coordinates)

    def __radd__(self, other):
        if isinstance(other, (int, float, complex, sp.Number)) and other == 0:
            return self
        return NotImplemented

    def __neg__(self):
        return self*(-1)

    def __sub__(self, other):
        if not isinstance(other, SymbolicHamiltonian):
            return NotImplemented
        return self+(-other)

    def __mul__(self, scalar):
        scalar = sp.sympify(scalar)
        if not isinstance(scalar, sp.Expr) or scalar.is_commutative is not True:
            return NotImplemented
        return type(self)(scalar*self.matrix, self.coordinates)

    def __rmul__(self, scalar):
        return self*scalar

    def hermiticity_residual(self):
        return sp.ImmutableMatrix((self.matrix-self.matrix.adjoint()).applyfunc(sp.simplify))

    def is_hermitian(self):
        """True/False/None: unknown symbolic assumptions are never a pass."""
        return _zero_status(self.hermiticity_residual())

    def transform(self, operation):
        """sign U H(R k)^(*) U^dagger, using Oper's input-action convention.

        Supports constant unitary/antiunitary operations, including antisymmetry.
        Momentum-dependent Bloch sewing is a separate, unimplemented operation.
        """
        from .operators import Oper
        if not isinstance(operation, Oper):
            raise TypeError('Oper required')
        u, r = operation.exact_matrices()
        if u.rows != self.norb or r.rows != self.dim:
            raise ValueError('operator/model dimensions differ')
        replacement = dict(zip(self.coordinates, r*sp.Matrix(self.coordinates)))
        h = self.matrix.subs(replacement, simultaneous=True)
        if operation.antiunitary:
            h = h.conjugate()
        h = (-1 if operation.antisymmetry else 1)*u*h*u.adjoint()
        return type(self)(h, self.coordinates).simplify()

    def symmetry_residual(self, operation):
        return sp.ImmutableMatrix((self.matrix-self.transform(operation).matrix).applyfunc(sp.simplify))

    def polynomial_terms(self):
        """Exact coefficient matrices keyed by momentum multi-index."""
        terms = {}
        for i in range(self.norb):
            for j in range(self.norb):
                for powers, coefficient in sp.Poly(sp.expand(self.matrix[i,j]), *self.coordinates).terms():
                    if powers not in terms:
                        terms[powers] = sp.zeros(self.norb)
                    terms[powers][i,j] = coefficient
        return MappingProxyType({p: sp.ImmutableMatrix(c) for p,c in sorted(terms.items())})

    def taylor(self, order, *, point=None):
        """Total momentum degree <= order about a real point (default origin).

        Uses a common scaling variable, rather than retaining a rectangular
        per-axis truncation. Nonanalytic/Laurent expansions are rejected when
        the scaling series is not a polynomial. Analyticity still belongs to
        the declared expression and assumptions; this is not a global proof.
        """
        if not isinstance(order, (int, np.integer)) or order < 0:
            raise ValueError('nonnegative integer Taylor order required')
        point = (sp.S.Zero,)*self.dim if point is None else tuple(map(sp.sympify, point))
        if (len(point) != self.dim or any(v.is_real is not True for v in point)
                or any(v.free_symbols & set(self.coordinates) for v in point)):
            raise ValueError('real expansion point independent of momentum required')
        scale = sp.Dummy('scale', positive=True)
        shifts = {k: v+scale*(k-v) for k,v in zip(self.coordinates, point)}
        def entry(e):
            try:
                series = e.subs(shifts, simultaneous=True).series(scale, 0, int(order)+1).removeO()
                polynomial = sp.Poly(series, scale)
                out = sp.expand(polynomial.as_expr().subs(scale, 1))
                sp.Poly(out, *self.coordinates)
            except (sp.PolynomialError, NotImplementedError, ValueError) as exc:
                raise ValueError('regular polynomial Taylor expansion unavailable') from exc
            return out
        return type(self)(self.matrix.applyfunc(entry), self.coordinates)

    def pauli_decomposition(self, *, max_qubits=4):
        """Exact trace projection onto Pauli tensor products (left factor first)."""
        q = self.norb.bit_length()-1
        if self.norb != 2**q or q < 1 or q > max_qubits:
            raise ValueError('Pauli decomposition requires 2**q orbitals, 1<=q<=max_qubits')
        matrices = (sp.eye(2), sp.Matrix([[0,1],[1,0]]),
                    sp.Matrix([[0,-sp.I],[sp.I,0]]), sp.diag(1,-1))
        coefficients = {}
        for indices in product(range(4), repeat=q):
            basis = sp.kronecker_product(*(matrices[i] for i in indices))
            c = sp.simplify(sp.trace(basis*self.matrix)/self.norb)
            if c != 0:
                coefficients[''.join('0xyz'[i] for i in indices)] = c
        return PauliDecomposition(coefficients, q)

    def to_hk(self, parameters=None):
        """Bind all parameters and explicitly lower a polynomial to numerical HK."""
        from .symbolic import HK
        bound = self.subs(parameters or {})
        if bound.parameters:
            raise ValueError(f'unbound symbolic parameters: {bound.parameters}')
        return HK.from_sympy(bound.matrix, bound.coordinates)

    def compile(self, parameters=None):
        """Bind parameters, CSE once, then evaluate with NumPy or nopython Numba.

        Smooth NumPy-printable scalar expressions are supported subject to the
        chosen backend. Unsupported Numba functions raise compilation errors;
        there is no silent object-mode fallback. No persistent compiled cache.
        """
        bound = self.subs(parameters or {})
        if bound.parameters:
            raise ValueError(f'unbound symbolic parameters: {bound.parameters}')
        return CompiledSymbolic(bound)


@dataclass(frozen=True)
class PauliDecomposition:
    """Sparse exact coefficients; labels like 'xz' mean sigma_x tensor sigma_z."""
    coefficients: object
    qubits: int

    def __post_init__(self):
        if not isinstance(self.qubits, int) or self.qubits < 1:
            raise ValueError('positive integer qubits required')
        coefficients = dict(self.coefficients)
        if any(not isinstance(label, str) or len(label) != self.qubits
               or set(label)-set('0xyz') for label in coefficients):
            raise ValueError('Pauli labels must contain one of 0xyz per qubit')
        coefficients = {label: sp.sympify(c) for label,c in coefficients.items()}
        if any(not isinstance(c, sp.Expr) or c.is_commutative is not True for c in coefficients.values()):
            raise ValueError('commutative scalar coefficients required')
        object.__setattr__(self, 'coefficients', MappingProxyType(coefficients))

    def reconstruct(self):
        matrices = {'0': sp.eye(2), 'x': sp.Matrix([[0,1],[1,0]]),
                    'y': sp.Matrix([[0,-sp.I],[sp.I,0]]), 'z': sp.diag(1,-1)}
        out = sp.zeros(2**self.qubits)
        for label,c in self.coefficients.items():
            out += c*sp.kronecker_product(*(matrices[i] for i in label))
        return sp.ImmutableMatrix(out)

    def latex(self):
        terms = []
        for label,c in self.coefficients.items():
            basis = r' \otimes '.join(r'\mathbb{1}' if i == '0' else r'\sigma_{'+i+'}' for i in label)
            terms.append(r'\left('+sp.latex(c)+r'\right)\,'+basis)
        return ' + '.join(terms) if terms else '0'

    def _repr_latex_(self):
        return '$\\displaystyle '+self.latex()+'$'


def _evaluators(expressions, coordinates):
    # A vector argument yields explicit indexing in generated scalar code. This
    # avoids dynamic *args and also handles constant/zero entries consistently.
    numpy_scalar = sp.lambdify([coordinates], tuple(expressions), modules='numpy',
                               cse=True, dummify=True, docstring_limit=0)
    scalar = njit(numpy_scalar)
    width = len(expressions)

    @njit(parallel=True)
    def batch(k):
        out = np.empty((len(k), width), dtype=np.complex128)
        for i in prange(len(k)):
            out[i] = np.asarray(scalar(k[i]), dtype=np.complex128)
        return out
    return numpy_scalar, batch


class CompiledSymbolic(TBkit):
    """Bound symbolic model; array API shared with numerical TBkit models.

    Cartesian variables and numerical energy units are inherited from the user.
    No lattice/periodicity metadata is invented for arbitrary expressions.
    """
    def __init__(self, symbolic):
        if not isinstance(symbolic, SymbolicHamiltonian):
            raise TypeError('SymbolicHamiltonian required')
        if symbolic.parameters:
            raise ValueError(f'unbound symbolic parameters: {symbolic.parameters}')
        self.symbolic = symbolic
        self.dim, self.norb = symbolic.dim, symbolic.norb
        self._h = _evaluators(list(symbolic.matrix), symbolic.coordinates)
        gradients = [e for k in symbolic.coordinates for e in symbolic.matrix.diff(k)]
        self._g = _evaluators(gradients, symbolic.coordinates)

    def _evaluate(self, k, backend, evaluators, shape):
        k, single = points(k, self.dim)
        if backend == 'numba':
            out = evaluators[1](k)
        elif backend == 'numpy':
            out = np.empty((len(k), int(np.prod(shape))), dtype=complex)
            for i, p in enumerate(k):
                out[i] = evaluators[0](p)
        else:
            raise ValueError('backend must be numba or numpy')
        out = out.reshape((len(k),)+shape)
        if not np.isfinite(out).all():
            raise ValueError('symbolic model is nonfinite at the requested momenta')
        return out[0] if single else out

    def hamiltonian(self, k, *, backend='numba'):
        return self._evaluate(k, backend, self._h, (self.norb,self.norb))

    def gradient(self, k, *, backend='numba'):
        return self._evaluate(k, backend, self._g, (self.dim,self.norb,self.norb))
