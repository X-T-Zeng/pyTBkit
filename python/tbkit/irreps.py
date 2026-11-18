"""Numerical ordinary irreps from the centre and commutant of the regular action.

This is a bounded dense small-group algorithm, not symbolic reconstruction,
antiunitary corepresentation theory or a Bloch/projective sewing generator.
"""
from math import isqrt
import numpy as np
from .groups import FiniteGroup,_integer
from .group_kernels import regular_blocks
from .representations import MatrixRepresentation,CharacterTable


class IrrepGenerationError(RuntimeError):
    """Numerical splitting or its independent representation checks failed."""


def _hermitian_parts(matrix):
    # These are defined Hermitian combinations, not repairs of an input model.
    yield (matrix+matrix.conj().T)/2
    yield (matrix-matrix.conj().T)/(2j)


def _spectral_spaces(basis,operator,spectral_tolerance):
    """Refine a subspace by a bounded Hermitian operator's compressed spectrum."""
    small=basis.conj().T@operator@basis
    if not np.allclose(small,small.conj().T,atol=1e-12,rtol=0):
        raise IrrepGenerationError('compressed splitting operator is not Hermitian')
    try:values,vectors=np.linalg.eigh(small)
    except np.linalg.LinAlgError as exc:
        raise IrrepGenerationError('Hermitian eigensolver did not converge') from exc
    if not np.isfinite(values).all() or not np.isfinite(vectors).all():
        raise IrrepGenerationError('nonfinite eigensolver output')
    cuts=[0]
    for i in range(1,len(values)):
        gap=values[i]-values[i-1]
        if gap>spectral_tolerance:
            if gap<10*spectral_tolerance:
                raise IrrepGenerationError('ambiguous spectral gap; decrease spectral_tolerance or use an exact method')
            cuts.append(i)
    cuts.append(len(values))
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        if values[hi-1]-values[lo]>spectral_tolerance:
            raise IrrepGenerationError('chained eigenvalue cluster exceeds spectral_tolerance')
    return [np.ascontiguousarray(basis@vectors[:,lo:hi]) for lo,hi in zip(cuts[:-1],cuts[1:])]


def _central_spaces(group,spectral_tolerance):
    order=group.order;classes=group.conjugacy_classes()
    spaces=[np.eye(order,dtype=complex)];columns=np.arange(order)
    for conjugacy_class in classes:
        # Class averages have norm <= 1; both Hermitian parts are necessary
        # when a class differs from its inverse (e.g. C3 complex characters).
        average=np.zeros((order,order),dtype=complex)
        for element in conjugacy_class:
            average[group.table[group.index(element)],columns]+=1/len(conjugacy_class)
        for operator in _hermitian_parts(average):
            spaces=[piece for basis in spaces for piece in _spectral_spaces(basis,operator,spectral_tolerance)]
            if len(spaces)>len(classes):
                raise IrrepGenerationError('spurious central splitting; spectral_tolerance is too small')
        if len(spaces)==len(classes):break
    if len(spaces)!=len(classes):
        raise IrrepGenerationError('centre not fully resolved; decrease spectral_tolerance')
    return spaces


def _one_copy(group,basis,degree,spectral_tolerance):
    # R(g)|h>=|h*g^-1> commutes with every L(a). Compressions of its Hermitian
    # parts still commute with L on any invariant subspace. Their linear span
    # is the full multiplicity algebra, so a reducible copy can be refined.
    columns=np.arange(group.order)
    while basis.shape[1]>degree:
        refined=False
        for element in group:
            inverse=group.index(group.inverse(element))
            right=np.zeros((group.order,group.order),dtype=complex)
            right[group.table[:,inverse],columns]=1
            for operator in _hermitian_parts(right):
                pieces=_spectral_spaces(basis,operator,spectral_tolerance)
                if len(pieces)==1:continue
                if any(piece.shape[1]%degree for piece in pieces):
                    raise IrrepGenerationError('commutant splitting violates irrep multiplicity')
                # Only one representative of an equivalence class is needed.
                basis=min(pieces,key=lambda piece:piece.shape[1]);refined=True;break
            if refined:break
        if not refined:
            raise IrrepGenerationError('multiplicity space not resolved within spectral_tolerance')
    return basis


def irreducible_representations(group,*,tolerance=1e-9,spectral_tolerance=1e-8,max_order=64):
    """Generate one numeric unitary irrep per ordinary complex equivalence class.

    Uses the left regular action's central isotypic spaces (dimension d**2),
    then its commuting right action to extract a d-dimensional copy. Both
    Hermitian parts are used, retaining complex and quaternionic irreps.

    Results are complex128 MatrixRepresentations, explicitly exact=False.
    Their bases/names are generic, not canonical Mulliken labels. No Float to
    exact SymPy guessing is performed. Every multiplication pair, unitarity,
    character orthogonality and sum d**2=|G| is checked before returning.

    max_order bounds the dense preparation. spectral_tolerance is absolute for
    norm<=1 splitting operators; 1..10 times that threshold is an ambiguity
    band. Unresolved/unstable splitting raises, never returns a partial list.
    tolerance controls final representation/character validation.
    """
    if not isinstance(group,FiniteGroup):raise TypeError('validated FiniteGroup required')
    if group.order>_integer(max_order,'max_order',1):raise ValueError('regular irrep generation exceeds max_order')
    for name,value in (('tolerance',tolerance),('spectral_tolerance',spectral_tolerance)):
        if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating))
                or not np.isfinite(value) or value<=0):
            raise ValueError(f'positive finite {name} required')
    candidates=[]
    for space in _central_spaces(group,float(spectral_tolerance)):
        degree=isqrt(space.shape[1])
        if degree**2!=space.shape[1]:
            raise IrrepGenerationError('central multiplicity is not a square; adjust spectral_tolerance')
        basis=_one_copy(group,space,degree,float(spectral_tolerance))
        if not np.allclose(basis.conj().T@basis,np.eye(degree),atol=tolerance,rtol=0):
            raise IrrepGenerationError('extracted irrep basis is not orthonormal')
        values=regular_blocks(group.table,basis)
        characters=np.trace(values,axis1=1,axis2=2)
        # Rounded keys order rows only; matrices and characters are not snapped.
        trivial=degree==1 and np.allclose(characters,1,atol=tolerance,rtol=0)
        key=(not trivial,degree,tuple((round(z.real,12),round(z.imag,12)) for z in characters))
        candidates.append((key,values))
    try:
        irreps=tuple(MatrixRepresentation(group,values,exact=False,tolerance=tolerance,name=f'rho_{i}')
                     for i,(_,values) in enumerate(sorted(candidates,key=lambda item:item[0])))
        CharacterTable(irreps)  # independent completeness/orthogonality checks
    except ValueError as exc:
        raise IrrepGenerationError('generated blocks failed representation/character validation') from exc
    return irreps
