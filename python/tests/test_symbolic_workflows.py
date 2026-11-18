"""Independent analytic physics and exact/presentation/numerical boundary checks."""
from dataclasses import FrozenInstanceError
import numpy as np
import pytest
import sympy as sp

from tbkit import SymbolicHamiltonian, HK, Htrig, Oper

x,y = sp.symbols('k_x k_y', real=True)
m,v,b = sp.symbols('m v b', real=True)
sx=sp.Matrix([[0,1],[1,0]])
sy=sp.Matrix([[0,-sp.I],[sp.I,0]])
sz=sp.diag(1,-1)


def dirac():
    return SymbolicHamiltonian(v*x*sx+v*y*sy+(m-b*(x*x+y*y))*sz, (x,y))


def test_exact_parameters_algebra_and_immutable_source():
    source=sp.Matrix([[sp.Rational(1,3)*m,sp.sqrt(2)*x],[sp.sqrt(2)*x,-sp.Rational(1,3)*m]])
    h=SymbolicHamiltonian(source,(x,))
    source[0,0]=7
    assert h.matrix[0,0]==m/3
    assert h.parameters==(m,)
    assert not h.matrix.atoms(sp.Float)
    assert (h+h-h).matrix==h.matrix
    assert sum([h,-h]).matrix==sp.zeros(2)
    assert h.dagger().matrix==h.matrix
    with pytest.raises(TypeError):h.matrix[0,0]=2
    with pytest.raises(FrozenInstanceError):h.matrix=sp.eye(2)
    with pytest.raises(TypeError):h+1  # no ambiguous elementwise vs identity shift
    with pytest.raises(ValueError):h+SymbolicHamiltonian(sp.eye(2),(y,))


def test_substitution_momentum_swap_and_assumptions():
    h=SymbolicHamiltonian(sp.diag(x+2*y,m),(x,y))
    swap=h.subs({x:y,y:x})
    assert swap.matrix==sp.diag(y+2*x,m)
    assert h.matrix==sp.diag(x+2*y,m)
    assert h.subs({m:v}).parameters==(v,)
    assert h.subs({m:sp.Rational(1,7)}).matrix[1,1]==sp.Rational(1,7)
    z=sp.Symbol('z')
    assert SymbolicHamiltonian(sp.diag(z,0),(x,)).is_hermitian() is None
    assert SymbolicHamiltonian(sp.diag(sp.I,0),(x,)).is_hermitian() is False
    assert dirac().is_hermitian() is True
    with pytest.raises(ValueError):SymbolicHamiltonian(sp.eye(2),(z,))


def test_polynomial_coefficients_derivatives_and_lowering():
    h=dirac()
    assert h.polynomial_terms()[(1,0)]==v*sx
    assert h.polynomial_terms()[(0,2)]==-b*sz
    assert h.derivative(0).matrix==v*sx-2*b*x*sz
    assert h.derivative(1,2).matrix==-2*b*sz
    assert h.derivative(0,0).matrix==h.matrix
    with pytest.raises(TypeError):h.polynomial_terms()[(1,0)]=sp.eye(2)
    with pytest.raises(ValueError):h.to_hk({m:1})
    numeric=h.to_hk({m:.7,b:.2,v:1.3})
    k=np.array([[.2,.4],[-.3,.1]])
    # Analytic two-band energy, independent of symbolic code generation.
    energy=np.sqrt(1.3**2*(k*k).sum(1)+(.7-.2*(k*k).sum(1))**2)
    np.testing.assert_allclose(numeric.bands(k),np.column_stack([-energy,energy]),atol=1e-13)
    np.testing.assert_allclose(numeric.to_symbolic((x,y)).compile().hamiltonian(k),numeric.hamiltonian(k),atol=1e-14)


def test_total_order_taylor_and_nonzero_center():
    h=SymbolicHamiltonian(sp.diag(sp.exp(x+y),sp.sin(x)*sp.cos(y)),(x,y))
    t=h.taylor(2)
    expected=sp.diag(1+x+y+x*x/2+x*y+y*y/2,x)
    assert t.matrix==expected
    assert max(sum(p) for p in t.polynomial_terms())==2
    shifted=SymbolicHamiltonian(sp.Matrix([[sp.sin(x)]]),(x,)).taylor(3,point=[sp.pi/2])
    assert sp.expand(shifted.matrix[0]-(1-(x-sp.pi/2)**2/2))==0
    for expr in (1/x,sp.sqrt(x),sp.Abs(x)):
        with pytest.raises(ValueError):SymbolicHamiltonian([[expr]],(x,)).taylor(2)
    with pytest.raises(ValueError):h.taylor(2,point=[x,0])


def test_symmetry_exact_actions_time_reversal_and_particle_hole():
    # Spin-1/2 TR: T=i sigma_y K, k->-k. A mass term breaks TR.
    linear=SymbolicHamiltonian(v*(x*sx+y*sy),(x,y))
    tr=Oper(sp.I*sy,-sp.eye(2),antiunitary=True)
    assert linear.symmetry_residual(tr)==sp.zeros(2)
    assert dirac().symmetry_residual(tr)==2*(m-b*(x*x+y*y))*sz
    # Kitaev continuum, C=sigma_x K; C H(-k) C^-1=-H(k).
    kitaev=SymbolicHamiltonian(v*x*sy+(m+b*x*x)*sz,(x,))
    ph=Oper(sx,sp.Matrix([[-1]]),antiunitary=True,antisymmetry=True)
    assert kitaev.symmetry_residual(ph)==sp.zeros(2)
    c4=Oper((sp.eye(2)-sp.I*sz)/sp.sqrt(2),sp.Matrix([[0,1],[-1,0]]))
    assert dirac().symmetry_residual(c4)==sp.zeros(2)


def test_pauli_tensor_decomposition_and_formula_exports():
    h=dirac()
    d=h.pauli_decomposition()
    assert d.coefficients=={'x':v*x,'y':v*y,'z':m-b*(x*x+y*y)}
    assert (d.reconstruct()-h.matrix).applyfunc(sp.simplify)==sp.zeros(2)
    # Nontrivial tensor order: left orbital, right spin.
    matrix=sp.Rational(2,3)*sp.kronecker_product(sx,sz)+m*sp.kronecker_product(sp.eye(2),sy)
    d4=SymbolicHamiltonian(matrix,(x,)).pauli_decomposition()
    assert d4.coefficients=={'0y':m,'xz':sp.Rational(2,3)}
    assert d4.reconstruct()==matrix
    assert r'\otimes' in d4.latex()
    assert r'\sigma_{x}' in d.latex()
    assert r'\begin{matrix}' in h.latex()
    assert h._repr_latex_().startswith('$') and h._repr_latex_().endswith('$')
    assert 'k_x' in h.pretty(use_unicode=False)
    assert not h.matrix.atoms(sp.Float)  # presentation must preserve exact data
    with pytest.raises(ValueError):SymbolicHamiltonian(sp.eye(3),(x,)).pauli_decomposition()
    with pytest.raises(TypeError):d.coefficients['x']=0


@pytest.mark.parametrize('backend',['numpy','numba'])
def test_trigonometric_compile_and_analytic_gradient(backend):
    # QWZ exact matrix, no polynomial approximation.
    h=SymbolicHamiltonian(v*sp.sin(x)*sx+v*sp.sin(y)*sy+(m+sp.cos(x)+sp.cos(y))*sz,(x,y))
    with pytest.raises(ValueError):h.compile({v:1.2})
    numeric=h.compile({m:-.7,v:1.2})
    k=np.array([[.23,-.41],[.7,.9]])
    dx,dy,dz=1.2*np.sin(k[:,0]),1.2*np.sin(k[:,1]),-.7+np.cos(k[:,0])+np.cos(k[:,1])
    reference=np.zeros((len(k),2,2),complex)
    reference[:,0,0]=dz;reference[:,1,1]=-dz
    reference[:,0,1]=dx-1j*dy;reference[:,1,0]=dx+1j*dy
    np.testing.assert_allclose(numeric.hamiltonian(k,backend=backend),reference,atol=1e-14)
    derivative=np.stack([1.2*np.cos(k[:,0])[:,None,None]*np.array(sx,complex)-np.sin(k[:,0])[:,None,None]*np.array(sz,complex),
                         1.2*np.cos(k[:,1])[:,None,None]*np.array(sy,complex)-np.sin(k[:,1])[:,None,None]*np.array(sz,complex)],axis=1)
    np.testing.assert_allclose(numeric.gradient(k,backend=backend),derivative,atol=1e-14)
    energy=np.sqrt(dx*dx+dy*dy+dz*dz)
    np.testing.assert_allclose(numeric.bands(k,backend=backend),np.column_stack([-energy,energy]),atol=1e-14)
    assert numeric.hamiltonian(k[0],backend=backend).shape==(2,2)
    assert numeric.gradient(k[0],backend=backend).shape==(2,2,2)
    assert numeric.hamiltonian(np.empty((0,2)),backend=backend).shape==(0,2,2)
    if backend=='numba':
        assert numeric._h[1].nopython_signatures
        assert numeric._g[1].nopython_signatures
    with pytest.raises(ValueError):numeric.hamiltonian(k,backend='invalid')
    with pytest.raises(ValueError):numeric.hamiltonian([1j,0],backend=backend)


@pytest.mark.parametrize('backend',['numpy','numba'])
def test_constant_complex_and_zero_compiled_entries(backend):
    h=SymbolicHamiltonian([[1,sp.I],[-sp.I,2]],(x,)).compile()
    k=np.array([[0.],[.2]])
    np.testing.assert_allclose(h.hamiltonian(k,backend=backend),np.broadcast_to([[1,1j],[-1j,2]],(2,2,2)))
    np.testing.assert_allclose(h.gradient(k,backend=backend),0)


def test_numerical_trig_view_and_explicit_float_policy():
    h=Htrig([[1,0],[0,1]],[np.array(sz,complex),np.array(sz,complex)],
            [np.array(sx,complex),np.array(sy,complex)])
    view=h.to_symbolic((x,y))
    assert view.matrix.atoms(sp.Float)
    k=np.array([[.3,.6],[-.4,.2]])
    np.testing.assert_allclose(view.compile().hamiltonian(k),h.hamiltonian(k),atol=1e-14)
    expanded=SymbolicHamiltonian(sp.eye(2)*(sp.sin(x)**2+sp.cos(x)**2),(x,)).simplify()
    assert expanded.matrix==sp.eye(2)
    assert dirac().collect().matrix.expand()==dirac().matrix.expand()
    exponential=SymbolicHamiltonian(sp.eye(2)*sp.cos(x),(x,)).rewrite(sp.exp)
    assert exponential.matrix[0,0].has(sp.exp)


@pytest.mark.parametrize('matrix,coordinates',[([], (x,)), ([[sp.oo]],(x,)), ([[1]],()), ([[1]],(x,x)), ([[1,2]],(x,))])
def test_invalid_symbolic_inputs(matrix,coordinates):
    with pytest.raises(ValueError):SymbolicHamiltonian(matrix,coordinates)


def test_exact_invariants_retain_symbols_and_present_before_binding():
    from tbkit import invariant_kp
    c3=Oper((sp.eye(2)-sp.I*sp.sqrt(3)*sz)/2,
            sp.Matrix([[-sp.Rational(1,2),sp.sqrt(3)/2],[-sp.sqrt(3)/2,-sp.Rational(1,2)]]))
    basis=invariant_kp(2,2,1,[c3],output='symbolic',coordinates=(x,y))
    assert len(basis)==4
    assert all(not h.matrix.atoms(sp.Float) for h in basis)
    assert all(h.symmetry_residual(c3)==sp.zeros(2) and h.is_hermitian() is True for h in basis)
    coefficients=sp.symbols(f'a0:{len(basis)}',real=True)
    model=sum(a*h for a,h in zip(coefficients,basis))
    assert set(model.parameters)==set(coefficients)
    assert model.symmetry_residual(c3)==sp.zeros(2)
    assert model.latex()
    bound=model.compile(dict(zip(coefficients,[.2,.4,.7,1.1])))
    k=np.array([[.1,.3],[-.2,.4]])
    assert c3.residual(bound,k)<1e-13
    with pytest.raises(ValueError):invariant_kp(2,2,1,[c3],method='numeric',output='symbolic')


def test_symbol_names_do_not_shadow_numeric_functions():
    sin=sp.Symbol('sin',real=True)
    h=SymbolicHamiltonian([[sp.sin(sin)]],(sin,)).compile()
    np.testing.assert_allclose(h.hamiltonian([.3]),[[np.sin(.3)]])


def test_basis_and_operation_rich_presentation_preserves_exact_data():
    from tbkit import Qnum,BasisFunc
    up=Qnum(1,0,sp.Rational(1,2),sp.Rational(1,2),2)
    down=Qnum(1,1,sp.Rational(1,2),-sp.Rational(1,2),2)
    state=BasisFunc({up:m/sp.sqrt(3),down:sp.I*sp.sqrt(2)/sp.sqrt(3)})
    tex=state.latex()
    assert r'\sqrt{3}' in tex and r'\right\rangle' in tex
    assert 's_z=' in up.latex()
    assert state.subs({m:1}).inner(state.subs({m:1}))==1
    assert state.terms[up]==m/sp.sqrt(3)
    tr=Oper(sp.I*sy,-sp.eye(2),antiunitary=True)
    assert r'^{*}' in tr.latex() and r'U^{\dagger}' in tr.latex()
    assert '1.0' not in tr.latex()
    assert tr._repr_latex_().startswith('$')


def test_internal_constraint_names_do_not_collide_with_momenta():
    from tbkit import invariant_kp,CompiledSymbolic,berry_curvature
    c0,c1=sp.symbols('c0 c1',real=True)
    c4=Oper((sp.eye(2)-sp.I*sz)/sp.sqrt(2),sp.Matrix([[0,1],[-1,0]]))
    basis=invariant_kp(2,2,1,[c4],output='symbolic',coordinates=(c0,c1))
    assert len(basis)==4
    assert all(h.symmetry_residual(c4)==sp.zeros(2) for h in basis)
    with pytest.raises(ValueError):CompiledSymbolic(dirac())
    # The generated gradient works in the existing spectral/geometry pipeline.
    model=SymbolicHamiltonian(x*sx+y*sy+m*sz,(x,y)).compile({m:.4})
    k=np.array([[.1,.2],[.3,-.4]])
    curvature=.4/(2*(.4**2+(k*k).sum(axis=1))**1.5)
    np.testing.assert_allclose(berry_curvature(model,k),np.column_stack([curvature,-curvature]),atol=1e-12)
