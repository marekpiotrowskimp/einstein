"""
Tests for einstein.geometry.Manifold

Covers:
- Minkowski spacetime: flat, all curvature tensors zero
- Schwarzschild metric: known Christoffel symbols, non-zero Ricci scalar = 0 (vacuum)
- 2-sphere: known positive curvature
- Einstein field equations residual in vacuum
"""

import sympy as sp
import pytest

from einstein.geometry import Manifold


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minkowski():
    t, x, y, z = sp.symbols("t x y z", real=True)
    eta = sp.diag(-1, 1, 1, 1)
    return Manifold([t, x, y, z], eta, name="Minkowski")


@pytest.fixture
def schwarzschild():
    t = sp.Symbol("t", real=True)
    r = sp.Symbol("r", positive=True)
    theta = sp.Symbol("theta", positive=True)
    phi = sp.Symbol("phi", real=True)
    M, G, c = sp.symbols("M G c", positive=True)
    rs = 2 * G * M / c**2

    g = sp.diag(
        -(1 - rs / r),
        1 / (1 - rs / r),
        r**2,
        r**2 * sp.sin(theta) ** 2,
    )
    return Manifold([t, r, theta, phi], g, name="Schwarzschild")


@pytest.fixture
def sphere_2d():
    """Unit 2-sphere embedded via polar coordinates."""
    theta, phi = sp.symbols("theta phi", real=True)
    g = sp.Matrix([
        [1, 0],
        [0, sp.sin(theta) ** 2],
    ])
    return Manifold([theta, phi], g, name="S2")


# ---------------------------------------------------------------------------
# Minkowski
# ---------------------------------------------------------------------------

class TestMinkowski:
    def test_repr(self, minkowski):
        assert "Minkowski" in repr(minkowski)
        assert "dim=4" in repr(minkowski)

    def test_metric_inv(self, minkowski):
        eta_inv = minkowski.metric_inv
        assert eta_inv == sp.diag(-1, 1, 1, 1)

    def test_christoffel_all_zero(self, minkowski):
        G = minkowski.christoffel
        n = minkowski.n
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    assert sp.simplify(G[lam][mu][nu]) == 0, \
                        f"Γ^{lam}_{mu}{nu} should be zero in Minkowski"

    def test_riemann_all_zero(self, minkowski):
        assert minkowski.is_flat()

    def test_ricci_tensor_zero(self, minkowski):
        Ric = minkowski.ricci_tensor
        assert Ric == sp.zeros(4, 4)

    def test_ricci_scalar_zero(self, minkowski):
        assert minkowski.ricci_scalar == 0

    def test_einstein_tensor_zero(self, minkowski):
        G = minkowski.einstein_tensor
        assert G == sp.zeros(4, 4)

    def test_field_equations_vacuum(self, minkowski):
        residual = minkowski.field_equations()
        assert residual == sp.zeros(4, 4)

    def test_latex_metric(self, minkowski):
        latex = minkowski.latex_metric()
        assert isinstance(latex, str)
        assert len(latex) > 0


# ---------------------------------------------------------------------------
# 2-sphere — known curvature
# ---------------------------------------------------------------------------

class TestSphere2D:
    def test_not_flat(self, sphere_2d):
        assert not sphere_2d.is_flat()

    def test_ricci_scalar(self, sphere_2d):
        """For a unit 2-sphere, R = 2."""
        R = sphere_2d.ricci_scalar
        assert sp.simplify(R - 2) == 0, f"Expected R=2, got {R}"

    def test_einstein_tensor(self, sphere_2d):
        """For a 2D manifold, G_μν = R_μν - ½ g_μν R.
        On S², this equals zero since R_μν = (R/2) g_μν."""
        G = sphere_2d.einstein_tensor
        for i in range(2):
            for j in range(2):
                assert sp.simplify(G[i, j]) == 0, \
                    f"G[{i},{j}] should be 0 on S² (holds in 2D)"


# ---------------------------------------------------------------------------
# Schwarzschild — vacuum solution
# ---------------------------------------------------------------------------

class TestSchwarzschild:
    def test_not_flat(self, schwarzschild):
        assert not schwarzschild.is_flat()

    def test_ricci_tensor_vacuum(self, schwarzschild):
        """Schwarzschild is a vacuum solution: R_μν = 0."""
        Ric = schwarzschild.ricci_tensor
        n = schwarzschild.n
        for mu in range(n):
            for nu in range(n):
                val = sp.simplify(Ric[mu, nu])
                assert val == 0, \
                    f"R[{mu},{nu}] = {val} should be 0 (vacuum)"

    def test_ricci_scalar_vacuum(self, schwarzschild):
        """Vacuum: R = 0."""
        assert sp.simplify(schwarzschild.ricci_scalar) == 0

    def test_einstein_tensor_vacuum(self, schwarzschild):
        """Vacuum: G_μν = 0."""
        G = schwarzschild.einstein_tensor
        n = schwarzschild.n
        for mu in range(n):
            for nu in range(n):
                assert sp.simplify(G[mu, nu]) == 0, \
                    f"G[{mu},{nu}] should be 0 in Schwarzschild vacuum"

    def test_field_equations_vacuum(self, schwarzschild):
        residual = schwarzschild.field_equations()
        n = schwarzschild.n
        for mu in range(n):
            for nu in range(n):
                assert sp.simplify(residual[mu, nu]) == 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_wrong_metric_shape(self):
        x, y = sp.symbols("x y")
        with pytest.raises(ValueError, match="shape"):
            Manifold([x, y], sp.eye(3))
