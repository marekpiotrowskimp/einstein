"""
Differential geometry core — symbolic tensor calculus.

Provides:
- Manifold: coordinate system and metric tensor
- Christoffel symbols (Γ^λ_μν)
- Riemann curvature tensor (R^ρ_σμν)
- Ricci tensor (R_μν)
- Ricci scalar (R)
- Einstein tensor (G_μν = R_μν - ½ g_μν R)
- Stress-energy tensor scaffold (T_μν)
- Einstein field equations: G_μν + Λ g_μν = 8πG/c⁴ · T_μν

All computations are symbolic via SymPy, enabling:
- Exact analytic results
- Simplification and substitution
- Export to LaTeX
"""

from __future__ import annotations

from typing import Sequence

import sympy as sp


class Manifold:
    """
    A pseudo-Riemannian manifold defined by coordinates and a metric tensor.

    Parameters
    ----------
    coords : sequence of sympy.Symbol
        Coordinate symbols, e.g. [t, r, theta, phi].
    metric : sympy.Matrix
        Covariant metric tensor g_μν (n×n symmetric matrix).
    name : str
        Human-readable label for the manifold.

    Examples
    --------
    Minkowski spacetime (flat, signature -+++) ::

        import sympy as sp
        from einstein.geometry import Manifold

        t, x, y, z = sp.symbols('t x y z', real=True)
        eta = sp.diag(-1, 1, 1, 1)
        M = Manifold([t, x, y, z], eta, name='Minkowski')

    Schwarzschild metric ::

        r, theta = sp.symbols('r theta', positive=True)
        phi = sp.Symbol('phi', real=True)
        t = sp.Symbol('t', real=True)
        M_mass, G, c = sp.symbols('M G c', positive=True)
        rs = 2 * G * M_mass / c**2

        g = sp.diag(
            -(1 - rs/r),
            1 / (1 - rs/r),
            r**2,
            r**2 * sp.sin(theta)**2,
        )
        S = Manifold([t, r, theta, phi], g, name='Schwarzschild')
    """

    def __init__(
        self,
        coords: Sequence[sp.Symbol],
        metric: sp.Matrix,
        name: str = "Manifold",
    ) -> None:
        self.coords = list(coords)
        self.n = len(self.coords)
        self.metric = sp.Matrix(metric)
        self.name = name

        if self.metric.shape != (self.n, self.n):
            raise ValueError(
                f"Metric shape {self.metric.shape} does not match "
                f"number of coordinates {self.n}."
            )

        # Cached computations (lazy)
        self._metric_inv: sp.Matrix | None = None
        self._christoffel: list | None = None
        self._riemann: list | None = None
        self._ricci_tensor: sp.Matrix | None = None
        self._ricci_scalar: sp.Expr | None = None
        self._einstein_tensor: sp.Matrix | None = None

    # ------------------------------------------------------------------
    # Metric inverse
    # ------------------------------------------------------------------

    @property
    def metric_inv(self) -> sp.Matrix:
        """Contravariant metric g^μν (inverse of g_μν)."""
        if self._metric_inv is None:
            self._metric_inv = self.metric.inv()
        return self._metric_inv

    # ------------------------------------------------------------------
    # Christoffel symbols  Γ^λ_μν
    # ------------------------------------------------------------------

    @property
    def christoffel(self) -> list[list[list[sp.Expr]]]:
        """
        Christoffel symbols of the second kind Γ^λ_μν.

        Returns
        -------
        list[list[list[sp.Expr]]]
            Indexed as ``christoffel[lam][mu][nu]``.

        Formula
        -------
        Γ^λ_μν = ½ g^λσ (∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)
        """
        if self._christoffel is not None:
            return self._christoffel

        n = self.n
        g = self.metric
        gi = self.metric_inv
        x = self.coords

        gamma = [
            [
                [sp.Integer(0) for _ in range(n)]
                for _ in range(n)
            ]
            for _ in range(n)
        ]

        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    s = sp.Integer(0)
                    for sigma in range(n):
                        s += gi[lam, sigma] * (
                            sp.diff(g[nu, sigma], x[mu])
                            + sp.diff(g[mu, sigma], x[nu])
                            - sp.diff(g[mu, nu], x[sigma])
                        )
                    gamma[lam][mu][nu] = sp.Rational(1, 2) * s

        self._christoffel = gamma
        return gamma

    # ------------------------------------------------------------------
    # Riemann curvature tensor  R^ρ_σμν
    # ------------------------------------------------------------------

    @property
    def riemann(self) -> list[list[list[list[sp.Expr]]]]:
        """
        Riemann curvature tensor R^ρ_σμν.

        Returns
        -------
        list indexed as ``riemann[rho][sigma][mu][nu]``.

        Formula
        -------
        R^ρ_σμν = ∂_μ Γ^ρ_νσ - ∂_ν Γ^ρ_μσ
                  + Γ^ρ_μλ Γ^λ_νσ - Γ^ρ_νλ Γ^λ_μσ
        """
        if self._riemann is not None:
            return self._riemann

        n = self.n
        x = self.coords
        G = self.christoffel

        R = [
            [
                [
                    [sp.Integer(0) for _ in range(n)]
                    for _ in range(n)
                ]
                for _ in range(n)
            ]
            for _ in range(n)
        ]

        for rho in range(n):
            for sigma in range(n):
                for mu in range(n):
                    for nu in range(n):
                        term = (
                            sp.diff(G[rho][nu][sigma], x[mu])
                            - sp.diff(G[rho][mu][sigma], x[nu])
                        )
                        for lam in range(n):
                            term += (
                                G[rho][mu][lam] * G[lam][nu][sigma]
                                - G[rho][nu][lam] * G[lam][mu][sigma]
                            )
                        R[rho][sigma][mu][nu] = term

        self._riemann = R
        return R

    # ------------------------------------------------------------------
    # Ricci tensor  R_μν
    # ------------------------------------------------------------------

    @property
    def ricci_tensor(self) -> sp.Matrix:
        """
        Ricci tensor R_μν = R^ρ_μρν (contraction of Riemann).

        Returns
        -------
        sympy.Matrix of shape (n, n).
        """
        if self._ricci_tensor is not None:
            return self._ricci_tensor

        n = self.n
        R = self.riemann
        Ric = sp.zeros(n, n)

        for mu in range(n):
            for nu in range(n):
                s = sp.Integer(0)
                for rho in range(n):
                    s += R[rho][mu][rho][nu]
                Ric[mu, nu] = sp.simplify(s)

        self._ricci_tensor = Ric
        return Ric

    # ------------------------------------------------------------------
    # Ricci scalar  R
    # ------------------------------------------------------------------

    @property
    def ricci_scalar(self) -> sp.Expr:
        """
        Ricci scalar R = g^μν R_μν.

        Returns
        -------
        sympy.Expr
        """
        if self._ricci_scalar is not None:
            return self._ricci_scalar

        n = self.n
        gi = self.metric_inv
        Ric = self.ricci_tensor
        s = sp.Integer(0)

        for mu in range(n):
            for nu in range(n):
                s += gi[mu, nu] * Ric[mu, nu]

        self._ricci_scalar = sp.simplify(s)
        return self._ricci_scalar

    # ------------------------------------------------------------------
    # Einstein tensor  G_μν
    # ------------------------------------------------------------------

    @property
    def einstein_tensor(self) -> sp.Matrix:
        """
        Einstein tensor G_μν = R_μν - ½ g_μν R.

        Returns
        -------
        sympy.Matrix of shape (n, n).

        Notes
        -----
        The Einstein field equations relate this tensor to the
        stress-energy content of spacetime:

            G_μν + Λ g_μν = (8πG/c⁴) T_μν
        """
        if self._einstein_tensor is not None:
            return self._einstein_tensor

        n = self.n
        g = self.metric
        Ric = self.ricci_tensor
        R = self.ricci_scalar
        G = sp.zeros(n, n)

        for mu in range(n):
            for nu in range(n):
                G[mu, nu] = sp.simplify(
                    Ric[mu, nu] - sp.Rational(1, 2) * g[mu, nu] * R
                )

        self._einstein_tensor = G
        return G

    # ------------------------------------------------------------------
    # Einstein field equations check
    # ------------------------------------------------------------------

    def field_equations(
        self,
        T: sp.Matrix | None = None,
        Lambda: sp.Expr = sp.Integer(0),
        G_newton: sp.Expr | None = None,
        c: sp.Expr | None = None,
    ) -> sp.Matrix:
        """
        Return the residual of Einstein field equations:

            G_μν + Λ g_μν - (8πG/c⁴) T_μν = 0

        Parameters
        ----------
        T : sympy.Matrix, optional
            Stress-energy tensor T_μν. Defaults to zero (vacuum).
        Lambda : sympy.Expr
            Cosmological constant Λ. Defaults to 0.
        G_newton : sympy.Expr, optional
            Newton's gravitational constant. Defaults to symbol ``G``.
        c : sympy.Expr, optional
            Speed of light. Defaults to symbol ``c``.

        Returns
        -------
        sympy.Matrix
            Residual matrix. All entries should be zero for a valid solution.
        """
        n = self.n
        g = self.metric

        if T is None:
            T = sp.zeros(n, n)
        if G_newton is None:
            G_newton = sp.Symbol("G", positive=True)
        if c is None:
            c = sp.Symbol("c", positive=True)

        kappa = 8 * sp.pi * G_newton / c**4
        G_tensor = self.einstein_tensor
        residual = sp.zeros(n, n)

        for mu in range(n):
            for nu in range(n):
                residual[mu, nu] = sp.simplify(
                    G_tensor[mu, nu]
                    + Lambda * g[mu, nu]
                    - kappa * T[mu, nu]
                )

        return residual

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def latex_metric(self) -> str:
        """Return LaTeX representation of the metric tensor."""
        return sp.latex(self.metric)

    def latex_christoffel(self) -> str:
        """Return LaTeX representation of all nonzero Christoffel symbols."""
        lines = []
        n = self.n
        G = self.christoffel
        x = self.coords

        for lam in range(n):
            for mu in range(n):
                for nu in range(nu, n):  # use symmetry mu↔nu
                    val = sp.simplify(G[lam][mu][nu])
                    if val != 0:
                        lines.append(
                            rf"\Gamma^{{{sp.latex(x[lam])}}}_{{{sp.latex(x[mu])}{sp.latex(x[nu])}}} = {sp.latex(val)}"
                        )
        return "\n".join(lines)

    def is_flat(self) -> bool:
        """
        Return True if the manifold is flat (all Riemann components vanish).
        """
        n = self.n
        R = self.riemann
        for rho in range(n):
            for sigma in range(n):
                for mu in range(n):
                    for nu in range(n):
                        if sp.simplify(R[rho][sigma][mu][nu]) != 0:
                            return False
        return True

    def __repr__(self) -> str:
        return (
            f"Manifold(name={self.name!r}, dim={self.n}, "
            f"coords={self.coords})"
        )
