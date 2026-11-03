"""
einstein.simulation.scalar_field
=================================
Numerical evolution of a real scalar field φ in 1+1D and 2+1D spacetimes.

Physics implemented
-------------------

1. Flat spacetime — Klein-Gordon equation
   □φ - m²φ = 0
   ∂²φ/∂t² = c²∇²φ - m²c⁴/ℏ² φ

   Discretised with leapfrog (Störmer-Verlet) — symplectic, energy-conserving.

2. Curved background — generalised KG on a static metric g_μν
   (1/√|g|) ∂_μ (√|g| g^μν ∂_ν φ) - m² φ = 0

   For a diagonal static metric: ds² = -A(r)dt² + B(r)dr² + ...
   The spatial Laplacian is replaced by the Laplace-Beltrami operator.

3. Nonlinear φ⁴ theory (soliton / kink dynamics)
   ∂²φ/∂t² = ∇²φ - dV/dφ
   V(φ) = λ/4 (φ² - v²)²
   dV/dφ = λ φ (φ² - v²)

   Supports stable kink solutions that model localised excitations (H3).

4. Gaussian wave packet evolution
   Initial condition: φ = A exp(-r²/σ²) cos(k·r)
   Demonstrates dispersion and propagation in flat / curved backgrounds.

Numerical methods
-----------------
- Leapfrog (Verlet) integration: 2nd-order, symplectic, O(dt²)
- Finite differences: 2nd-order central differences for ∇²
- Optional: RK4 for comparison
- Periodic or absorbing boundary conditions

Output
------
Each simulation returns a SimulationResult dataclass containing:
- phi[t, x, (y)]  — field history
- pi[t, x, (y)]   — conjugate momentum π = ∂φ/∂t history
- energy[t]       — total energy at each timestep
- times[]         — time axis
- metadata        — dict with all simulation parameters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """
    Container for scalar field simulation output.

    Attributes
    ----------
    phi : np.ndarray
        Field values. Shape (n_steps+1, n_x) for 1D or (n_steps+1, n_x, n_y) for 2D.
    pi : np.ndarray
        Conjugate momentum ∂φ/∂t. Same shape as phi.
    energy : np.ndarray
        Total energy at each saved timestep. Shape (n_steps+1,).
    times : np.ndarray
        Time values for each saved frame. Shape (n_steps+1,).
    metadata : dict
        Simulation parameters (dx, dt, mass, etc.).
    """
    phi: np.ndarray
    pi: np.ndarray
    energy: np.ndarray
    times: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def n_dims(self) -> int:
        return self.phi.ndim - 1  # subtract time axis

    @property
    def n_steps(self) -> int:
        return len(self.times) - 1

    def energy_conservation_error(self) -> float:
        """Max relative deviation of energy from initial value."""
        e0 = self.energy[0]
        if e0 == 0:
            return float(np.max(np.abs(self.energy)))
        return float(np.max(np.abs((self.energy - e0) / e0)))


# ---------------------------------------------------------------------------
# 1D Klein-Gordon on flat spacetime (leapfrog)
# ---------------------------------------------------------------------------

def simulate_kg_1d(
    n_x: int = 200,
    x_max: float = 10.0,
    t_max: float = 20.0,
    dt: float = 0.01,
    mass: float = 0.0,
    c: float = 1.0,
    initial_phi: Callable[[np.ndarray], np.ndarray] | None = None,
    initial_pi: Callable[[np.ndarray], np.ndarray] | None = None,
    boundary: str = "periodic",
    save_every: int = 10,
) -> SimulationResult:
    """
    Evolve the Klein-Gordon field in 1+1D flat spacetime.

    Equation of motion:
        ∂²φ/∂t² = c² ∂²φ/∂x² - m² φ

    Numerical scheme: leapfrog (Störmer-Verlet), 2nd-order symplectic.

    Parameters
    ----------
    n_x : int
        Number of spatial grid points.
    x_max : float
        Half-domain size: x ∈ [-x_max, x_max].
    t_max : float
        Total simulation time.
    dt : float
        Timestep. Must satisfy CFL: c*dt/dx < 1.
    mass : float
        Field mass m (in natural units ℏ=1). m=0 → massless wave.
    c : float
        Wave speed (speed of light). Default = 1.
    initial_phi : callable, optional
        Function x -> φ(x, t=0). Defaults to a Gaussian wave packet.
    initial_pi : callable, optional
        Function x -> π(x, t=0) = ∂φ/∂t(x,0). Defaults to zero.
    boundary : str
        'periodic' or 'absorbing' (Dirichlet zero).
    save_every : int
        Save state every N timesteps.

    Returns
    -------
    SimulationResult

    Raises
    ------
    ValueError
        If CFL condition is violated (c*dt/dx ≥ 1).

    Examples
    --------
    Massless wave packet::

        result = simulate_kg_1d(mass=0.0, t_max=15.0)

    Standing wave (massive field)::

        result = simulate_kg_1d(mass=1.0, t_max=10.0)
    """
    dx = 2.0 * x_max / n_x
    x = np.linspace(-x_max, x_max, n_x, endpoint=False)

    cfl = c * dt / dx
    if cfl >= 1.0:
        raise ValueError(
            f"CFL condition violated: c*dt/dx = {cfl:.3f} ≥ 1. "
            f"Reduce dt or increase n_x."
        )

    # Initial conditions
    if initial_phi is None:
        def initial_phi(xarr: np.ndarray) -> np.ndarray:
            return np.exp(-xarr**2 / 1.0) * np.cos(2.0 * np.pi * xarr / 3.0)

    if initial_pi is None:
        def initial_pi(xarr: np.ndarray) -> np.ndarray:
            return np.zeros_like(xarr)

    phi = initial_phi(x).astype(float)
    pi = initial_pi(x).astype(float)

    n_steps = int(t_max / dt)
    n_saved = n_steps // save_every + 1

    phi_hist = np.zeros((n_saved, n_x))
    pi_hist = np.zeros((n_saved, n_x))
    energy_hist = np.zeros(n_saved)
    times = np.zeros(n_saved)

    def laplacian_1d(f: np.ndarray) -> np.ndarray:
        if boundary == "periodic":
            return (np.roll(f, -1) - 2 * f + np.roll(f, 1)) / dx**2
        else:  # absorbing
            lap = np.zeros_like(f)
            lap[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / dx**2
            return lap

    def compute_energy(phi_: np.ndarray, pi_: np.ndarray) -> float:
        ke = 0.5 * np.sum(pi_**2)
        # Central differences (consistent with leapfrog spatial discretization)
        grad = np.zeros_like(phi_)
        grad[1:-1] = (phi_[2:] - phi_[:-2]) / (2 * dx)
        grad[0] = (phi_[1] - phi_[-1]) / (2 * dx)   # periodic wrap
        grad[-1] = (phi_[0] - phi_[-2]) / (2 * dx)
        pe = 0.5 * c**2 * np.sum(grad**2)
        me = 0.5 * mass**2 * np.sum(phi_**2)
        return float((ke + pe + me) * dx)

    # Save initial state
    phi_hist[0] = phi.copy()
    pi_hist[0] = pi.copy()
    energy_hist[0] = compute_energy(phi, pi)
    times[0] = 0.0

    save_idx = 1

    # Leapfrog integration
    # Half-step for pi
    acc = c**2 * laplacian_1d(phi) - mass**2 * phi
    pi_half = pi + 0.5 * dt * acc

    for step in range(1, n_steps + 1):
        # Full step for phi
        phi = phi + dt * pi_half

        # Boundary conditions
        if boundary == "absorbing":
            phi[0] = 0.0
            phi[-1] = 0.0

        # Full step for pi
        acc = c**2 * laplacian_1d(phi) - mass**2 * phi
        pi_half_new = pi_half + dt * acc

        # Recover pi at integer step for energy
        pi = pi_half + 0.5 * dt * acc

        if step % save_every == 0 and save_idx < n_saved:
            phi_hist[save_idx] = phi.copy()
            pi_hist[save_idx] = pi.copy()
            energy_hist[save_idx] = compute_energy(phi, pi)
            times[save_idx] = step * dt
            save_idx += 1

        pi_half = pi_half_new

    return SimulationResult(
        phi=phi_hist[:save_idx],
        pi=pi_hist[:save_idx],
        energy=energy_hist[:save_idx],
        times=times[:save_idx],
        metadata={
            "n_x": n_x, "x_max": x_max, "dx": dx,
            "t_max": t_max, "dt": dt, "mass": mass,
            "c": c, "boundary": boundary,
            "cfl": cfl, "equation": "Klein-Gordon 1D",
        },
    )


# ---------------------------------------------------------------------------
# 2D Klein-Gordon on flat spacetime
# ---------------------------------------------------------------------------

def simulate_kg_2d(
    n_x: int = 100,
    n_y: int = 100,
    x_max: float = 8.0,
    t_max: float = 10.0,
    dt: float = 0.02,
    mass: float = 0.0,
    c: float = 1.0,
    initial_phi: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    initial_pi: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    boundary: str = "periodic",
    save_every: int = 5,
) -> SimulationResult:
    """
    Evolve the Klein-Gordon field in 2+1D flat spacetime.

    Equation of motion:
        ∂²φ/∂t² = c² (∂²φ/∂x² + ∂²φ/∂y²) - m² φ

    Parameters
    ----------
    n_x, n_y : int
        Grid dimensions.
    x_max : float
        Domain: x,y ∈ [-x_max, x_max].
    t_max : float
        Total simulation time.
    dt : float
        Timestep. CFL: c*dt*sqrt(2)/dx < 1.
    mass : float
        Field mass.
    c : float
        Wave speed.
    initial_phi : callable, optional
        Function (X, Y) -> φ(x,y,0). Defaults to a 2D Gaussian wave packet.
    initial_pi : callable, optional
        Function (X, Y) -> π(x,y,0). Defaults to zero.
    boundary : str
        'periodic' or 'absorbing'.
    save_every : int
        Save state every N timesteps.

    Returns
    -------
    SimulationResult
        phi shape: (n_saved, n_x, n_y)
    """
    dx = 2.0 * x_max / n_x
    dy = 2.0 * x_max / n_y
    x = np.linspace(-x_max, x_max, n_x, endpoint=False)
    y = np.linspace(-x_max, x_max, n_y, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    cfl = c * dt * np.sqrt(2) / min(dx, dy)
    if cfl >= 1.0:
        raise ValueError(
            f"CFL condition violated: c*dt*sqrt(2)/dx = {cfl:.3f} ≥ 1."
        )

    if initial_phi is None:
        def initial_phi(Xg: np.ndarray, Yg: np.ndarray) -> np.ndarray:
            r2 = Xg**2 + Yg**2
            return np.exp(-r2 / 1.5) * np.cos(2.0 * Xg)

    if initial_pi is None:
        def initial_pi(Xg: np.ndarray, Yg: np.ndarray) -> np.ndarray:
            return np.zeros_like(Xg)

    phi = initial_phi(X, Y).astype(float)
    pi = initial_pi(X, Y).astype(float)

    n_steps = int(t_max / dt)
    n_saved = n_steps // save_every + 1

    phi_hist = np.zeros((n_saved, n_x, n_y))
    pi_hist = np.zeros((n_saved, n_x, n_y))
    energy_hist = np.zeros(n_saved)
    times = np.zeros(n_saved)

    def laplacian_2d(f: np.ndarray) -> np.ndarray:
        if boundary == "periodic":
            d2x = (np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)) / dx**2
            d2y = (np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)) / dy**2
        else:
            d2x = np.zeros_like(f)
            d2x[1:-1, :] = (f[2:, :] - 2 * f[1:-1, :] + f[:-2, :]) / dx**2
            d2y = np.zeros_like(f)
            d2y[:, 1:-1] = (f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]) / dy**2
        return d2x + d2y

    def compute_energy(phi_: np.ndarray, pi_: np.ndarray) -> float:
        ke = 0.5 * np.sum(pi_**2)
        # Central differences on interior, one-sided at edges
        gx = np.zeros_like(phi_)
        gx[1:-1, :] = (phi_[2:, :] - phi_[:-2, :]) / (2 * dx)
        gx[0, :] = (phi_[1, :] - phi_[-1, :]) / (2 * dx)
        gx[-1, :] = (phi_[0, :] - phi_[-2, :]) / (2 * dx)
        gy = np.zeros_like(phi_)
        gy[:, 1:-1] = (phi_[:, 2:] - phi_[:, :-2]) / (2 * dy)
        gy[:, 0] = (phi_[:, 1] - phi_[:, -1]) / (2 * dy)
        gy[:, -1] = (phi_[:, 0] - phi_[:, -2]) / (2 * dy)
        pe = 0.5 * c**2 * np.sum(gx**2 + gy**2)
        me = 0.5 * mass**2 * np.sum(phi_**2)
        return float((ke + pe + me) * dx * dy)

    phi_hist[0] = phi.copy()
    pi_hist[0] = pi.copy()
    energy_hist[0] = compute_energy(phi, pi)
    times[0] = 0.0

    save_idx = 1
    acc = c**2 * laplacian_2d(phi) - mass**2 * phi
    pi_half = pi + 0.5 * dt * acc

    for step in range(1, n_steps + 1):
        phi = phi + dt * pi_half

        if boundary == "absorbing":
            phi[0, :] = phi[-1, :] = phi[:, 0] = phi[:, -1] = 0.0

        acc = c**2 * laplacian_2d(phi) - mass**2 * phi
        pi_half_new = pi_half + dt * acc
        pi = pi_half + 0.5 * dt * acc

        if step % save_every == 0 and save_idx < n_saved:
            phi_hist[save_idx] = phi.copy()
            pi_hist[save_idx] = pi.copy()
            energy_hist[save_idx] = compute_energy(phi, pi)
            times[save_idx] = step * dt
            save_idx += 1

        pi_half = pi_half_new

    return SimulationResult(
        phi=phi_hist[:save_idx],
        pi=pi_hist[:save_idx],
        energy=energy_hist[:save_idx],
        times=times[:save_idx],
        metadata={
            "n_x": n_x, "n_y": n_y, "x_max": x_max,
            "dx": dx, "dy": dy, "t_max": t_max, "dt": dt,
            "mass": mass, "c": c, "boundary": boundary,
            "cfl": cfl, "equation": "Klein-Gordon 2D",
        },
    )


# ---------------------------------------------------------------------------
# φ⁴ soliton / kink dynamics in 1D (H3: matter-as-excitation)
# ---------------------------------------------------------------------------

def simulate_phi4_1d(
    n_x: int = 300,
    x_max: float = 15.0,
    t_max: float = 30.0,
    dt: float = 0.005,
    lam: float = 1.0,
    v_vev: float = 1.0,
    n_kinks: int = 1,
    kink_positions: list[float] | None = None,
    kink_velocities: list[float] | None = None,
    save_every: int = 20,
) -> SimulationResult:
    """
    Evolve the φ⁴ scalar field theory in 1+1D — kink soliton dynamics.

    Equation of motion:
        ∂²φ/∂t² = ∂²φ/∂x² - λ φ (φ² - v²)

    Potential:
        V(φ) = λ/4 (φ² - v²)²

    Vacuum states: φ = ±v  (spontaneous symmetry breaking)

    Kink solution (exact static):
        φ_kink(x) = v · tanh(√(λ/2) · v · (x - x₀))

    Physical interpretation (H3 — matter as excitation):
        - The kink is a topologically stable localized excitation
        - Cannot be unwound without infinite energy
        - Behaves like a particle: has definite mass, can scatter
        - Two kinks can collide and pass through each other
        - Kink + antikink can annihilate (→ radiation)

    Parameters
    ----------
    n_x : int
        Spatial grid points.
    x_max : float
        Domain: x ∈ [-x_max, x_max].
    t_max : float
        Simulation time.
    dt : float
        Timestep.
    lam : float
        Coupling constant λ > 0.
    v_vev : float
        Vacuum expectation value v (φ → ±v at ±∞).
    n_kinks : int
        Number of kinks (1 = single kink, 2 = kink-antikink pair).
    kink_positions : list of float, optional
        Initial center positions. Defaults to [-5, 5] for n_kinks=2.
    kink_velocities : list of float, optional
        Initial velocities (boost). Defaults to [0] or [0.3, -0.3].
    save_every : int
        Save every N steps.

    Returns
    -------
    SimulationResult

    Examples
    --------
    Single static kink::

        result = simulate_phi4_1d(n_kinks=1)

    Kink-antikink collision::

        result = simulate_phi4_1d(n_kinks=2, kink_velocities=[0.5, -0.5])
    """
    dx = 2.0 * x_max / n_x
    x = np.linspace(-x_max, x_max, n_x, endpoint=False)

    cfl = dt / dx
    if cfl >= 1.0:
        raise ValueError(f"CFL violated: dt/dx = {cfl:.3f} ≥ 1.")

    # Build initial condition from kink solutions
    width = 1.0 / (np.sqrt(lam / 2.0) * v_vev + 1e-10)

    if kink_positions is None:
        if n_kinks == 1:
            kink_positions = [0.0]
        else:
            kink_positions = [-5.0, 5.0]

    if kink_velocities is None:
        if n_kinks == 1:
            kink_velocities = [0.0]
        else:
            kink_velocities = [0.3, -0.3]

    def kink_profile(x0: float, sign: int, vel: float) -> np.ndarray:
        """Boosted kink solution: φ = v * tanh(γ(x-x0)/w) * sign."""
        gamma = 1.0 / np.sqrt(max(1.0 - vel**2, 1e-6))
        xi = gamma * (x - x0)
        return v_vev * np.tanh(xi / width) * sign

    def kink_velocity_profile(x0: float, sign: int, vel: float) -> np.ndarray:
        """Time derivative of boosted kink: π = -v * sign * γ*v_kink/w * sech²."""
        gamma = 1.0 / np.sqrt(max(1.0 - vel**2, 1e-6))
        xi = gamma * (x - x0)
        sech2 = 1.0 / np.cosh(xi / width)**2
        return -v_vev * sign * vel * gamma / width * sech2

    # Build initial condition
    if n_kinks == 1:
        x0 = kink_positions[0]
        vel = kink_velocities[0]
        phi = kink_profile(x0, sign=1, vel=vel)
        pi = kink_velocity_profile(x0, sign=1, vel=vel)
    else:
        # Superpose kinks: start from -v vacuum, add each kink transition
        phi = np.full(n_x, -v_vev, dtype=float)
        pi = np.zeros(n_x, dtype=float)
        running_sign = 1
        for i in range(n_kinks):
            x0 = kink_positions[i] if i < len(kink_positions) else 0.0
            vel = kink_velocities[i] if i < len(kink_velocities) else 0.0
            # Each kink adds v*tanh to the field, alternating direction
            gamma = 1.0 / np.sqrt(max(1.0 - vel**2, 1e-6))
            xi = gamma * (x - x0)
            phi += v_vev * (np.tanh(xi / width) + 1.0) * running_sign
            sech2 = 1.0 / np.cosh(xi / width)**2
            pi += -v_vev * running_sign * vel * gamma / width * sech2
            running_sign *= -1
        # Normalize to correct vacuum at far ends
        phi = np.clip(phi, -2 * v_vev, 2 * v_vev)

    n_steps = int(t_max / dt)
    n_saved = n_steps // save_every + 1

    phi_hist = np.zeros((n_saved, n_x))
    pi_hist = np.zeros((n_saved, n_x))
    energy_hist = np.zeros(n_saved)
    times = np.zeros(n_saved)

    def laplacian(f: np.ndarray) -> np.ndarray:
        # Neumann (zero-derivative) BC: ghost points equal boundary values
        # This preserves the kink topology at the boundaries
        lap = np.zeros_like(f)
        lap[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / dx**2
        # At boundaries: f_ghost = f_boundary (zero second derivative)
        lap[0] = (f[1] - 2 * f[0] + f[0]) / dx**2   # = (f[1] - f[0]) / dx²
        lap[-1] = (f[-1] - 2 * f[-1] + f[-2]) / dx**2  # = (f[-2] - f[-1]) / dx²
        return lap

    def dV_dphi(phi_: np.ndarray) -> np.ndarray:
        return lam * phi_ * (phi_**2 - v_vev**2)

    def compute_energy(phi_: np.ndarray, pi_: np.ndarray) -> float:
        ke = 0.5 * np.sum(pi_**2)
        # One-sided differences at boundaries (kink is NOT periodic)
        grad = np.zeros_like(phi_)
        grad[1:-1] = (phi_[2:] - phi_[:-2]) / (2 * dx)
        grad[0] = (phi_[1] - phi_[0]) / dx
        grad[-1] = (phi_[-1] - phi_[-2]) / dx
        ge = 0.5 * np.sum(grad**2)
        pe = np.sum(lam / 4.0 * (phi_**2 - v_vev**2)**2)
        return float((ke + ge + pe) * dx)

    phi_hist[0] = phi.copy()
    pi_hist[0] = pi.copy()
    energy_hist[0] = compute_energy(phi, pi)
    times[0] = 0.0
    save_idx = 1

    acc = laplacian(phi) - dV_dphi(phi)
    pi_half = pi + 0.5 * dt * acc

    for step in range(1, n_steps + 1):
        phi = phi + dt * pi_half
        acc = laplacian(phi) - dV_dphi(phi)
        pi_half_new = pi_half + dt * acc
        pi = pi_half + 0.5 * dt * acc

        if step % save_every == 0 and save_idx < n_saved:
            phi_hist[save_idx] = phi.copy()
            pi_hist[save_idx] = pi.copy()
            energy_hist[save_idx] = compute_energy(phi, pi)
            times[save_idx] = step * dt
            save_idx += 1

        pi_half = pi_half_new

    return SimulationResult(
        phi=phi_hist[:save_idx],
        pi=pi_hist[:save_idx],
        energy=energy_hist[:save_idx],
        times=times[:save_idx],
        metadata={
            "n_x": n_x, "x_max": x_max, "dx": dx,
            "t_max": t_max, "dt": dt,
            "lambda": lam, "v_vev": v_vev,
            "n_kinks": n_kinks, "cfl": cfl,
            "equation": "phi^4 kink 1D",
        },
    )


# ---------------------------------------------------------------------------
# Curved background: KG on Schwarzschild-like potential (1D effective)
# ---------------------------------------------------------------------------

def simulate_kg_curved_1d(
    n_x: int = 300,
    r_min: float = 1.5,
    r_max: float = 20.0,
    t_max: float = 30.0,
    dt: float = 0.01,
    mass: float = 0.0,
    r_s: float = 1.0,
    l_angular: int = 0,
    save_every: int = 10,
) -> SimulationResult:
    """
    Evolve a scalar field on a Schwarzschild background (1+1D radial equation).

    Uses the Regge-Wheeler tortoise coordinate x* = r + r_s ln|r/r_s - 1|
    which maps the exterior region r ∈ (r_s, ∞) to x* ∈ (-∞, ∞).

    In tortoise coordinates the wave equation becomes:
        ∂²Ψ/∂t² - ∂²Ψ/∂x*² + V_eff(r) Ψ = 0

    where Ψ = r·φ and the effective potential is:
        V_eff(r) = (1 - r_s/r) [m² + l(l+1)/r² + r_s/r³]

    This describes:
    - Quasinormal mode ringing near the black hole
    - Greybody factors (partial absorption by the potential barrier)
    - Late-time tails (Price's law: power-law decay)
    - Nothing escapes from r < r_s (event horizon)

    Parameters
    ----------
    n_x : int
        Grid points in tortoise coordinate.
    r_min : float
        Minimum r (must be > r_s). Maps to x*_min.
    r_max : float
        Maximum r. Maps to x*_max.
    t_max : float
        Simulation time.
    dt : float
        Timestep.
    mass : float
        Scalar field mass.
    r_s : float
        Schwarzschild radius (= 2GM/c²).
    l_angular : int
        Angular momentum quantum number l.
    save_every : int
        Save every N steps.

    Returns
    -------
    SimulationResult
        phi contains Ψ = r·φ in tortoise coordinates.
    """
    # Tortoise coordinate grid
    def tortoise(r: np.ndarray) -> np.ndarray:
        return r + r_s * np.log(np.abs(r / r_s - 1.0))

    def r_from_tortoise_approx(xstar: np.ndarray) -> np.ndarray:
        """Approximate inverse using Newton's method."""
        r = xstar.copy() + r_s  # initial guess
        for _ in range(20):
            f = r + r_s * np.log(np.abs(r / r_s - 1.0)) - xstar
            df = 1.0 + r_s / (r - r_s)
            r = r - f / df
            r = np.maximum(r, r_s + 1e-6)
        return r

    r_grid = np.linspace(r_min, r_max, n_x)
    xstar_grid = tortoise(r_grid)
    dx = (xstar_grid[-1] - xstar_grid[0]) / (n_x - 1)

    # Effective potential V_eff(r)
    def V_eff(r: np.ndarray) -> np.ndarray:
        f = 1.0 - r_s / r
        return f * (mass**2 + l_angular * (l_angular + 1) / r**2 + r_s / r**3)

    V = V_eff(r_grid)

    # Initial condition: Gaussian pulse outside the barrier
    r0 = 0.5 * (r_min + r_max)
    sigma = 1.0
    psi = np.exp(-(r_grid - r0)**2 / sigma**2)
    pi_field = np.zeros(n_x)

    n_steps = int(t_max / dt)
    n_saved = n_steps // save_every + 1

    phi_hist = np.zeros((n_saved, n_x))
    pi_hist = np.zeros((n_saved, n_x))
    energy_hist = np.zeros(n_saved)
    times = np.zeros(n_saved)

    def laplacian_abs(f: np.ndarray) -> np.ndarray:
        lap = np.zeros_like(f)
        lap[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / dx**2
        return lap

    def compute_energy(psi_: np.ndarray, pi_: np.ndarray) -> float:
        ke = 0.5 * np.sum(pi_**2)
        grad = np.gradient(psi_, dx)
        pe = 0.5 * np.sum(grad**2 + V * psi_**2)
        return float((ke + pe) * dx)

    phi_hist[0] = psi.copy()
    pi_hist[0] = pi_field.copy()
    energy_hist[0] = compute_energy(psi, pi_field)
    times[0] = 0.0
    save_idx = 1

    acc = laplacian_abs(psi) - V * psi
    pi_half = pi_field + 0.5 * dt * acc

    for step in range(1, n_steps + 1):
        psi = psi + dt * pi_half
        psi[0] = 0.0   # absorbing at inner boundary (horizon)
        psi[-1] = 0.0  # absorbing at outer boundary

        acc = laplacian_abs(psi) - V * psi
        pi_half_new = pi_half + dt * acc
        pi_field = pi_half + 0.5 * dt * acc

        if step % save_every == 0 and save_idx < n_saved:
            phi_hist[save_idx] = psi.copy()
            pi_hist[save_idx] = pi_field.copy()
            energy_hist[save_idx] = compute_energy(psi, pi_field)
            times[save_idx] = step * dt
            save_idx += 1

        pi_half = pi_half_new

    return SimulationResult(
        phi=phi_hist[:save_idx],
        pi=pi_hist[:save_idx],
        energy=energy_hist[:save_idx],
        times=times[:save_idx],
        metadata={
            "n_x": n_x, "r_min": r_min, "r_max": r_max,
            "dx_tortoise": dx, "t_max": t_max, "dt": dt,
            "mass": mass, "r_s": r_s, "l_angular": l_angular,
            "equation": "KG on Schwarzschild (tortoise coords)",
        },
    )
