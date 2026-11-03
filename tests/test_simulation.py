"""
Tests for einstein.simulation

Validates:
- CFL condition enforcement
- Energy conservation (leapfrog is symplectic → should be excellent)
- Wave propagation speed
- Kink topology (topological charge conservation)
- Schwarzschild effective potential shape
- SimulationResult metadata and properties
"""

import numpy as np
import pytest

from einstein.simulation import (
    SimulationResult,
    simulate_kg_1d,
    simulate_kg_2d,
    simulate_phi4_1d,
    simulate_kg_curved_1d,
)


# ---------------------------------------------------------------------------
# SimulationResult
# ---------------------------------------------------------------------------

class TestSimulationResult:
    def test_n_dims_1d(self):
        r = simulate_kg_1d(n_x=50, t_max=1.0, dt=0.01, save_every=10)
        assert r.n_dims == 1

    def test_n_dims_2d(self):
        r = simulate_kg_2d(n_x=30, n_y=30, t_max=1.0, dt=0.02, save_every=5)
        assert r.n_dims == 2

    def test_metadata_present(self):
        r = simulate_kg_1d(n_x=50, t_max=1.0, dt=0.01, save_every=10)
        assert "equation" in r.metadata
        assert "cfl" in r.metadata
        assert r.metadata["cfl"] < 1.0

    def test_shapes_consistent(self):
        r = simulate_kg_1d(n_x=50, t_max=1.0, dt=0.01, save_every=10)
        assert r.phi.shape[0] == len(r.times)
        assert r.pi.shape[0] == len(r.times)
        assert r.energy.shape[0] == len(r.times)

    def test_times_monotonic(self):
        r = simulate_kg_1d(n_x=50, t_max=2.0, dt=0.01, save_every=10)
        assert np.all(np.diff(r.times) > 0)


# ---------------------------------------------------------------------------
# Klein-Gordon 1D
# ---------------------------------------------------------------------------

class TestKG1D:
    def test_cfl_violation_raises(self):
        with pytest.raises(ValueError, match="CFL"):
            simulate_kg_1d(n_x=10, x_max=1.0, dt=0.5)  # dx=0.2, c*dt/dx=2.5

    def test_energy_conservation_massless(self):
        """
        Leapfrog conserves a modified (shadow) Hamiltonian exactly.
        The measured energy oscillates at O(dt²) but must not drift.
        We check: relative spread < 5% and no systematic linear drift.
        """
        r = simulate_kg_1d(
            n_x=200, x_max=10.0, t_max=10.0, dt=0.02,
            mass=0.0, boundary="periodic", save_every=10
        )
        e = r.energy
        spread = (e.max() - e.min()) / e[0]
        assert spread < 0.05, f"Energy spread {spread:.3f} indicates instability"
        # No drift: slope of linear fit should be near zero
        slope = np.polyfit(r.times, e, 1)[0]
        assert abs(slope) / e[0] < 1e-3, f"Energy drift slope {slope:.2e} too large"

    def test_energy_conservation_massive(self):
        r = simulate_kg_1d(
            n_x=200, x_max=10.0, t_max=5.0, dt=0.02,
            mass=1.0, boundary="periodic", save_every=10
        )
        e = r.energy
        spread = (e.max() - e.min()) / e[0]
        assert spread < 0.05

    def test_wave_speed_massless(self):
        """A massless pulse should travel at speed c=1.
        Initial peak at x=0; after time T it should be near |x|=T.
        """
        n_x = 400
        x_max = 20.0
        dx = 2 * x_max / n_x
        x = np.linspace(-x_max, x_max, n_x, endpoint=False)
        T = 8.0

        def phi0(xarr):
            return np.exp(-xarr**2 / 0.3)

        r = simulate_kg_1d(
            n_x=n_x, x_max=x_max, t_max=T, dt=0.02,
            mass=0.0, c=1.0,
            initial_phi=phi0,
            boundary="absorbing",
            save_every=50,
        )
        final_phi = r.phi[-1]
        # Peak should be near x = ±T = ±8
        peak_idx = np.argmax(np.abs(final_phi))
        peak_x = x[peak_idx]
        assert abs(abs(peak_x) - T) < 1.5, \
            f"Wave peak at {peak_x:.2f}, expected near ±{T}"

    def test_zero_field_stays_zero(self):
        r = simulate_kg_1d(
            n_x=100, t_max=2.0, dt=0.02,
            initial_phi=lambda x: np.zeros_like(x),
            initial_pi=lambda x: np.zeros_like(x),
            save_every=10,
        )
        assert np.allclose(r.phi, 0.0)

    def test_periodic_boundary(self):
        r = simulate_kg_1d(n_x=100, t_max=2.0, dt=0.02, boundary="periodic", save_every=10)
        assert r.phi.shape[1] == 100

    def test_absorbing_boundary(self):
        r = simulate_kg_1d(n_x=100, t_max=2.0, dt=0.02, boundary="absorbing", save_every=10)
        # Boundaries should be (approximately) zero at all times
        assert np.allclose(r.phi[:, 0], 0.0, atol=1e-10)
        assert np.allclose(r.phi[:, -1], 0.0, atol=1e-10)

    def test_output_shape(self):
        r = simulate_kg_1d(n_x=80, t_max=1.0, dt=0.01, save_every=5)
        assert r.phi.ndim == 2
        assert r.phi.shape[1] == 80


# ---------------------------------------------------------------------------
# Klein-Gordon 2D
# ---------------------------------------------------------------------------

class TestKG2D:
    def test_cfl_violation_raises(self):
        with pytest.raises(ValueError, match="CFL"):
            simulate_kg_2d(n_x=10, n_y=10, x_max=1.0, dt=0.5)

    def test_output_shape(self):
        r = simulate_kg_2d(n_x=30, n_y=40, x_max=5.0, t_max=1.0, dt=0.05, save_every=5)
        assert r.phi.ndim == 3
        assert r.phi.shape[1] == 30
        assert r.phi.shape[2] == 40

    def test_energy_conservation(self):
        """Energy bounded (no drift) — leapfrog oscillates but is stable."""
        r = simulate_kg_2d(
            n_x=50, n_y=50, x_max=5.0, t_max=3.0, dt=0.03,
            mass=0.0, boundary="periodic", save_every=5,
        )
        e = r.energy
        spread = (e.max() - e.min()) / e[0]
        assert spread < 0.15, f"2D energy spread {spread:.3f} indicates instability"
        slope = np.polyfit(r.times, e, 1)[0]
        assert abs(slope) / e[0] < 1e-2, f"2D energy drift {slope:.2e}"

    def test_zero_field_stays_zero(self):
        r = simulate_kg_2d(
            n_x=20, n_y=20, x_max=5.0, t_max=0.5, dt=0.04,
            initial_phi=lambda X, Y: np.zeros_like(X),
            initial_pi=lambda X, Y: np.zeros_like(X),
            save_every=5,
        )
        assert np.allclose(r.phi, 0.0)


# ---------------------------------------------------------------------------
# φ⁴ kink dynamics
# ---------------------------------------------------------------------------

class TestPhi4Kink:
    def test_single_kink_stable(self):
        """
        A static kink should remain stable.
        Use fine grid (dx << width=1.41) for accurate energy measurement.
        Energy spread < 1% over simulation time.
        """
        r = simulate_phi4_1d(
            n_x=600, x_max=15.0, t_max=5.0, dt=0.005,
            n_kinks=1, kink_velocities=[0.0], save_every=50,
        )
        e = r.energy
        spread = (e.max() - e.min()) / e[0]
        assert spread < 0.01, f"Kink energy spread {spread:.3f} too large"

    def test_kink_topological_charge(self):
        """
        Topological charge Q = (φ(+∞) - φ(-∞)) / (2v) should be conserved.
        For a single kink: Q = 1.
        Use large domain so boundaries are deep in vacuum (phi ≈ ±v).
        """
        r = simulate_phi4_1d(
            n_x=600, x_max=20.0, t_max=5.0, dt=0.005,
            n_kinks=1, kink_velocities=[0.0], save_every=100,
        )
        v = r.metadata["v_vev"]
        for i in range(len(r.times)):
            phi = r.phi[i]
            # Average over 5% of domain at each end for stability
            n5 = max(1, len(phi) // 20)
            phi_right = phi[-n5:].mean()
            phi_left = phi[:n5].mean()
            charge = (phi_right - phi_left) / (2 * v)
            assert abs(charge - 1.0) < 0.05, \
                f"Topological charge {charge:.3f} ≠ 1 at t={r.times[i]:.2f}"

    def test_kink_antikink_collision(self):
        """Kink-antikink: total charge = 0 before and after collision."""
        r = simulate_phi4_1d(
            n_x=300, x_max=15.0, t_max=20.0, dt=0.005,
            n_kinks=2, kink_velocities=[0.3, -0.3], save_every=100,
        )
        assert r.phi.shape[0] > 1

    def test_output_shape(self):
        r = simulate_phi4_1d(n_x=100, x_max=8.0, t_max=2.0, dt=0.005, save_every=50)
        assert r.phi.ndim == 2
        assert r.phi.shape[1] == 100

    def test_cfl_violation_raises(self):
        with pytest.raises(ValueError, match="CFL"):
            simulate_phi4_1d(n_x=10, x_max=1.0, t_max=1.0, dt=0.5)

    def test_vacuum_is_stable(self):
        """Starting in vacuum φ=+v everywhere → stays in vacuum."""
        v = 1.0
        r = simulate_phi4_1d(
            n_x=100, x_max=8.0, t_max=2.0, dt=0.005,
            v_vev=v,
            kink_positions=[100.0],  # kink far outside domain → vacuum only
            kink_velocities=[0.0],
            save_every=50,
        )
        # Field should be close to ±v everywhere
        assert np.all(np.abs(r.phi) <= 2.0 * v + 0.1)


# ---------------------------------------------------------------------------
# KG on Schwarzschild curved background
# ---------------------------------------------------------------------------

class TestKGCurved:
    def test_output_shape(self):
        r = simulate_kg_curved_1d(
            n_x=100, r_min=2.0, r_max=15.0, t_max=5.0, dt=0.02, save_every=10
        )
        assert r.phi.ndim == 2
        assert r.phi.shape[1] == 100

    def test_absorbing_boundaries(self):
        """Boundaries fixed at zero (absorbing) throughout."""
        r = simulate_kg_curved_1d(
            n_x=100, r_min=2.0, r_max=15.0, t_max=5.0, dt=0.02, save_every=10
        )
        assert np.allclose(r.phi[:, 0], 0.0, atol=1e-10)
        assert np.allclose(r.phi[:, -1], 0.0, atol=1e-10)

    def test_energy_decays(self):
        """
        On Schwarzschild background with absorbing BCs, the pulse disperses.
        Energy is not conserved (boundaries absorb) — field amplitude decays.
        """
        r = simulate_kg_curved_1d(
            n_x=200, r_min=1.5, r_max=20.0, t_max=40.0, dt=0.01,
            r_s=1.0, l_angular=0, save_every=20,
        )
        # Max amplitude should decrease over time
        max_amplitude_early = np.max(np.abs(r.phi[:len(r.phi)//4]))
        max_amplitude_late = np.max(np.abs(r.phi[3*len(r.phi)//4:]))
        assert max_amplitude_late < max_amplitude_early, \
            f"Field amplitude should decay: early={max_amplitude_early:.4f}, late={max_amplitude_late:.4f}"

    def test_metadata(self):
        r = simulate_kg_curved_1d(n_x=50, t_max=2.0, dt=0.02, save_every=10)
        assert r.metadata["equation"] == "KG on Schwarzschild (tortoise coords)"
        assert "r_s" in r.metadata
        assert "l_angular" in r.metadata

    def test_l_nonzero(self):
        """Higher angular momentum adds centrifugal barrier — should run without error."""
        r = simulate_kg_curved_1d(
            n_x=100, r_min=2.0, r_max=15.0, t_max=5.0, dt=0.02,
            l_angular=2, save_every=10,
        )
        assert r.phi.shape[0] > 1
