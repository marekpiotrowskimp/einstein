"""
Tests for einstein.visualization

Uses matplotlib Agg backend (no display required).
Verifies that each plot function:
- Returns a matplotlib Figure
- Does not raise exceptions
- Produces output files when save_path is provided
"""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot

import numpy as np
import pytest

from einstein.visualization import (
    plot_embedding_diagram,
    plot_geodesics,
    plot_curvature_heatmap,
    plot_schwarzschild_spacetime,
    plot_expansion,
    plot_field_excitation,
)
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test to free memory."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# 1. Embedding diagram
# ---------------------------------------------------------------------------

class TestEmbeddingDiagram:
    def test_returns_figure(self):
        def flat(u, v):
            return np.zeros_like(u)

        fig = plot_embedding_diagram(flat, n_points=20)
        assert isinstance(fig, plt.Figure)

    def test_gravity_well(self):
        def well(u, v):
            r = np.sqrt(u**2 + v**2) + 0.5
            return -1.0 / r

        fig = plot_embedding_diagram(well, n_points=20, title="Gravity Well")
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, tmp_path):
        path = str(tmp_path / "embedding.png")

        def fn(u, v):
            return np.exp(-(u**2 + v**2))

        plot_embedding_diagram(fn, n_points=15, save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# 2. Geodesics
# ---------------------------------------------------------------------------

class TestGeodesics:
    def _flat_christoffel(self, x, y):
        """Zero Christoffel symbols — flat space, straight geodesics."""
        return np.zeros((2, 2, 2))

    def test_straight_lines_flat_space(self):
        ics = [(0.0, 0.0, 1.0, 0.5), (-2.0, 1.0, 0.8, -0.3)]
        fig = plot_geodesics(self._flat_christoffel, ics, tau_max=5.0, n_steps=100)
        assert isinstance(fig, plt.Figure)

    def test_single_geodesic(self):
        fig = plot_geodesics(
            self._flat_christoffel,
            [(0.0, 0.0, 1.0, 0.0)],
            tau_max=3.0,
            n_steps=50,
        )
        assert isinstance(fig, plt.Figure)

    def test_with_background(self):
        def bg(x, y):
            return np.exp(-(x**2 + y**2))

        fig = plot_geodesics(
            self._flat_christoffel,
            [(1.0, 0.0, 0.0, 1.0)],
            tau_max=3.0,
            n_steps=50,
            background_fn=bg,
        )
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, tmp_path):
        path = str(tmp_path / "geodesics.png")
        plot_geodesics(
            self._flat_christoffel,
            [(0.0, 0.0, 1.0, 0.5)],
            tau_max=2.0,
            n_steps=50,
            save_path=path,
        )
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# 3. Curvature heatmap
# ---------------------------------------------------------------------------

class TestCurvatureHeatmap:
    def test_flat_space(self):
        def zero_curvature(X, Y):
            return np.zeros_like(X)

        fig = plot_curvature_heatmap(zero_curvature, n_points=50)
        assert isinstance(fig, plt.Figure)

    def test_point_mass_curvature(self):
        def R_point(X, Y):
            r2 = X**2 + Y**2 + 0.1
            return 1.0 / r2

        fig = plot_curvature_heatmap(R_point, n_points=80, title="Point Mass R")
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, tmp_path):
        path = str(tmp_path / "curvature.png")
        plot_curvature_heatmap(lambda X, Y: X**2 - Y**2, n_points=50, save_path=path)
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# 4. Schwarzschild spacetime
# ---------------------------------------------------------------------------

class TestSchwarzschild:
    def test_default(self):
        fig = plot_schwarzschild_spacetime()
        assert isinstance(fig, plt.Figure)

    def test_custom_rs(self):
        fig = plot_schwarzschild_spacetime(r_s=2.0, r_max=10.0, n_cones=6)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, tmp_path):
        path = str(tmp_path / "schwarzschild.png")
        plot_schwarzschild_spacetime(save_path=path)
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# 5. Cosmological expansion
# ---------------------------------------------------------------------------

class TestExpansion:
    def test_default_lcdm(self):
        fig = plot_expansion()
        assert isinstance(fig, plt.Figure)

    def test_custom_scale_factor(self):
        def de_sitter(t):
            return np.exp(0.1 * t)

        fig = plot_expansion(scale_factor_fn=de_sitter, t_max=10.0)
        assert isinstance(fig, plt.Figure)

    def test_matter_dominated(self):
        def matter(t):
            return (t / 14.0) ** (2 / 3)

        fig = plot_expansion(scale_factor_fn=matter, n_worldlines=5)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, tmp_path):
        path = str(tmp_path / "expansion.png")
        plot_expansion(save_path=path)
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# 6. Scalar field excitation
# ---------------------------------------------------------------------------

class TestFieldExcitation:
    def test_default(self):
        fig = plot_field_excitation()
        assert isinstance(fig, plt.Figure)

    def test_custom_field(self):
        def plane_wave(X, Y):
            return np.sin(2 * np.pi * X / 3.0) * np.exp(-(Y**2) / 4)

        fig = plot_field_excitation(psi_fn=plane_wave, n_points=80)
        assert isinstance(fig, plt.Figure)

    def test_double_soliton(self):
        def two_particles(X, Y):
            r1 = (X - 1.5)**2 + Y**2
            r2 = (X + 1.5)**2 + Y**2
            return 1.5 * np.exp(-r1 / 0.5) + 1.0 * np.exp(-r2 / 0.5)

        fig = plot_field_excitation(
            psi_fn=two_particles,
            title="Two-particle excitation",
        )
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, tmp_path):
        path = str(tmp_path / "excitation.png")
        plot_field_excitation(save_path=path)
        assert os.path.exists(path)
