"""
einstein.visualization — Geometric visualization of curved spacetime.

Provides publication-quality matplotlib figures for:

1. Embedding diagrams         — curved 2D surfaces embedded in 3D
2. Geodesic trajectories      — paths of free-falling objects
3. Curvature heatmaps         — Ricci scalar R across parameter space
4. Schwarzschild spacetime    — light cones, event horizon, singularity
5. Cosmological expansion     — Hubble sphere, particle/event horizons
6. Scalar field excitations   — matter-as-excitation profiles (H3)

All functions return matplotlib Figure objects and optionally save to disk.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

_DARK_BG = "#0d0d1a"
_ACCENT = "#00d4ff"
_ACCENT2 = "#ff6b35"
_ACCENT3 = "#7fff7f"
_GRID_COLOR = "#1e1e3a"
_TEXT_COLOR = "#e0e0e0"


def _apply_dark_style(fig: plt.Figure, axes) -> None:
    fig.patch.set_facecolor(_DARK_BG)
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(_DARK_BG)
        ax.tick_params(colors=_TEXT_COLOR)
        ax.xaxis.label.set_color(_TEXT_COLOR)
        ax.yaxis.label.set_color(_TEXT_COLOR)
        ax.title.set_color(_TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID_COLOR)


# ===========================================================================
# 1. Embedding diagram
# ===========================================================================

def plot_embedding_diagram(
    surface_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    u_range: tuple[float, float] = (-3.0, 3.0),
    v_range: tuple[float, float] = (-3.0, 3.0),
    n_points: int = 60,
    title: str = "Embedding Diagram",
    colormap: str = "plasma",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Render a 2D curved surface embedded in 3D space.

    The surface is defined by a function z = f(u, v), visualizing how
    curvature deforms the geometry of space.

    Parameters
    ----------
    surface_fn : callable
        Function (u_grid, v_grid) -> z_grid. All arrays are 2D numpy arrays.
    u_range, v_range : tuple
        Parameter ranges.
    n_points : int
        Grid resolution.
    title : str
        Plot title.
    colormap : str
        Matplotlib colormap name.
    save_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    Schwarzschild-like gravity well::

        def gravity_well(u, v):
            r = np.sqrt(u**2 + v**2) + 0.5
            return -1.0 / r

        fig = plot_embedding_diagram(gravity_well, title="Gravity Well")

    Gaussian excitation (H3: matter as local space excitation)::

        def excitation(u, v):
            return np.exp(-(u**2 + v**2) / 0.5)

        fig = plot_embedding_diagram(excitation, title="Local Excitation")
    """
    u = np.linspace(*u_range, n_points)
    v = np.linspace(*v_range, n_points)
    U, V = np.meshgrid(u, v)
    Z = surface_fn(U, V)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _apply_dark_style(fig, ax)
    ax.set_facecolor(_DARK_BG)

    surf = ax.plot_surface(
        U, V, Z,
        cmap=colormap,
        linewidth=0,
        antialiased=True,
        alpha=0.9,
    )

    # Contour projection on bottom
    ax.contourf(U, V, Z, zdir="z", offset=Z.min() - 0.3, cmap=colormap, alpha=0.3)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label("z (curvature depth)", color=_TEXT_COLOR)
    cbar.ax.yaxis.set_tick_params(color=_TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_TEXT_COLOR)

    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_zlabel("z")
    ax.set_title(title, pad=15)
    ax.tick_params(colors=_TEXT_COLOR)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    return fig


# ===========================================================================
# 2. Geodesic trajectories (numerical integration via RK4)
# ===========================================================================

def _rk4_step(
    f: Callable,
    y: np.ndarray,
    t: float,
    dt: float,
) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt * k1 / 2)
    k3 = f(t + dt / 2, y + dt * k2 / 2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def plot_geodesics(
    christoffel_fn: Callable[[float, float], np.ndarray],
    initial_conditions: Sequence[tuple[float, float, float, float]],
    tau_max: float = 10.0,
    n_steps: int = 2000,
    title: str = "Geodesics",
    background_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Integrate and plot geodesic trajectories on a 2D manifold.

    Solves the geodesic equation numerically:
        d²xᵅ/dτ² + Γᵅ_μν (dxᵘ/dτ)(dxᵛ/dτ) = 0

    Parameters
    ----------
    christoffel_fn : callable
        Function (x, y) -> Gamma[alpha][mu][nu] as 2x2x2 numpy array.
        Christoffel symbols evaluated at position (x, y).
    initial_conditions : list of (x0, y0, vx0, vy0)
        Starting position and velocity for each geodesic.
    tau_max : float
        Affine parameter range [0, tau_max].
    n_steps : int
        Integration steps.
    title : str
        Plot title.
    background_fn : callable, optional
        If provided, renders z = f(x, y) as a heatmap background.
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 9))
    _apply_dark_style(fig, ax)

    dtau = tau_max / n_steps
    colors = [_ACCENT, _ACCENT2, _ACCENT3, "#ff00ff", "#ffff00", "#00ffaa"]

    # Background curvature field
    if background_fn is not None:
        xs = np.linspace(-5, 5, 200)
        ys = np.linspace(-5, 5, 200)
        X, Y = np.meshgrid(xs, ys)
        Z = background_fn(X, Y)
        ax.imshow(
            Z,
            extent=[-5, 5, -5, 5],
            origin="lower",
            cmap="inferno",
            alpha=0.4,
            aspect="auto",
        )

    for i, (x0, y0, vx0, vy0) in enumerate(initial_conditions):
        color = colors[i % len(colors)]
        state = np.array([x0, y0, vx0, vy0], dtype=float)
        xs, ys = [x0], [y0]

        for _ in range(n_steps):
            x, y, vx, vy = state

            def dydt(tau: float, s: np.ndarray) -> np.ndarray:
                px, py, pvx, pvy = s
                try:
                    G = christoffel_fn(px, py)
                    ax_acc = -(
                        G[0][0][0] * pvx * pvx
                        + 2 * G[0][0][1] * pvx * pvy
                        + G[0][1][1] * pvy * pvy
                    )
                    ay_acc = -(
                        G[1][0][0] * pvx * pvx
                        + 2 * G[1][0][1] * pvx * pvy
                        + G[1][1][1] * pvy * pvy
                    )
                except Exception:
                    ax_acc = ay_acc = 0.0
                return np.array([pvx, pvy, ax_acc, ay_acc])

            state = _rk4_step(dydt, state, 0, dtau)
            xs.append(state[0])
            ys.append(state[1])

        ax.plot(xs, ys, color=color, lw=1.5, alpha=0.85,
                label=f"geodesic {i+1}: ({x0:.1f},{y0:.1f}) v=({vx0:.1f},{vy0:.1f})")
        ax.plot(x0, y0, "o", color=color, markersize=6)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(fontsize=8, labelcolor=_TEXT_COLOR,
               facecolor=_DARK_BG, edgecolor=_GRID_COLOR)
    ax.grid(True, color=_GRID_COLOR, lw=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    return fig


# ===========================================================================
# 3. Curvature heatmap
# ===========================================================================

def plot_curvature_heatmap(
    curvature_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_range: tuple[float, float] = (-4.0, 4.0),
    y_range: tuple[float, float] = (-4.0, 4.0),
    n_points: int = 300,
    title: str = "Ricci Scalar Curvature R(x, y)",
    colormap: str = "RdBu_r",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Render Ricci scalar curvature R as a 2D heatmap.

    Parameters
    ----------
    curvature_fn : callable
        Function (X_grid, Y_grid) -> R_grid. Scalar curvature at each point.
    x_range, y_range : tuple
        Spatial extent.
    n_points : int
        Resolution.
    title : str
        Plot title.
    colormap : str
        Diverging colormap recommended (red=positive, blue=negative curvature).
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x = np.linspace(*x_range, n_points)
    y = np.linspace(*y_range, n_points)
    X, Y = np.meshgrid(x, y)
    R = curvature_fn(X, Y)

    fig, ax = plt.subplots(figsize=(9, 8))
    _apply_dark_style(fig, ax)

    vmax = np.percentile(np.abs(R[np.isfinite(R)]), 95)
    im = ax.imshow(
        R,
        extent=[*x_range, *y_range],
        origin="lower",
        cmap=colormap,
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("R  (Ricci scalar)", color=_TEXT_COLOR)
    cbar.ax.yaxis.set_tick_params(color=_TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_TEXT_COLOR)

    # Contour lines (skip if field is constant / all-zero)
    if vmax > 0:
        levels = np.linspace(-vmax, vmax, 15)
        ax.contour(X, Y, R, levels=levels, colors="white", alpha=0.15, linewidths=0.5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    return fig


# ===========================================================================
# 4. Schwarzschild spacetime diagram
# ===========================================================================

def plot_schwarzschild_spacetime(
    r_s: float = 1.0,
    r_max: float = 6.0,
    n_cones: int = 8,
    title: str = "Schwarzschild Spacetime — Light Cones & Event Horizon",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Schwarzschild spacetime diagram showing light cone tilting near the
    event horizon (Schwarzschild radius r_s = 2GM/c²).

    As r → r_s:
    - Light cones tilt toward the singularity
    - At r = r_s: the horizon — future light cone points inward
    - r < r_s: all future-directed paths lead to singularity

    Parameters
    ----------
    r_s : float
        Schwarzschild radius (event horizon). Default = 1.
    r_max : float
        Maximum radial extent of diagram.
    n_cones : int
        Number of light cones to draw.
    title : str
        Plot title.
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(11, 9))
    _apply_dark_style(fig, ax)

    r_vals = np.linspace(0.15, r_max, 500)
    t_vals = np.linspace(-4, 4, 500)

    # Singularity region
    ax.axvspan(0, r_s, color="#3a0000", alpha=0.7, label="Singularity / r < r_s")

    # Event horizon
    ax.axvline(r_s, color="#ff2222", lw=2.5, ls="--", label=f"Event horizon  r = r_s = {r_s}")

    # Ingoing / outgoing null geodesics (Eddington-Finkelstein style)
    # dr/dt = ±(1 - r_s/r) for outgoing/ingoing light
    for r0 in np.linspace(1.3 * r_s, r_max * 0.95, 6):
        # outgoing light from r0
        r_out = [r0]
        t_out = [0.0]
        dt = 0.05
        for _ in range(60):
            r_cur = r_out[-1]
            if r_cur > r_max or r_cur < 0.01:
                break
            slope = 1 - r_s / r_cur
            r_out.append(r_cur + slope * dt)
            t_out.append(t_out[-1] + dt)
        ax.plot(r_out, t_out, color=_ACCENT, lw=1.0, alpha=0.5)

        # ingoing light from r0
        r_in = [r0]
        t_in = [0.0]
        for _ in range(80):
            r_cur = r_in[-1]
            if r_cur < 0.01 or r_cur > r_max:
                break
            slope = -(1 - r_s / r_cur) if r_cur > r_s else -0.5
            r_in.append(r_cur + slope * dt)
            t_in.append(t_in[-1] + dt)
        ax.plot(r_in, t_in, color=_ACCENT2, lw=1.0, alpha=0.5)

    # Light cones at selected radii
    cone_positions = np.linspace(0.3 * r_s, r_max * 0.9, n_cones)
    for r0 in cone_positions:
        factor = max(0.0, 1.0 - r_s / r0) if r0 > r_s else 0.0
        cone_half = 0.25
        dt_cone = 0.35

        # Right arm (outgoing)
        ax.annotate(
            "",
            xy=(r0 + factor * cone_half, dt_cone),
            xytext=(r0, 0),
            arrowprops=dict(arrowstyle="->", color=_ACCENT, lw=1.2),
        )
        # Left arm (ingoing)
        ax.annotate(
            "",
            xy=(r0 - cone_half, dt_cone),
            xytext=(r0, 0),
            arrowprops=dict(arrowstyle="->", color=_ACCENT2, lw=1.2),
        )

    # Labels
    ax.text(r_s * 0.5, 3.5, "r < r_s\n(interior)", color="#ff8888",
            ha="center", fontsize=9)
    ax.text(r_s * 1.05, -3.8, f"r_s = {r_s}", color="#ff4444",
            fontsize=9, rotation=90)
    ax.text(r_max * 0.7, 3.5, "outgoing light →", color=_ACCENT, fontsize=8)
    ax.text(r_max * 0.7, 3.0, "← ingoing light", color=_ACCENT2, fontsize=8)

    ax.set_xlim(0, r_max)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("r  (radial coordinate)")
    ax.set_ylabel("t  (coordinate time)")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="upper right", labelcolor=_TEXT_COLOR,
               facecolor=_DARK_BG, edgecolor=_GRID_COLOR)
    ax.grid(True, color=_GRID_COLOR, lw=0.4)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    return fig


# ===========================================================================
# 5. Cosmological expansion
# ===========================================================================

def plot_expansion(
    scale_factor_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    t_max: float = 14.0,
    n_worldlines: int = 7,
    title: str = "Cosmological Expansion — Hubble Sphere & Event Horizons",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualize cosmological expansion: comoving worldlines, the Hubble sphere,
    and the cosmological event horizon.

    Parameters
    ----------
    scale_factor_fn : callable, optional
        Function t -> a(t). Defaults to ΛCDM-like: a(t) = (t/t0)^(2/3) * exp(Λ*t).
    t_max : float
        Time range in Gyr.
    n_worldlines : int
        Number of comoving worldlines (galaxies) to draw.
    title : str
        Plot title.
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    t = np.linspace(0.1, t_max, 1000)
    t0 = t_max

    if scale_factor_fn is None:
        # ΛCDM-like: matter-dominated then accelerated expansion
        def scale_factor_fn(t_arr: np.ndarray) -> np.ndarray:
            a_matter = (t_arr / t0) ** (2 / 3)
            a_lambda = np.exp(0.04 * (t_arr - t0))
            return a_matter * a_lambda

    a = scale_factor_fn(t)
    a = a / a[-1]  # normalize to a(t0) = 1

    # Hubble parameter H(t) = da/dt / a
    da_dt = np.gradient(a, t)
    H = da_dt / a
    # Hubble sphere: proper distance d_H = c/H (in units where c=1)
    d_hubble = 1.0 / np.where(H > 1e-6, H, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    _apply_dark_style(fig, axes)

    # --- Left panel: comoving worldlines ---
    ax1 = axes[0]
    comoving_positions = np.linspace(-3, 3, n_worldlines)
    colors_wl = cm.coolwarm(np.linspace(0, 1, n_worldlines))

    for chi, color in zip(comoving_positions, colors_wl):
        proper_d = chi * a
        ax1.plot(proper_d, t, color=color, lw=1.5, alpha=0.85)

    # Hubble sphere
    ax1.plot(d_hubble, t, color=_ACCENT, lw=2.5, ls="--", label="Hubble sphere (c/H)")
    ax1.plot(-d_hubble, t, color=_ACCENT, lw=2.5, ls="--")

    # Our worldline
    ax1.axvline(0, color=_ACCENT3, lw=2, label="Our worldline")

    ax1.set_xlabel("Proper distance (arbitrary units)")
    ax1.set_ylabel("Time (Gyr)")
    ax1.set_title("Comoving Worldlines")
    ax1.set_xlim(-5, 5)
    ax1.legend(fontsize=8, labelcolor=_TEXT_COLOR,
                facecolor=_DARK_BG, edgecolor=_GRID_COLOR)
    ax1.grid(True, color=_GRID_COLOR, lw=0.4)

    # --- Right panel: scale factor & Hubble parameter ---
    ax2 = axes[1]
    ax2.plot(t, a, color=_ACCENT, lw=2.5, label="Scale factor a(t)")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(t, H, color=_ACCENT2, lw=2.0, ls=":", label="H(t) Hubble param")
    ax2_twin.set_ylabel("H(t)", color=_ACCENT2)
    ax2_twin.tick_params(axis="y", colors=_ACCENT2)
    ax2_twin.set_facecolor(_DARK_BG)

    ax2.set_xlabel("Time (Gyr)")
    ax2.set_ylabel("a(t)", color=_ACCENT)
    ax2.tick_params(axis="y", colors=_ACCENT)
    ax2.set_title("Scale Factor & Hubble Parameter")
    ax2.grid(True, color=_GRID_COLOR, lw=0.4)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
               labelcolor=_TEXT_COLOR, facecolor=_DARK_BG, edgecolor=_GRID_COLOR)

    plt.suptitle(title, color=_TEXT_COLOR, fontsize=12, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    return fig


# ===========================================================================
# 6. Scalar field excitation (H3: matter as excitation of space)
# ===========================================================================

def plot_field_excitation(
    psi_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    x_range: tuple[float, float] = (-5.0, 5.0),
    y_range: tuple[float, float] = (-5.0, 5.0),
    n_points: int = 300,
    title: str = "Scalar Field Excitation — H3: Matter as Local Excitation of Space",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualize a scalar field ψ(x, y) representing a localized excitation
    of space, illustrating Hypothesis H3: matter as stable excitation.

    Default: superposition of a stable soliton-like excitation + vacuum
    fluctuation noise, showing how a localized "particle" emerges from
    the underlying field.

    Parameters
    ----------
    psi_fn : callable, optional
        Function (X_grid, Y_grid) -> psi_grid. Defaults to a Gaussian
        soliton with vacuum fluctuations.
    x_range, y_range : tuple
        Spatial extent.
    n_points : int
        Resolution.
    title : str
        Plot title.
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x = np.linspace(*x_range, n_points)
    y = np.linspace(*y_range, n_points)
    X, Y = np.meshgrid(x, y)

    if psi_fn is None:
        # Default: stable Gaussian excitation (particle) + vacuum noise
        rng = np.random.default_rng(42)

        def psi_fn(Xg: np.ndarray, Yg: np.ndarray) -> np.ndarray:
            # Primary excitation (the "particle")
            r2 = Xg**2 + Yg**2
            particle = 2.0 * np.exp(-r2 / 0.8)
            # Secondary excitation (virtual particle / fluctuation)
            r2b = (Xg - 2.5)**2 + (Yg + 1.5)**2
            virtual = 0.4 * np.exp(-r2b / 0.3)
            # Vacuum energy density (zero-point fluctuations, H1)
            vacuum = 0.08 * rng.standard_normal(Xg.shape)
            return particle + virtual + vacuum

    psi = psi_fn(X, Y)
    energy_density = psi**2  # |ψ|² ~ energy density

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _apply_dark_style(fig, axes)

    # --- Left: field amplitude ψ ---
    ax1 = axes[0]
    im1 = ax1.imshow(
        psi,
        extent=[*x_range, *y_range],
        origin="lower",
        cmap="seismic",
        vmin=-np.max(np.abs(psi)),
        vmax=np.max(np.abs(psi)),
        aspect="auto",
    )
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label("ψ(x,y)  field amplitude", color=_TEXT_COLOR)
    cbar1.ax.yaxis.set_tick_params(color=_TEXT_COLOR)
    plt.setp(cbar1.ax.yaxis.get_ticklabels(), color=_TEXT_COLOR)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Field Amplitude ψ(x,y)")
    ax1.contour(X, Y, psi, levels=8, colors="white", alpha=0.2, linewidths=0.5)

    # --- Right: energy density |ψ|² ---
    ax2 = axes[1]
    im2 = ax2.imshow(
        energy_density,
        extent=[*x_range, *y_range],
        origin="lower",
        cmap="inferno",
        aspect="auto",
    )
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label("|ψ|²  energy density", color=_TEXT_COLOR)
    cbar2.ax.yaxis.set_tick_params(color=_TEXT_COLOR)
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color=_TEXT_COLOR)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Energy Density |ψ|²(x,y)")

    # Annotate the "particle" peak
    peak_idx = np.unravel_index(np.argmax(energy_density), energy_density.shape)
    peak_x = x[peak_idx[1]]
    peak_y = y[peak_idx[0]]
    ax2.annotate(
        "stable\nexcitation\n(particle)",
        xy=(peak_x, peak_y),
        xytext=(peak_x + 1.5, peak_y + 1.5),
        color=_ACCENT,
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=_ACCENT, lw=1.5),
    )

    plt.suptitle(title, color=_TEXT_COLOR, fontsize=11, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    return fig
