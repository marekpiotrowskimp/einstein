# Einstein — Tools for Exploring Unified Physics Hypotheses

## Motivation

Modern physics rests on two pillars — General Relativity (GR) and Quantum Field Theory (QFT) — that are mutually incompatible at the deepest level. Current models may represent a local minimum of hypotheses, artificially separating concepts that could be manifestations of a single underlying structure:

- **space**
- **energy**
- **matter**
- **time**
- **gravity**
- **quantum fields**

This project builds computational tools to explore, simulate, and test hypotheses toward a more fundamental unified theory.

---

## Core Hypotheses Under Investigation

### H1 — Space is not an empty container
Space is an active medium with physical properties. The quantum vacuum is not "nothing" — it contains zero-point energy, quantum fluctuations, and spontaneous particle-antiparticle pair creation.

### H2 — Space and energy are manifestations of the same fundamental structure
Rather than energy existing *in* space, space and energy may be two descriptions of the same underlying entity viewed at different levels of abstraction.

### H3 — Matter is a stable local excitation of space
Particles are not fundamental objects placed into space. They are localized, stable configurations of the fundamental space-energy structure. Annihilation (e⁻ + e⁺ → γ + γ) supports this: particle identity is destroyed, but energy remains — matter was a state of fields, not a fundamental thing.

### H4 — Gravity is emergent, not fundamental
Gravity is not a force but an emergent effect of gradients and curvature in the fundamental structure (consistent with GR, but potentially derivable from deeper principles — cf. Verlinde's entropic gravity).

### H5 — Time is emergent
Time may not be a fundamental dimension but an emergent consequence of changes in the space-energy configuration. The "flow" of time reflects the evolution of the underlying structure.

### H6 — The expansion of the universe is a dynamic transformation of the fundamental structure
Cosmological expansion is not matter moving through pre-existing space, but the geometry itself evolving — potentially a phase transition or large-scale dynamics of the fundamental structure.

---

## Phenomena to Explain

A successful theory built on these hypotheses must account for:

| Phenomenon | Current explanation | Open questions |
|---|---|---|
| Speed of light limit *c* | SR postulate | Why is *c* finite? What sets its value? |
| Time dilation | SR/GR metric | Emergent from what? |
| Particle creation/annihilation | QFT | Why does structure crystallize into stable excitations? |
| Dark energy | Cosmological constant Λ | Why is vacuum energy ~120 orders of magnitude off? |
| Event horizons | GR | Information paradox |
| Origin of mass | Higgs mechanism | Why these masses? |
| Quantum discreteness | QM postulate | Why quantization? |
| Quantum vacuum energy | QFT | Cosmological constant problem |
| Spacetime singularities | GR breakdown | Need quantum gravity |

---

## Module Overview

### `einstein.geometry` — Algebraic Tools ✓
Symbolic tensor calculus via SymPy.

| Class / function | Description |
|---|---|
| `Manifold(coords, metric)` | Core manifold with lazy-cached tensors |
| `.metric_inv` | Contravariant metric g^μν |
| `.christoffel` | Christoffel symbols Γ^λ_μν |
| `.riemann` | Riemann tensor R^ρ_σμν |
| `.ricci_tensor` / `.ricci_scalar` | R_μν, R |
| `.einstein_tensor` | G_μν = R_μν − ½ g_μν R |
| `.field_equations(T, Λ)` | Residual of G_μν + Λg_μν = 8πG/c⁴ T_μν |
| `.is_flat()` | Test: all Riemann components = 0 |
| `.latex_metric()` | LaTeX export |

Validated on: **Minkowski** (flat), **Schwarzschild** (vacuum), **S²** (R=2) — 18 tests passing.

### `einstein.visualization` — Geometric Visualizations ✓
Matplotlib-based rendering of curved spacetime geometry.

| Function | Description |
|---|---|
| `plot_embedding_diagram(manifold)` | 2D/3D embedding of curved surface |
| `plot_geodesics(manifold, initial_conditions)` | Geodesic trajectories on manifold |
| `plot_curvature_heatmap(manifold)` | Ricci scalar R as heatmap |
| `plot_schwarzschild_spacetime()` | Schwarzschild light cones + horizons |
| `plot_expansion(scale_factor_fn)` | Cosmological expansion / Hubble sphere |
| `plot_field_excitation(psi_fn)` | Scalar field excitation profile (H3) |

### `einstein.simulation` — Numerical Simulations ✓
Leapfrog (Störmer-Verlet) integration — symplectic, energy-stable.

| Function | Equation | Key features |
|---|---|---|
| `simulate_kg_1d(...)` | □φ − m²φ = 0 (1+1D) | Periodic / absorbing BC, CFL check |
| `simulate_kg_2d(...)` | □φ − m²φ = 0 (2+1D) | 2D wave propagation |
| `simulate_phi4_1d(...)` | ∂²φ/∂t² = ∂²φ/∂x² − λφ(φ²−v²) | Kink solitons, kink-antikink collision |
| `simulate_kg_curved_1d(...)` | KG on Schwarzschild (tortoise coords) | Quasinormal modes, Price's law tails |

All return `SimulationResult(phi, pi, energy, times, metadata)`.

Physical phenomena modeled:
- **Wave dispersion** — massless (m=0) vs massive (m>0) Klein-Gordon
- **φ⁴ kink solitons** — topologically stable excitations (H3: matter as excitation)
- **Kink-antikink collision** — particle-like scattering and annihilation
- **Quasinormal modes** — characteristic ringing of Schwarzschild black holes
- **Greybody factors** — partial absorption by the Regge-Wheeler potential barrier

### `einstein.hypotheses` — Hypothesis Testing Framework ✓
Formal specification, numerical evaluation, and consistency analysis of H1–H6.

| Module | Description |
|---|---|
| `specification.py` | `Hypothesis` dataclass, `HypothesisRegistry`, status enum, theory relations |
| `predictions.py` | 14 computable evaluators (H1–H6): Casimir, Verlinde, kink mass, BH entropy, ΛCDM… |
| `consistency.py` | `ConsistencyChecker` — 19 cross-checks (hypothesis ↔ GR/QFT/SR/QM/ΛCDM) |
| `observational.py` | `ObservationalChecker` — 11 observational constraints vs CODATA/Planck 2018 |
| `report.py` | `generate_report()` → `outputs/hypothesis_report.md` |

Current status:

| Hypothesis | Status | Notes |
|---|---|---|
| H1 — Active vacuum | ✅ supported | Casimir, zero-point energy, CC problem quantified |
| H2 — Space=Energy | ✅ supported | Planck units, E·λ = hc duality |
| H3 — Matter=excitation | ✅ supported | Kink mass, width, topological charge |
| H4 — Emergent gravity | 🟡 partial | Verlinde Newton + BH entropy; graviton conflict |
| H5 — Emergent time | ✅ supported | GR + SR time dilation, Planck time |
| H6 — Dynamic expansion | 🟡 partial | Hubble sphere, z_eq, comoving distance |

### `einstein.llm` — LLM-Assisted Hypothesis Exploration (planned)
- Structured prompting frameworks for hypothesis generation
- Consistency checking of new theoretical constructs
- Mapping hypotheses to existing literature (LQG, String Theory, CDT, AdS/CFT)

### `einstein.observation` — Observational Data Analysis (planned)
- CMB data processing
- Large-scale structure analysis
- Testing theoretical predictions against astronomical observations

---

## Relationship to Existing Theories

| Theory | Overlap with project hypotheses |
|---|---|
| General Relativity | Space as dynamic geometry (H1, H4) |
| Quantum Field Theory | Matter as field excitations (H3), active vacuum (H1) |
| Loop Quantum Gravity | Discrete quantum structure of space (H1, H2) |
| String Theory | Fundamental excitations as matter (H3) |
| Causal Dynamical Triangulations | Emergent spacetime (H2, H5) |
| Entropic Gravity (Verlinde) | Gravity as emergent phenomenon (H4) |
| AdS/CFT Correspondence | Emergent gravity from field theory (H4) |

The goal is not to replace these programs but to build tools that can probe the hypothesis space they share and explore the unifying structure beneath them.

---

## Roadmap

```
Phase 1 — Foundations
  [x] Project context and hypotheses documentation
  [x] Tech stack: Python 3.14 + SymPy + NumPy + SciPy + Matplotlib
  [x] Core tensor/geometry utilities (einstein/geometry/manifold.py)

Phase 2 — Algebraic Tools
  [x] Metric tensor + inverse (g_μν, g^μν)
  [x] Christoffel symbols Γ^λ_μν
  [x] Riemann tensor R^ρ_σμν
  [x] Ricci tensor R_μν and scalar R
  [x] Einstein tensor G_μν
  [x] Einstein field equations residual check
  [x] Validated on: Minkowski, Schwarzschild, S² (18 tests passing)
  [ ] Einstein field equations solver (symbolic)

Phase 3 — Simulations
  [x] Klein-Gordon 1D (flat spacetime, leapfrog, periodic/absorbing BC)
  [x] Klein-Gordon 2D (flat spacetime)
  [x] φ⁴ kink solitons — matter-as-excitation model (H3)
  [x] Kink-antikink collision (particle scattering analog)
  [x] KG on Schwarzschild background (tortoise coords, l=0,1,2...)
  [x] Quasinormal modes, energy decay, Price's law tails
  [x] SimulationResult container with energy tracking
  [ ] Tensor field evolution
  [ ] Emergent geometry from discrete structures

Phase 4 — Visualizations
  [x] Embedding diagrams (2D/3D curved surfaces)
  [x] Geodesic trajectories
  [x] Ricci scalar curvature heatmap
  [x] Schwarzschild spacetime — light cones, event horizon
  [x] Cosmological expansion — Hubble sphere, event horizon
  [x] Scalar field excitation profile (H3: matter-as-excitation)

Phase 5 — Hypothesis Testing
  [x] Formal specification of H1–H6 (Hypothesis dataclass + HypothesisRegistry)
  [x] Derived predictions — 14 computable evaluators
  [x] Cross-consistency analysis (19 hypothesis ↔ theory checks)
  [x] Comparison with observational constraints (11 tests vs CODATA/Planck 2018)
  [x] Auto-generated Markdown report (outputs/hypothesis_report.md)
  [x] 68 unit tests — 135 total passing

Phase 6 — LLM Integration
  [x] LLM client abstraction (Mock, Copilot, Ollama) + get_client() factory
  [x] Prompt templates (consistency, validation, literature, bibliography)
  [x] Pairwise hypothesis consistency checker (15 pairs, full matrix + summary)
  [x] New hypothesis validator (verdict, novelty, suggestions)
  [x] Literature mapping tool (TheoryMapping per hypothesis)
  [x] Bibliography generator (BibTeX + Markdown export)
  [x] Report integration (generate_report(include_llm=True))
  [x] 89 unit tests — 224 total passing

Phase 7 — Interactive Exploration (Jupyter Notebook)
  [x] Unified demonstration notebook (notebooks/einstein_exploration.ipynb)
  [x] Phase 2 walkthrough — symbolic tensor calculus on Minkowski/Schwarzschild/S²
  [x] Phase 3 walkthrough — scalar field simulations with energy plots
  [x] Phase 4 walkthrough — geometry visualizations with dark theme
  [x] Phase 5 walkthrough — hypothesis evaluation, consistency matrix, observational checks
  [x] Phase 6 walkthrough — LLM consistency checker + literature mapper (Mock client)
  [x] End-to-end: generate full report (Markdown + BibTeX) from notebook
  [x] nbmake integration for CI notebook testing
```

---

## Key References

- Einstein, A. (1915). *Die Feldgleichungen der Gravitation*
- Weinberg, S. (1995). *The Quantum Theory of Fields*
- Rovelli, C. (2004). *Quantum Gravity* — Loop Quantum Gravity
- Verlinde, E. (2011). *On the Origin of Gravity and the Laws of Newton*
- Penrose, R. (2004). *The Road to Reality*
- Carroll, S. (2019). *Something Deeply Hidden* — quantum fields
- Susskind, L. & Lindesay, J. (2005). *The Holographic Universe*

---

## Project Status

> **Early exploration phase.** Hypotheses are working assumptions, not established theory.  
> The goal is to build tools rigorous enough to find contradictions, constraints, and — potentially — new predictions.
