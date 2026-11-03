"""
einstein.hypotheses
====================
Formal hypothesis testing framework for the six core hypotheses H1–H6.

Quick start
-----------
from einstein.hypotheses import build_registry

registry = build_registry()
for h in registry:
    print(h.id, h.status)

# Generate full report
from einstein.hypotheses.report import generate_report
generate_report(output_path="outputs/hypothesis_report.md")
"""

from __future__ import annotations

from einstein.hypotheses.specification import (
    Hypothesis,
    HypothesisRegistry,
    HypothesisStatus,
    ConsistencyVerdict,
    TheoryRelation,
    Prediction,
    FalsificationCondition,
)
import einstein.hypotheses.predictions as P


# ---------------------------------------------------------------------------
# Build the full hypothesis registry
# ---------------------------------------------------------------------------

def build_registry() -> HypothesisRegistry:
    """
    Construct and return a HypothesisRegistry with all six hypotheses.

    Each hypothesis has:
    - Full textual specification
    - Computable predictions linked to evaluator functions in predictions.py
    - Falsification conditions
    - Theory relations
    """

    # ── H1 — Space is an active medium ────────────────────────────────────
    h1 = Hypothesis(
        id="H1",
        name="Space is an active medium (quantum vacuum)",
        statement=(
            "Space is not an empty container but an active physical medium. "
            "The quantum vacuum has measurable energy density, supports zero-point "
            "fluctuations, and can produce spontaneous particle-antiparticle pairs. "
            "The vacuum is the ground state of all quantum fields, not 'nothing'."
        ),
        assumptions=[
            "Quantum mechanics applies at all length scales",
            "Fields are the fundamental ontology (QFT picture)",
            "The vacuum state |0⟩ has non-zero energy ⟨0|H|0⟩ ≠ 0",
        ],
        predictions=[
            Prediction(
                id="H1-P1",
                description="QFT vacuum energy density >> observed Λ (cosmological constant problem)",
                computable=True,
                evaluator=P.h1_p1_vacuum_energy_density,
                references=["Weinberg 1989 — cosmological constant review"],
            ),
            Prediction(
                id="H1-P2",
                description="Zero-point energy E₀ = ½ħω for each field mode",
                computable=True,
                evaluator=P.h1_p2_zero_point_energy_harmonic_oscillator,
                references=["Dirac 1930 — quantum oscillator"],
            ),
            Prediction(
                id="H1-P3",
                description="Casimir force between parallel plates (attractive, F/A = −π²ħc/240d⁴)",
                computable=True,
                evaluator=P.h1_p3_casimir_force_sign,
                references=["Casimir 1948", "Lamoreaux 1997"],
            ),
        ],
        falsification_conditions=[
            FalsificationCondition(
                description="Casimir force measured to be zero or repulsive for uncharged plates in vacuum",
                evaluator=None,
            ),
            FalsificationCondition(
                description="Zero-point energy experimentally shown to not contribute to physical observables",
                evaluator=None,
            ),
        ],
        theory_relations=[
            TheoryRelation("QFT",              ConsistencyVerdict.CONSISTENT, "H1 is the QFT vacuum picture"),
            TheoryRelation("GR",               ConsistencyVerdict.TENSION,    "Cosmological constant problem"),
            TheoryRelation("Cosmology (ΛCDM)", ConsistencyVerdict.TENSION,    "Λ not derivable from vacuum"),
            TheoryRelation("SR",               ConsistencyVerdict.CONSISTENT, "Lorentz-invariant vacuum"),
        ],
        notes=(
            "The cosmological constant problem (ρ_QFT / ρ_obs ≈ 10^120) is the most severe "
            "fine-tuning problem in physics. H1 does not resolve it — it quantifies it as "
            "the central challenge any unified theory must address."
        ),
    )

    # ── H2 — Space and energy are the same structure ──────────────────────
    h2 = Hypothesis(
        id="H2",
        name="Space and energy are manifestations of the same fundamental structure",
        statement=(
            "Space and energy are not distinct substances — they are two descriptions "
            "of the same underlying entity viewed at different levels of abstraction. "
            "Energy does not exist 'in' space; rather, what we call 'space' and what "
            "we call 'energy' are aspects of a single primordial structure."
        ),
        assumptions=[
            "There exists a more fundamental level of description than fields-in-spacetime",
            "The Planck scale is the natural scale of unification",
            "Dimensional analysis in Planck units reveals the unification",
        ],
        predictions=[
            Prediction(
                id="H2-P1",
                description="Planck units unify [length], [time], [mass], [energy] (dimensional consistency)",
                computable=True,
                evaluator=P.h2_p1_natural_units_dimensional_consistency,
                references=["Planck 1899 — natural units"],
            ),
            Prediction(
                id="H2-P2",
                description="Energy-wavelength duality: E·λ = 2πħc (space-energy trade-off)",
                computable=True,
                evaluator=P.h2_p2_energy_length_trade_off,
                references=["Einstein 1905 — photoelectric effect", "de Broglie 1924"],
            ),
        ],
        falsification_conditions=[
            FalsificationCondition(
                description="Discovery of a particle or field that carries energy but has no spatial extent whatsoever",
                evaluator=None,
            ),
            FalsificationCondition(
                description="Demonstration that space can exist completely independently of any energy content",
                evaluator=None,
            ),
        ],
        theory_relations=[
            TheoryRelation("GR",  ConsistencyVerdict.CONSISTENT, "GR: metric = dynamical field carrying energy"),
            TheoryRelation("QFT", ConsistencyVerdict.CONSISTENT, "Fields as fundamental, particles as excitations"),
            TheoryRelation("SR",  ConsistencyVerdict.CONSISTENT, "E=mc², E·λ=hc — space-energy duality"),
            TheoryRelation("LQG", ConsistencyVerdict.CONSISTENT, "Space as quantum geometry = quantum energy"),
        ],
        notes=(
            "H2 is currently qualitative. A falsifiable quantitative form requires specifying "
            "the mathematical structure that unifies the metric and the stress-energy tensor "
            "into a single object. Candidate: the Einstein field equations G_μν = 8πG T_μν "
            "already suggest this, but the left side (geometry) and right side (energy) "
            "are treated as separate inputs."
        ),
    )

    # ── H3 — Matter is a stable excitation ────────────────────────────────
    h3 = Hypothesis(
        id="H3",
        name="Matter is a stable local excitation of space",
        statement=(
            "Particles are not fundamental objects placed into space. They are localized, "
            "stable configurations (excitations) of the fundamental space-energy field. "
            "Particle identity can be destroyed (e⁻+e⁺→γγ) because a particle is a state "
            "of the field, not a fundamental thing. Topology guarantees stability of certain excitations."
        ),
        assumptions=[
            "Fields are the fundamental ontology",
            "Topological stability is the mechanism for particle-like persistence",
            "The φ⁴ kink is a valid classical analog of a stable excitation",
        ],
        predictions=[
            Prediction(
                id="H3-P1",
                description="φ⁴ kink mass: M = (4/3)v³√(λ/2) ≈ 0.9428 for λ=v=1",
                computable=True,
                evaluator=P.h3_p1_kink_mass_formula,
                references=["Rajaraman 1982 — solitons and instantons"],
            ),
            Prediction(
                id="H3-P2",
                description="φ⁴ kink width: ξ = 1/(v√(λ/2)) = √2 for λ=v=1",
                computable=True,
                evaluator=P.h3_p2_kink_width,
                references=["Rajaraman 1982"],
            ),
            Prediction(
                id="H3-P3",
                description="Topological charge Q = 1 (discrete, quantized — analog of particle number)",
                computable=True,
                evaluator=P.h3_p3_kink_topological_charge,
                references=["Rajaraman 1982"],
            ),
            Prediction(
                id="H3-P4",
                description="φ⁴ potential has two degenerate minima V(±v)=0 — required for topological stability",
                computable=True,
                evaluator=P.h3_p4_kink_potential_minima,
                references=["Rajaraman 1982"],
            ),
        ],
        falsification_conditions=[
            FalsificationCondition(
                description="Discovery of a truly elementary point-like particle with no field-theoretic description",
                evaluator=None,
            ),
            FalsificationCondition(
                description="Experimental confirmation that an electron cannot be converted to photons under any conditions",
                evaluator=None,
            ),
        ],
        theory_relations=[
            TheoryRelation("QFT",            ConsistencyVerdict.CONSISTENT, "QFT IS matter as field excitations"),
            TheoryRelation("GR",             ConsistencyVerdict.CONSISTENT, "T_μν couples solitonic matter to geometry"),
            TheoryRelation("Standard Model", ConsistencyVerdict.TENSION,    "SM needs spin, charge, colour — kink has none"),
            TheoryRelation("SR",             ConsistencyVerdict.CONSISTENT, "Lorentz-boosted kink is a valid soliton"),
        ],
        notes=(
            "The φ⁴ kink captures topological stability and discreteness but lacks "
            "internal degrees of freedom (spin, charge, colour). Extension to Dirac solitons "
            "and gauge vortices is required to connect to the Standard Model particle content."
        ),
    )

    # ── H4 — Gravity is emergent ───────────────────────────────────────────
    h4 = Hypothesis(
        id="H4",
        name="Gravity is emergent, not fundamental",
        statement=(
            "Gravity is not a fundamental force but an emergent thermodynamic effect "
            "arising from entropy gradients on holographic screens. "
            "Newton's law and Einstein's field equations can be derived from "
            "the Clausius relation δQ=TdS applied to causal horizons (Jacobson 1995) "
            "or from holographic entropy (Verlinde 2011). "
            "The graviton is not a fundamental particle — gravitational interaction "
            "is statistical, like pressure or temperature."
        ),
        assumptions=[
            "The holographic principle holds: S ∝ A for any causal horizon",
            "Unruh temperature T_U = ħa/(2πk_B c) is physical",
            "Equipartition holds for holographic degrees of freedom",
        ],
        predictions=[
            Prediction(
                id="H4-P1",
                description="Verlinde entropic gravity reproduces F = GMm/r² (Newton's law)",
                computable=True,
                evaluator=P.h4_p1_verlinde_newton_force,
                references=["Verlinde 2011"],
            ),
            Prediction(
                id="H4-P2",
                description="Bekenstein-Hawking entropy S = πk_B(r_s/L_P)² — area law holds",
                computable=True,
                evaluator=P.h4_p2_gravitational_entropy_area_law,
                references=["Bekenstein 1973", "Hawking 1975"],
            ),
        ],
        falsification_conditions=[
            FalsificationCondition(
                description="Detection of a graviton as a free asymptotic particle state",
                evaluator=None,
            ),
            FalsificationCondition(
                description="Breakdown of the area law S∝A for black hole entropy",
                evaluator=None,
            ),
        ],
        theory_relations=[
            TheoryRelation("GR",             ConsistencyVerdict.SUBSUMES,  "Jacobson derives EFE from thermodynamics"),
            TheoryRelation("QFT",            ConsistencyVerdict.TENSION,   "QFT graviton picture conflicts with emergence"),
            TheoryRelation("AdS/CFT",        ConsistencyVerdict.CONSISTENT,"Gravity emergent from boundary CFT"),
            TheoryRelation("Thermodynamics", ConsistencyVerdict.CONSISTENT,"Gravity = thermodynamics of horizons"),
        ],
        notes=(
            "Jacobson (1995) derives the full Einstein equation G_μν + Λg_μν = 8πG T_μν "
            "from the first law of thermodynamics for local Rindler horizons. "
            "Verlinde (2011) derives Newton's law from entropic forces. "
            "The extension to non-linear GR and the derivation of Λ remain open."
        ),
    )

    # ── H5 — Time is emergent ─────────────────────────────────────────────
    h5 = Hypothesis(
        id="H5",
        name="Time is emergent",
        statement=(
            "Time is not a fundamental dimension of reality. It emerges from the "
            "evolution of the underlying space-energy configuration. "
            "The 'flow' of time reflects the rate of change of the fundamental structure. "
            "Gravitational and kinematic time dilation are consequences of how "
            "different configurations evolve at different rates."
        ),
        assumptions=[
            "There exists a configuration space whose evolution defines 'time'",
            "At the Planck scale, time is discrete with quantum t_P",
            "Lorentz invariance is preserved as a symmetry of configuration dynamics",
        ],
        predictions=[
            Prediction(
                id="H5-P1",
                description="Gravitational time dilation: dτ/dt = √(1−r_s/r) at r=2r_s → 1/√2",
                computable=True,
                evaluator=P.h5_p1_time_dilation_schwarzschild,
                references=["Schwarzschild 1916", "Pound & Rebka 1959"],
            ),
            Prediction(
                id="H5-P2",
                description="Planck time t_P = √(ħG/c⁵) ≈ 5.391e-44 s as minimal time quantum",
                computable=True,
                evaluator=P.h5_p2_planck_time_as_minimum_interval,
                references=["Planck 1899"],
            ),
            Prediction(
                id="H5-P3",
                description="SR time dilation: dτ/dt = √(1−v²/c²) at v=0.9c → √0.19",
                computable=True,
                evaluator=P.h5_p3_special_relativistic_time_dilation,
                references=["Einstein 1905 — special relativity"],
            ),
        ],
        falsification_conditions=[
            FalsificationCondition(
                description="Experimental proof that time flows uniformly regardless of energy configuration",
                evaluator=None,
            ),
            FalsificationCondition(
                description="Violation of gravitational time dilation (Shapiro delay, GPS corrections fail)",
                evaluator=None,
            ),
        ],
        theory_relations=[
            TheoryRelation("SR",  ConsistencyVerdict.CONSISTENT, "SR time dilation reinterpreted as config evolution"),
            TheoryRelation("GR",  ConsistencyVerdict.CONSISTENT, "GR time dilation is geometrical manifestation of H5"),
            TheoryRelation("QM",  ConsistencyVerdict.TENSION,    "QM treats t as external parameter, not emergent"),
            TheoryRelation("LQG", ConsistencyVerdict.CONSISTENT, "LQG also has emergent / relational time"),
        ],
        notes=(
            "The Page-Wootters mechanism (1983) provides a concrete QM realisation: "
            "time emerges from entanglement between a 'clock' subsystem and the rest of the universe. "
            "This is consistent with H5 but requires extension to the field-theoretic setting."
        ),
    )

    # ── H6 — Expansion is dynamic geometry ────────────────────────────────
    h6 = Hypothesis(
        id="H6",
        name="The expansion of the universe is a dynamic transformation of the fundamental structure",
        statement=(
            "Cosmological expansion is not matter moving through pre-existing space. "
            "It is the geometry itself evolving — the fundamental structure undergoing "
            "a large-scale phase transition or dynamical reconfiguration. "
            "The scale factor a(t) encodes not just distances but the state of the "
            "fundamental structure at cosmological scales."
        ),
        assumptions=[
            "The FLRW metric is the correct large-scale description",
            "Ω_m ≈ 0.315, Ω_Λ ≈ 0.685 (Planck 2018)",
            "H₀ ≈ 67.4 km/s/Mpc (CMB-based)",
        ],
        predictions=[
            Prediction(
                id="H6-P1",
                description="Hubble sphere radius r_H = c/H₀ ≈ 1.374e26 m",
                computable=True,
                evaluator=P.h6_p1_hubble_sphere_radius,
                references=["Planck 2018"],
            ),
            Prediction(
                id="H6-P2",
                description="Deceleration→acceleration transition at z_eq ≈ 0.32 (Ω_m/Ω_Λ ratio)",
                computable=True,
                evaluator=P.h6_p2_scale_factor_deceleration_to_acceleration,
                references=["Riess et al. 1998", "Perlmutter et al. 1999"],
            ),
            Prediction(
                id="H6-P3",
                description="Comoving distance to z=1: χ ≈ 1.018e26 m (numerical ΛCDM integral)",
                computable=True,
                evaluator=P.h6_p3_comoving_distance_integral,
                references=["Planck 2018 — cosmological distances"],
            ),
        ],
        falsification_conditions=[
            FalsificationCondition(
                description="Observation of space with zero expansion but non-zero energy content everywhere",
                evaluator=None,
            ),
            FalsificationCondition(
                description="CMB anisotropies inconsistent with FLRW geometry at > 5σ",
                evaluator=None,
            ),
        ],
        theory_relations=[
            TheoryRelation("GR",               ConsistencyVerdict.CONSISTENT, "FLRW is exact GR solution"),
            TheoryRelation("Cosmology (ΛCDM)", ConsistencyVerdict.CONSISTENT, "H6 extends ΛCDM with physical interpretation"),
            TheoryRelation("QFT",              ConsistencyVerdict.TENSION,    "Cosmological constant problem unresolved"),
            TheoryRelation("Inflation",        ConsistencyVerdict.CONSISTENT, "Inflation is a phase transition of the early structure"),
        ],
        notes=(
            "H6 adds a physical interpretation to FLRW cosmology: expansion is a "
            "reconfiguration of the fundamental structure, not merely 'space stretching'. "
            "This framing naturally connects to phase transitions and could explain "
            "the origin of the inflationary epoch and dark energy."
        ),
    )

    return HypothesisRegistry([h1, h2, h3, h4, h5, h6])


__all__ = [
    "build_registry",
    "Hypothesis",
    "HypothesisRegistry",
    "HypothesisStatus",
    "ConsistencyVerdict",
    "TheoryRelation",
    "Prediction",
    "FalsificationCondition",
]
