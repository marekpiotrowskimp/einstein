"""
Tests for einstein.hypotheses — Phase 5 hypothesis testing framework.

Tests cover:
- Specification: Hypothesis dataclass, HypothesisRegistry
- Predictions: all evaluators return (passed, value, expected, notes)
- Consistency: ConsistencyChecker produces correct verdicts
- Observational: ObservationalChecker matches known constraints
- Report: generate_report() produces valid Markdown
"""

from __future__ import annotations

import math
import pytest

from einstein.hypotheses import build_registry
from einstein.hypotheses.specification import (
    Hypothesis, HypothesisStatus, ConsistencyVerdict, HypothesisRegistry,
    Prediction, FalsificationCondition, TheoryRelation,
)
import einstein.hypotheses.predictions as P
from einstein.hypotheses.consistency import ConsistencyChecker, ConsistencyResult
from einstein.hypotheses.observational import ObservationalChecker, CONSTRAINTS
from einstein.hypotheses.report import generate_report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def registry():
    return build_registry()


@pytest.fixture(scope="module")
def consistency_checker():
    return ConsistencyChecker()


@pytest.fixture(scope="module")
def observational_checker():
    return ObservationalChecker()


# ---------------------------------------------------------------------------
# Registry and specification tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_has_six_hypotheses(self, registry):
        assert len(registry) == 6

    def test_all_ids_present(self, registry):
        ids = {h.id for h in registry}
        assert ids == {"H1", "H2", "H3", "H4", "H5", "H6"}

    def test_getitem(self, registry):
        h1 = registry["H1"]
        assert h1.id == "H1"

    def test_all_have_statements(self, registry):
        for h in registry:
            assert len(h.statement) > 20

    def test_all_have_assumptions(self, registry):
        for h in registry:
            assert len(h.assumptions) >= 2

    def test_all_have_predictions(self, registry):
        for h in registry:
            assert len(h.predictions) >= 1

    def test_all_have_theory_relations(self, registry):
        for h in registry:
            assert len(h.theory_relations) >= 2

    def test_all_have_falsification_conditions(self, registry):
        for h in registry:
            assert len(h.falsification_conditions) >= 1

    def test_summary_keys(self, registry):
        s = registry.summary()
        assert set(s.keys()) == {"H1", "H2", "H3", "H4", "H5", "H6"}
        for info in s.values():
            assert "name" in info
            assert "status" in info
            assert "n_predictions" in info
            assert "n_computable" in info

    def test_consistency_matrix_structure(self, registry):
        mat = registry.consistency_matrix()
        for hid, row in mat.items():
            assert hid in {"H1", "H2", "H3", "H4", "H5", "H6"}
            for theory, verdict in row.items():
                assert isinstance(verdict, str)

    def test_by_status(self, registry):
        supported = registry.by_status(HypothesisStatus.SUPPORTED)
        partial   = registry.by_status(HypothesisStatus.PARTIAL)
        falsified = registry.by_status(HypothesisStatus.FALSIFIED)
        total = len(supported) + len(partial) + len(falsified)
        assert total <= 6

    def test_no_falsified_hypotheses(self, registry):
        """All current hypotheses should be at least partial (none definitively falsified)."""
        for h in registry:
            assert h.status != HypothesisStatus.FALSIFIED


# ---------------------------------------------------------------------------
# Hypothesis status evaluation
# ---------------------------------------------------------------------------

class TestHypothesisStatus:
    def test_h1_status(self, registry):
        assert registry["H1"].status in (HypothesisStatus.SUPPORTED, HypothesisStatus.PARTIAL)

    def test_h2_status(self, registry):
        assert registry["H2"].status in (HypothesisStatus.SUPPORTED, HypothesisStatus.PARTIAL)

    def test_h3_status(self, registry):
        assert registry["H3"].status in (HypothesisStatus.SUPPORTED, HypothesisStatus.PARTIAL)

    def test_h4_status(self, registry):
        assert registry["H4"].status in (HypothesisStatus.SUPPORTED, HypothesisStatus.PARTIAL)

    def test_h5_status(self, registry):
        assert registry["H5"].status in (HypothesisStatus.SUPPORTED, HypothesisStatus.PARTIAL)

    def test_h6_status(self, registry):
        assert registry["H6"].status in (HypothesisStatus.SUPPORTED, HypothesisStatus.PARTIAL)

    def test_evaluate_predictions_returns_list(self, registry):
        for h in registry:
            results = h.evaluate_predictions()
            assert isinstance(results, list)
            for r in results:
                assert "id" in r
                assert "description" in r
                assert "passed" in r


# ---------------------------------------------------------------------------
# Prediction evaluators — H1
# ---------------------------------------------------------------------------

class TestH1Predictions:
    def test_vacuum_energy_ratio_above_100_orders(self):
        passed, log10_ratio, expected_str, notes = P.h1_p1_vacuum_energy_density()
        assert passed
        assert log10_ratio > 100
        assert "10^" in notes

    def test_zero_point_energy(self):
        passed, E0, E0_exp, notes = P.h1_p2_zero_point_energy_harmonic_oscillator()
        assert passed
        assert math.isclose(E0, E0_exp, rel_tol=1e-10)

    def test_casimir_force_is_negative(self):
        passed, F, expected_str, notes = P.h1_p3_casimir_force_sign()
        assert passed
        assert F < 0


# ---------------------------------------------------------------------------
# Prediction evaluators — H2
# ---------------------------------------------------------------------------

class TestH2Predictions:
    def test_planck_units_consistency(self):
        passed, rel_L, expected_str, notes = P.h2_p1_natural_units_dimensional_consistency()
        assert passed
        assert rel_L < 1e-6

    def test_energy_wavelength_duality(self):
        passed, product, expected, notes = P.h2_p2_energy_length_trade_off()
        assert passed
        assert math.isclose(product, expected, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# Prediction evaluators — H3
# ---------------------------------------------------------------------------

class TestH3Predictions:
    def test_kink_mass_formula(self):
        passed, M, M_exp, notes = P.h3_p1_kink_mass_formula()
        assert passed
        assert math.isclose(M, 4.0 / (3.0 * math.sqrt(2.0)), rel_tol=1e-10)

    def test_kink_width(self):
        passed, xi, xi_exp, notes = P.h3_p2_kink_width()
        assert passed
        assert math.isclose(xi, math.sqrt(2.0), rel_tol=1e-10)

    def test_topological_charge_equals_one(self):
        passed, Q, Q_exp, notes = P.h3_p3_kink_topological_charge()
        assert passed
        assert math.isclose(Q, 1.0, rel_tol=1e-10)

    def test_degenerate_minima(self):
        passed, V_zero, V_exp, notes = P.h3_p4_kink_potential_minima()
        assert passed

    def test_kink_mass_positive(self):
        _, M, *_ = P.h3_p1_kink_mass_formula()
        assert M > 0

    def test_kink_width_positive(self):
        _, xi, *_ = P.h3_p2_kink_width()
        assert xi > 0


# ---------------------------------------------------------------------------
# Prediction evaluators — H4
# ---------------------------------------------------------------------------

class TestH4Predictions:
    def test_verlinde_newton(self):
        passed, F_verlinde, F_newton, notes = P.h4_p1_verlinde_newton_force()
        assert passed
        assert F_verlinde > 0
        assert F_newton > 0
        assert abs(F_verlinde - F_newton) / F_newton < 1e-4

    def test_bh_entropy_area_law(self):
        passed, S_BH, S_exp, notes = P.h4_p2_gravitational_entropy_area_law()
        assert passed
        assert math.isclose(S_BH, S_exp, rel_tol=1e-6)
        assert S_BH > 0


# ---------------------------------------------------------------------------
# Prediction evaluators — H5
# ---------------------------------------------------------------------------

class TestH5Predictions:
    def test_gravitational_time_dilation(self):
        passed, dtau_dt, expected, notes = P.h5_p1_time_dilation_schwarzschild()
        assert passed
        assert math.isclose(dtau_dt, 1.0 / math.sqrt(2.0), rel_tol=1e-10)

    def test_planck_time(self):
        passed, t_P, t_P_known, notes = P.h5_p2_planck_time_as_minimum_interval()
        assert passed
        assert t_P < 1e-43

    def test_sr_time_dilation(self):
        passed, dtau_dt, expected, notes = P.h5_p3_special_relativistic_time_dilation()
        assert passed
        assert math.isclose(dtau_dt, math.sqrt(0.19), rel_tol=1e-10)

    def test_time_dilation_less_than_one(self):
        _, dtau_dt, *_ = P.h5_p1_time_dilation_schwarzschild()
        assert 0 < dtau_dt < 1


# ---------------------------------------------------------------------------
# Prediction evaluators — H6
# ---------------------------------------------------------------------------

class TestH6Predictions:
    def test_hubble_sphere(self):
        passed, r_H, r_known, notes = P.h6_p1_hubble_sphere_radius()
        assert passed
        assert abs(r_H - r_known) / r_known < 0.01

    def test_transition_redshift(self):
        passed, z_eq, z_exp, notes = P.h6_p2_scale_factor_deceleration_to_acceleration()
        assert passed
        assert 0.20 < z_eq < 0.45

    def test_comoving_distance(self):
        passed, chi, chi_exp, notes = P.h6_p3_comoving_distance_integral()
        assert passed
        assert abs(chi - chi_exp) / chi_exp < 0.02


# ---------------------------------------------------------------------------
# Consistency checker
# ---------------------------------------------------------------------------

class TestConsistencyChecker:
    def test_run_all_returns_results(self, consistency_checker):
        results = consistency_checker.run_all()
        assert len(results) > 10

    def test_all_results_have_verdict(self, consistency_checker):
        for r in consistency_checker.run_all():
            assert isinstance(r.verdict, ConsistencyVerdict)

    def test_no_conflicts(self, consistency_checker):
        """None of the current hypotheses should directly conflict with established theories."""
        conflicts = consistency_checker.conflicts()
        assert len(conflicts) == 0

    def test_some_tensions_exist(self, consistency_checker):
        """We expect at least some tensions (e.g. H1 vs GR cosmological constant problem)."""
        tensions = consistency_checker.tensions()
        assert len(tensions) >= 3

    def test_h1_vs_gr_is_tension(self, consistency_checker):
        results = consistency_checker.run_all()
        h1_gr = [r for r in results if r.hypothesis == "H1" and r.theory == "GR"]
        assert len(h1_gr) == 1
        assert h1_gr[0].verdict == ConsistencyVerdict.TENSION

    def test_h4_vs_gr_is_subsumes(self, consistency_checker):
        results = consistency_checker.run_all()
        h4_gr = [r for r in results if r.hypothesis == "H4" and r.theory == "GR"]
        assert len(h4_gr) == 1
        assert h4_gr[0].verdict == ConsistencyVerdict.SUBSUMES

    def test_h3_vs_sm_is_tension(self, consistency_checker):
        results = consistency_checker.run_all()
        h3_sm = [r for r in results if r.hypothesis == "H3" and r.theory == "Standard Model"]
        assert len(h3_sm) == 1
        assert h3_sm[0].verdict == ConsistencyVerdict.TENSION

    def test_summary_counts_non_empty(self, consistency_checker):
        counts = consistency_checker.summary_counts()
        assert sum(counts.values()) > 0
        assert "consistent" in counts

    def test_matrix_structure(self, consistency_checker):
        matrix = consistency_checker.matrix()
        assert isinstance(matrix, dict)
        for hyp, row in matrix.items():
            assert isinstance(row, dict)

    def test_mutual_consistency_checks_present(self, consistency_checker):
        results = consistency_checker.run_all()
        mutual = [r for r in results if "↔" in r.hypothesis]
        assert len(mutual) >= 3


# ---------------------------------------------------------------------------
# Observational checker
# ---------------------------------------------------------------------------

class TestObservationalChecker:
    def test_run_all_returns_results(self, observational_checker):
        results = observational_checker.run_all()
        assert len(results) >= 6

    def test_all_pass(self, observational_checker):
        """All current observational tests should pass."""
        for r in observational_checker.run_all():
            assert r.passed, f"FAILED: {r.hypothesis} — {r.constraint_name}: {r.notes}"

    def test_casimir_passes(self, observational_checker):
        results = observational_checker.run_all()
        casimir = [r for r in results if "Casimir" in r.constraint_name]
        assert len(casimir) == 1
        assert casimir[0].passed

    def test_hubble_passes(self, observational_checker):
        results = observational_checker.run_all()
        hub = [r for r in results if "Hubble" in r.constraint_name]
        assert len(hub) >= 1
        assert all(r.passed for r in hub)

    def test_sigma_deviations_reasonable(self, observational_checker):
        """Where sigma is computed, it should be < 5."""
        for r in observational_checker.run_all():
            if r.sigma_deviation is not None:
                assert r.sigma_deviation < 5.0, (
                    f"{r.hypothesis} — {r.constraint_name}: {r.sigma_deviation:.2f}σ"
                )

    def test_summary_groups_by_hypothesis(self, observational_checker):
        summary = observational_checker.summary()
        assert isinstance(summary, dict)
        for hid, entries in summary.items():
            assert isinstance(entries, list)
            for e in entries:
                assert "constraint" in e
                assert "passed" in e

    def test_constraints_catalog(self, observational_checker):
        constraints = observational_checker.all_constraints()
        assert len(constraints) >= 8
        for c in constraints:
            assert len(c.name) > 0
            assert len(c.relevant_hypotheses) >= 1


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

class TestReportGeneration:
    def test_generate_report_returns_string(self, registry):
        md = generate_report()
        assert isinstance(md, str)
        assert len(md) > 1000

    def test_report_contains_all_hypothesis_ids(self, registry):
        md = generate_report()
        for hid in ["H1", "H2", "H3", "H4", "H5", "H6"]:
            assert hid in md

    def test_report_contains_key_sections(self):
        md = generate_report()
        assert "## Hypothesis Status Summary" in md
        assert "## Hypothesis Details" in md
        assert "## Consistency Analysis" in md
        assert "## Observational Constraints" in md
        assert "## Open Research Questions" in md

    def test_report_contains_status_badges(self):
        md = generate_report()
        # At least one status badge must appear
        assert any(badge in md for badge in ["✅", "🟡", "❌", "⬜"])

    def test_report_saves_to_file(self, tmp_path):
        out = tmp_path / "test_report.md"
        md = generate_report(output_path=str(out))
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert content == md

    def test_report_contains_references(self):
        md = generate_report()
        assert "Verlinde" in md
        assert "Bekenstein" in md
        assert "Planck" in md


# ---------------------------------------------------------------------------
# Physical constant sanity checks
# ---------------------------------------------------------------------------

class TestPhysicalConstants:
    """Verify that constants used in predictions are internally consistent."""

    def test_speed_of_light(self):
        assert math.isclose(P.C, 2.99792458e8, rel_tol=1e-10)

    def test_planck_length(self):
        L_P = math.sqrt(P.HBAR * P.G_N / P.C**3)
        assert math.isclose(L_P, P.L_PLANCK, rel_tol=1e-10)

    def test_planck_time(self):
        T_P = math.sqrt(P.HBAR * P.G_N / P.C**5)
        assert math.isclose(T_P, P.T_PLANCK, rel_tol=1e-10)

    def test_planck_energy(self):
        E_P = P.M_PLANCK * P.C**2
        assert math.isclose(E_P, P.E_PLANCK, rel_tol=1e-10)

    def test_planck_length_order_of_magnitude(self):
        # L_P should be ~1.6e-35 m
        assert 1e-36 < P.L_PLANCK < 1e-34

    def test_hubble_constant_order(self):
        # H0 should be ~2e-18 s⁻¹
        assert 1e-19 < P.H0_SI < 1e-17
