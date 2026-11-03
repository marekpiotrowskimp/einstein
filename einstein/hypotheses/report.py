"""
einstein.hypotheses.report
============================
Generate a Markdown report summarising all hypothesis test results.

Usage
-----
from einstein.hypotheses.report import generate_report
md = generate_report()
print(md)
# or save to file:
generate_report(output_path="outputs/hypothesis_report.md")
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path

from einstein.hypotheses.specification import (
    HypothesisStatus,
    ConsistencyVerdict,
)
from einstein.hypotheses.consistency import ConsistencyChecker
from einstein.hypotheses.observational import ObservationalChecker
from einstein.hypotheses import build_registry


# ---------------------------------------------------------------------------
# Status badge helpers
# ---------------------------------------------------------------------------

_STATUS_EMOJI = {
    HypothesisStatus.SUPPORTED:  "✅",
    HypothesisStatus.PARTIAL:    "🟡",
    HypothesisStatus.FALSIFIED:  "❌",
    HypothesisStatus.UNTESTED:   "⬜",
}

_VERDICT_EMOJI = {
    ConsistencyVerdict.CONSISTENT:   "✅",
    ConsistencyVerdict.SUBSUMES:     "🔼",
    ConsistencyVerdict.TENSION:      "⚠️",
    ConsistencyVerdict.CONFLICT:     "❌",
    ConsistencyVerdict.UNDETERMINED: "❓",
}

_PASS_EMOJI = {True: "✅", False: "❌", None: "⬜"}


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _header(registry) -> str:
    lines = [
        "# Einstein — Hypothesis Testing Report",
        "",
        f"> Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Overview",
        "",
        "This report evaluates the six core hypotheses H1–H6 against:",
        "1. Derived numerical predictions (analytic and computational)",
        "2. Cross-consistency with established theories (GR, QFT, SR, QM, ΛCDM)",
        "3. Observational / experimental constraints",
        "",
    ]
    return "\n".join(lines)


def _status_table(registry) -> str:
    summary = registry.summary()
    lines = [
        "## Hypothesis Status Summary",
        "",
        "| ID | Name | Status | Predictions | Computable | Theory relations |",
        "|---|---|---|---|---|---|",
    ]
    for hid, info in summary.items():
        h = registry[hid]
        badge = _STATUS_EMOJI.get(h.status, "?")
        lines.append(
            f"| **{hid}** | {info['name']} | {badge} {h.status.value} "
            f"| {info['n_predictions']} | {info['n_computable']} "
            f"| {info['n_theory_relations']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _hypothesis_details(registry) -> str:
    sections = ["## Hypothesis Details", ""]
    for h in registry:
        badge = _STATUS_EMOJI.get(h.status, "?")
        sections += [
            f"### {h.id} — {h.name}  {badge}",
            "",
            f"**Statement:** {h.statement}",
            "",
            "**Assumptions:**",
        ]
        for a in h.assumptions:
            sections.append(f"- {a}")

        sections += ["", "**Predictions:**", ""]
        results = h.evaluate_predictions()
        rows = ["| ID | Description | Passed | Value | Expected | Notes |",
                "|---|---|---|---|---|---|"]
        for r in results:
            badge_p = _PASS_EMOJI.get(r["passed"], "?")
            val  = f"{r['value']:.4g}"  if isinstance(r["value"],    float) else str(r["value"])
            exp  = f"{r['expected']:.4g}" if isinstance(r["expected"], float) else str(r["expected"])
            note = (r["notes"] or "")[:120].replace("|", "∣")
            rows.append(f"| {r['id']} | {r['description']} | {badge_p} | {val} | {exp} | {note} |")
        sections += rows

        sections += ["", "**Falsification conditions:**"]
        for fc in h.falsification_conditions:
            met = "—"
            if fc.evaluator:
                try:
                    met = "MET ❌" if fc.evaluator() else "not met ✅"
                except Exception:
                    met = "error"
            sections.append(f"- {fc.description} → {met}")

        if h.notes:
            sections += ["", f"**Notes:** {h.notes}"]

        sections.append("")
        sections.append("---")
        sections.append("")
    return "\n".join(sections)


def _consistency_section() -> str:
    checker = ConsistencyChecker()
    results = checker.run_all()
    counts  = checker.summary_counts()

    lines = [
        "## Consistency Analysis",
        "",
        "### Summary",
        "",
        "| Verdict | Count |",
        "|---|---|",
    ]
    for verdict, n in sorted(counts.items()):
        emoji = _VERDICT_EMOJI.get(ConsistencyVerdict(verdict), "?")
        lines.append(f"| {emoji} {verdict} | {n} |")

    lines += ["", "### Tension / Conflict Details", ""]
    tensions = [r for r in results if r.verdict in (
        ConsistencyVerdict.TENSION, ConsistencyVerdict.CONFLICT)]
    if tensions:
        for r in tensions:
            emoji = _VERDICT_EMOJI[r.verdict]
            lines.append(f"**{r.hypothesis} ↔ {r.theory}** {emoji}")
            lines.append(f"> {r.rationale}")
            if r.quantitative_check:
                lines.append(f"> Quantitative: {r.quantitative_check}")
            lines.append("")
    else:
        lines.append("_No conflicts detected._")
        lines.append("")

    lines += ["", "### Full Consistency Matrix", ""]

    matrix = checker.matrix()
    # Collect all unique theories
    all_theories: list[str] = []
    seen: set[str] = set()
    for row in matrix.values():
        for t in row:
            if t not in seen:
                all_theories.append(t)
                seen.add(t)

    header_row = "| Hypothesis | " + " | ".join(all_theories) + " |"
    sep_row    = "|---|" + "---|" * len(all_theories)
    lines += [header_row, sep_row]

    for hid, row in matrix.items():
        cells = []
        for t in all_theories:
            if t in row:
                v = ConsistencyVerdict(row[t])
                cells.append(f"{_VERDICT_EMOJI.get(v, '?')} {row[t]}")
            else:
                cells.append("—")
        lines.append(f"| {hid} | " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def _observational_section() -> str:
    checker = ObservationalChecker()
    by_hyp  = checker.summary()

    lines = [
        "## Observational Constraints",
        "",
        "Comparison of hypothesis predictions with measured physical quantities.",
        "",
    ]

    for hid in sorted(by_hyp.keys()):
        entries = by_hyp[hid]
        lines += [f"### {hid}", ""]
        rows = [
            "| Constraint | Passed | Predicted | Observed | σ | Notes |",
            "|---|---|---|---|---|---|",
        ]
        for e in entries:
            badge = _PASS_EMOJI.get(e["passed"], "?")
            pred  = f"{e['predicted']:.4g}"  if isinstance(e["predicted"], float) else "—"
            obs   = f"{e['observed']:.4g}"   if isinstance(e["observed"],  float) else "—"
            sigma = f"{e['sigma']:.2f}" if isinstance(e["sigma"], float) else "—"
            note  = (e["notes"] or "")[:100].replace("|", "∣")
            rows.append(f"| {e['constraint']} | {badge} | {pred} | {obs} | {sigma} | {note} |")
        lines += rows
        lines.append("")

    return "\n".join(lines)


def _open_questions(registry) -> str:
    lines = [
        "## Open Research Questions",
        "",
        "Based on the analysis above, the following questions require further work:",
        "",
    ]

    questions = [
        ("H1", "What mechanism resolves the 10^120 cosmological constant discrepancy?"),
        ("H1/H2", "Can the vacuum energy density be derived from first principles that match Λ_obs?"),
        ("H3", "What field parameters (λ, v, energy scale) reproduce the electron mass and other SM masses?"),
        ("H3", "Can the kink model be extended to include spin-½ fermions and gauge charges?"),
        ("H4", "Does Verlinde's entropic gravity extend to full non-linear GR (beyond Newton's limit)?"),
        ("H4", "What is the microscopic origin of the holographic screen in a non-AdS spacetime?"),
        ("H5", "Can the Page-Wootters mechanism be realised in the φ⁴ field framework?"),
        ("H5", "Is there a measurable physical consequence of discretised Planck-time evolution?"),
        ("H6", "What drives the large-scale phase transition / expansion of the fundamental structure?"),
        ("H1–H6", "Is there a single unified mathematical framework that contains all six hypotheses?"),
    ]

    for hid, q in questions:
        lines.append(f"- **[{hid}]** {q}")

    lines.append("")
    return "\n".join(lines)


def _footer() -> str:
    return "\n".join([
        "---",
        "",
        "## References",
        "",
        "- Einstein, A. (1915). *Die Feldgleichungen der Gravitation*",
        "- Bekenstein, J. D. (1973). Black holes and entropy. *Phys. Rev. D*, 7, 2333.",
        "- Hawking, S. W. (1975). Particle creation by black holes. *Commun. Math. Phys.*, 43, 199.",
        "- Jacobson, T. (1995). Thermodynamics of spacetime. *Phys. Rev. Lett.*, 75, 1260.",
        "- Verlinde, E. (2011). On the origin of gravity and the laws of Newton. *JHEP*, 04, 029.",
        "- Planck Collaboration (2018). Planck 2018 results VI. *A&A*, 641, A6.",
        "- Lamoreaux, S. K. (1997). Demonstration of the Casimir force. *Phys. Rev. Lett.*, 78, 5.",
        "",
        "> *This report is auto-generated by `einstein.hypotheses.report.generate_report()`.*",
        "",
    ])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _llm_section(llm_client) -> str:
    """
    Generate the LLM analysis section (Phase 6).

    Parameters
    ----------
    llm_client : LLMClient
        An initialised LLM client.

    Returns
    -------
    str
        Markdown section string.
    """
    from einstein.llm.consistency import LLMConsistencyChecker
    from einstein.llm.literature import LiteratureMapper

    registry = build_registry()
    lines = [
        "## LLM Analysis (Phase 6)",
        "",
        f"> Provider: `{llm_client.provider_name}` | Model: `{llm_client.model_name}`",
        "",
    ]

    # ── Consistency matrix ─────────────────────────────────────────────────
    lines += ["### Pairwise Consistency Matrix (LLM)", ""]
    checker = LLMConsistencyChecker(llm_client, registry)
    results = checker.full_matrix()

    _VERDICT_EMOJI_LLM = {
        "consistent":   "✅",
        "tension":      "⚠️",
        "conflict":     "❌",
        "undetermined": "❓",
    }

    ids = ["H1", "H2", "H3", "H4", "H5", "H6"]
    mat: dict[str, dict[str, str]] = {}
    for r in results:
        mat.setdefault(r.h1, {})[r.h2] = r.verdict
        mat.setdefault(r.h2, {})[r.h1] = r.verdict

    header = "| | " + " | ".join(ids) + " |"
    sep    = "|---|" + "---|" * len(ids)
    lines += [header, sep]
    for row_id in ids:
        cells = []
        for col_id in ids:
            if row_id == col_id:
                cells.append("—")
            else:
                v = mat.get(row_id, {}).get(col_id, "?")
                cells.append(f"{_VERDICT_EMOJI_LLM.get(v, '?')} {v}")
        lines.append(f"| **{row_id}** | " + " | ".join(cells) + " |")
    lines.append("")

    # Summary counts
    counts = checker.summary()
    lines += ["**Verdict counts:**", ""]
    for v, n in sorted(counts.items()):
        lines.append(f"- {_VERDICT_EMOJI_LLM.get(v, '?')} {v}: {n}")
    lines.append("")

    # Tensions / conflicts detail
    problematic = [r for r in results if r.verdict in ("tension", "conflict")]
    if problematic:
        lines += ["**Tensions and conflicts:**", ""]
        for r in problematic:
            emoji = _VERDICT_EMOJI_LLM.get(r.verdict, "?")
            lines.append(f"{emoji} **{r.h1} ↔ {r.h2}** (confidence: {r.confidence:.2f})")
            lines.append(f"> {r.rationale}")
            if r.open_questions:
                for q in r.open_questions[:2]:
                    lines.append(f"> - {q}")
            lines.append("")

    # ── Literature mapping ─────────────────────────────────────────────────
    lines += ["---", "", "### Theory Mapping (LLM)", ""]
    mapper = LiteratureMapper(llm_client, registry)
    for hid in ids:
        mapping = mapper.map_hypothesis(hid)
        lines.append(f"#### {hid} — {registry[hid].name}")
        lines.append("")
        for t in mapping.theories:
            emoji_map = {"direct": "🔴", "strong": "🟠", "moderate": "🟡", "weak": "⚪"}
            e = emoji_map.get(t.overlap, "•")
            lines.append(f"- {e} **{t.name}** — {t.connection}")
        if mapping.summary:
            lines.append(f"\n> {mapping.summary}")
        lines.append("")

    # ── Bibliography ───────────────────────────────────────────────────────
    lines += ["---", "", "### Bibliography (LLM-generated)", ""]
    for hid in ids:
        bib = mapper.generate_bibliography(hid)
        lines.append(f"#### {hid}")
        lines.append("")
        for i, e in enumerate(bib.entries, 1):
            lines.append(f"{i}. {e.to_markdown()}")
        lines.append("")

    return "\n".join(lines)


def generate_report(
    output_path: str | Path | None = None,
    include_llm: bool = False,
    llm_client=None,
    bibtex_path: str | Path | None = None,
) -> str:
    """
    Generate the full hypothesis testing report in Markdown format.

    Parameters
    ----------
    output_path : str or Path, optional
        If provided, write the report to this file in addition to returning it.
    include_llm : bool, optional
        If True, append the LLM analysis section (Phase 6). Requires llm_client.
        Default: False.
    llm_client : LLMClient, optional
        LLM client for Phase 6 section. If None and include_llm=True,
        a default MockLLMClient is used.
    bibtex_path : str or Path, optional
        If provided and include_llm=True, also save a .bib file.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    registry = build_registry()

    sections = [
        _header(registry),
        _status_table(registry),
        _hypothesis_details(registry),
        _consistency_section(),
        _observational_section(),
        _open_questions(registry),
        _footer(),
    ]

    if include_llm:
        if llm_client is None:
            from einstein.llm.client import get_client
            llm_client = get_client("mock")
        sections.append(_llm_section(llm_client))

        if bibtex_path is not None:
            from einstein.llm.literature import LiteratureMapper
            mapper = LiteratureMapper(llm_client, registry)
            mapper.save_bibtex(str(bibtex_path))

    report = "\n".join(sections)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")

    return report
