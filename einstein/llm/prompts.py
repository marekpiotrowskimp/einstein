"""
einstein.llm.prompts
=====================
Prompt templates for all LLM tasks in Phase 6.

Each public function renders a complete prompt string given hypothesis data.
Prompts are designed to elicit structured JSON responses.

All prompts instruct the model to return ONLY valid JSON — no markdown prose.
"""

from __future__ import annotations

from einstein.hypotheses.specification import Hypothesis, HypothesisRegistry


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

PHYSICS_EXPERT_SYSTEM = """\
You are an expert theoretical physicist with deep knowledge of:
- General Relativity and differential geometry
- Quantum Field Theory and the Standard Model
- Cosmology (ΛCDM, inflation, dark energy)
- Emergent gravity, holographic principles, entropic gravity
- Topological solitons and nonlinear field theory
- Foundations of quantum mechanics (relational QM, Page-Wootters, Wheeler-DeWitt)
- Loop Quantum Gravity, String Theory, Causal Dynamical Triangulations

Your role is to rigorously evaluate hypotheses about the foundations of physics.
You are objective, critical, and precise. You distinguish between:
- Established results (theorems, experimental facts)
- Speculative but well-motivated ideas
- Ideas with known internal inconsistencies

You ALWAYS respond with valid JSON only. No markdown, no prose outside the JSON object."""

LITERATURE_EXPERT_SYSTEM = """\
You are an expert physicist and science librarian with comprehensive knowledge of
the theoretical physics literature from 1900 to 2024. You know:
- The key papers, authors, and journals in each subfield
- How different theoretical frameworks relate to each other
- Correct bibliographic information (authors, year, title, journal, DOI)

You ALWAYS respond with valid JSON only. No markdown, no prose outside the JSON object."""


# ---------------------------------------------------------------------------
# Consistency check prompts
# ---------------------------------------------------------------------------

def render_consistency_prompt(
    h1: Hypothesis,
    h2: Hypothesis,
) -> str:
    """
    Render a user prompt for pairwise hypothesis consistency check.

    Parameters
    ----------
    h1, h2 : Hypothesis
        The two hypotheses to compare.

    Returns
    -------
    str
        User prompt string requesting structured JSON.
    """
    return f"""\
CONSISTENCY_CHECK

Evaluate the logical consistency between these two physics hypotheses.

=== HYPOTHESIS {h1.id}: {h1.name} ===
Statement: {h1.statement}
Assumptions: {'; '.join(h1.assumptions)}

=== HYPOTHESIS {h2.id}: {h2.name} ===
Statement: {h2.statement}
Assumptions: {'; '.join(h2.assumptions)}

Analyse:
1. Are the two hypotheses logically compatible? Can both be true simultaneously?
2. Do they reinforce each other, or are they in tension?
3. What open questions arise from their combination?

Respond with ONLY this JSON structure:
{{
  "verdict": "<consistent|tension|conflict|undetermined>",
  "rationale": "<2-4 sentences explaining the verdict>",
  "confidence": <float 0.0-1.0>,
  "open_questions": ["<question 1>", "<question 2>"],
  "supporting_theories": ["<theory 1>", "<theory 2>"]
}}

Verdict definitions:
- consistent: both can be true simultaneously, may reinforce each other
- tension: both could be true but require non-trivial reconciliation
- conflict: if one is true, the other is likely false
- undetermined: insufficient information to judge"""


def render_validation_prompt(
    statement: str,
    assumptions: list[str],
    registry: HypothesisRegistry,
) -> str:
    """
    Render a prompt to validate a new hypothesis against H1–H6.

    Parameters
    ----------
    statement : str
        The new hypothesis statement.
    assumptions : list[str]
        Explicit assumptions of the new hypothesis.
    registry : HypothesisRegistry
        The existing H1–H6 registry for context.

    Returns
    -------
    str
        User prompt string.
    """
    existing = "\n".join(
        f"  {h.id}: {h.statement[:120]}..."
        for h in registry
    )

    return f"""\
VALIDATE_HYPOTHESIS

Evaluate whether a NEW hypothesis is consistent with the existing H1–H6 framework.

=== EXISTING HYPOTHESES (H1–H6) ===
{existing}

=== NEW HYPOTHESIS ===
Statement: {statement}
Assumptions: {'; '.join(assumptions) if assumptions else 'None specified'}

Analyse:
1. With which existing hypotheses (H1–H6) is the new one consistent?
2. With which does it create tension or conflict?
3. Does it add something new, or is it redundant?
4. What modifications would make it more consistent with H1–H6?

Respond with ONLY this JSON structure:
{{
  "verdict": "<consistent|partial|tension|conflict>",
  "consistent_with": ["H1", ...],
  "tensions_with": ["H4", ...],
  "conflicts_with": [],
  "rationale": "<3-5 sentences>",
  "confidence": <float 0.0-1.0>,
  "suggestions": ["<suggestion 1>", "<suggestion 2>"],
  "novelty": "<redundant|complementary|extension|orthogonal>"
}}"""


# ---------------------------------------------------------------------------
# Literature mapping prompts
# ---------------------------------------------------------------------------

def render_literature_map_prompt(h: Hypothesis) -> str:
    """
    Render a prompt for mapping a hypothesis to related theories and literature.

    Parameters
    ----------
    h : Hypothesis

    Returns
    -------
    str
        User prompt string.
    """
    return f"""\
LITERATURE_MAP

Map the following physics hypothesis to related established theories and research programs.

=== HYPOTHESIS {h.id}: {h.name} ===
Statement: {h.statement}
Assumptions: {'; '.join(h.assumptions)}

List the most relevant established theories, research programs, and frameworks
that overlap with or are relevant to this hypothesis.

For each, specify:
- name: the theory/program name
- connection: how it relates to the hypothesis (1-2 sentences)
- overlap: "direct" | "strong" | "moderate" | "weak"
- key_authors: list of 1-3 key authors/names

Respond with ONLY this JSON structure:
{{
  "hypothesis": "{h.id}",
  "theories": [
    {{
      "name": "<theory name>",
      "connection": "<how it relates>",
      "overlap": "<direct|strong|moderate|weak>",
      "key_authors": ["<author 1>", "<author 2>"]
    }}
  ],
  "summary": "<1-2 sentence overall assessment of the hypothesis in context of existing theory>"
}}

Include between 3 and 6 theories. Prioritise those with the most direct relevance."""


def render_bibliography_prompt(h: Hypothesis) -> str:
    """
    Render a prompt for generating a bibliography for a hypothesis.

    Parameters
    ----------
    h : Hypothesis

    Returns
    -------
    str
        User prompt string.
    """
    return f"""\
BIBLIOGRAPHY

Generate a bibliography of key papers for the following hypothesis.

=== HYPOTHESIS {h.id}: {h.name} ===
Statement: {h.statement}

List the 3-6 most important scientific papers directly relevant to this hypothesis.
Include seminal works and recent key papers.

Respond with ONLY this JSON structure:
{{
  "hypothesis": "{h.id}",
  "entries": [
    {{
      "authors": "<Last, F. M. and Last, F. M.>",
      "year": <integer>,
      "title": "<full paper title>",
      "journal": "<journal or book name>",
      "volume": "<volume or empty string>",
      "pages": "<pages or article number or empty string>",
      "doi": "<DOI or empty string>",
      "relevance": "<one sentence why this paper is relevant>"
    }}
  ]
}}

Use correct bibliographic information. If uncertain about a DOI, leave it empty."""


# ---------------------------------------------------------------------------
# Full-registry batch prompts
# ---------------------------------------------------------------------------

def render_consistency_matrix_intro(registry: HypothesisRegistry) -> str:
    """
    Render a brief context summary of all hypotheses for use in batch checks.
    Not a standalone prompt — used as preamble in multi-turn conversations.
    """
    lines = ["=== HYPOTHESIS FRAMEWORK OVERVIEW ===\n"]
    for h in registry:
        lines.append(f"{h.id} — {h.name}")
        lines.append(f"  {h.statement[:160]}...\n")
    return "\n".join(lines)
