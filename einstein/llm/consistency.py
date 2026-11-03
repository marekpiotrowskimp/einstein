"""
einstein.llm.consistency
=========================
LLM-powered hypothesis consistency checker.

Complements the programmatic ConsistencyChecker from Phase 5 with qualitative
LLM reasoning about hypothesis compatibility.

Features
--------
- Pairwise consistency check for all 15 pairs of H1–H6
- Validation of a new user-supplied hypothesis against H1–H6
- Structured results with verdict, rationale, confidence, open questions

Usage
-----
from einstein.llm.client import get_client
from einstein.llm.consistency import LLMConsistencyChecker
from einstein.hypotheses import build_registry

client   = get_client()           # default: mock
registry = build_registry()
checker  = LLMConsistencyChecker(client, registry)

# Single pair
result = checker.check_pair("H1", "H3")
print(result.verdict, result.confidence)

# Full 15-pair matrix
matrix = checker.full_matrix()

# Validate a new hypothesis
vr = checker.validate_new_hypothesis(
    statement="Dark matter consists of stable φ⁶ solitons at the electroweak scale.",
    assumptions=["φ⁶ potential has three degenerate minima", "soliton mass ~ 100 GeV"],
)
print(vr.verdict, vr.consistent_with)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from einstein.llm.client import LLMClient, get_client
from einstein.llm.prompts import (
    PHYSICS_EXPERT_SYSTEM,
    render_consistency_prompt,
    render_validation_prompt,
)
from einstein.hypotheses import build_registry
from einstein.hypotheses.specification import HypothesisRegistry

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

VALID_VERDICTS = {"consistent", "tension", "conflict", "undetermined"}
VALID_VALIDATION_VERDICTS = {"consistent", "partial", "tension", "conflict"}
VALID_NOVELTY = {"redundant", "complementary", "extension", "orthogonal"}


@dataclass
class LLMConsistencyResult:
    """
    Result of a pairwise hypothesis consistency check by LLM.

    Attributes
    ----------
    h1, h2 : str
        Hypothesis IDs (e.g. "H1", "H3").
    verdict : str
        "consistent" | "tension" | "conflict" | "undetermined"
    rationale : str
        LLM explanation (2-4 sentences).
    confidence : float
        Model confidence 0.0–1.0.
    open_questions : list[str]
        Questions the LLM identified for further investigation.
    supporting_theories : list[str]
        Established theories relevant to both hypotheses.
    raw_response : str
        Raw text response from the LLM (for debugging).
    """
    h1: str
    h2: str
    verdict: str
    rationale: str
    confidence: float
    open_questions: list[str] = field(default_factory=list)
    supporting_theories: list[str] = field(default_factory=list)
    raw_response: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VALID_VERDICTS:
            self.verdict = "undetermined"
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


@dataclass
class ValidationResult:
    """
    Result of validating a new hypothesis against H1–H6.

    Attributes
    ----------
    statement : str
        The new hypothesis statement that was validated.
    verdict : str
        "consistent" | "partial" | "tension" | "conflict"
    consistent_with : list[str]
        Hypothesis IDs the new hypothesis is consistent with.
    tensions_with : list[str]
        Hypothesis IDs it has tension with.
    conflicts_with : list[str]
        Hypothesis IDs it directly conflicts with.
    rationale : str
        LLM explanation.
    confidence : float
        Model confidence 0.0–1.0.
    suggestions : list[str]
        Suggestions for improving the new hypothesis.
    novelty : str
        "redundant" | "complementary" | "extension" | "orthogonal"
    raw_response : str
        Raw LLM text.
    """
    statement: str
    verdict: str
    consistent_with: list[str]
    tensions_with: list[str]
    conflicts_with: list[str]
    rationale: str
    confidence: float
    suggestions: list[str] = field(default_factory=list)
    novelty: str = "complementary"
    raw_response: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VALID_VALIDATION_VERDICTS:
            self.verdict = "partial"
        if self.novelty not in VALID_NOVELTY:
            self.novelty = "complementary"
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


# ---------------------------------------------------------------------------
# LLMConsistencyChecker
# ---------------------------------------------------------------------------

class LLMConsistencyChecker:
    """
    LLM-powered hypothesis consistency checker.

    Parameters
    ----------
    client : LLMClient
        The LLM client to use (mock, copilot, ollama…).
    registry : HypothesisRegistry, optional
        Hypothesis registry. Defaults to build_registry().
    """

    ALL_IDS = ["H1", "H2", "H3", "H4", "H5", "H6"]

    def __init__(
        self,
        client: LLMClient | None = None,
        registry: HypothesisRegistry | None = None,
    ) -> None:
        self._client   = client   or get_client()
        self._registry = registry or build_registry()

    @property
    def client(self) -> LLMClient:
        return self._client

    @property
    def registry(self) -> HypothesisRegistry:
        return self._registry

    # ── Single pair ────────────────────────────────────────────────────────

    def check_pair(self, id1: str, id2: str) -> LLMConsistencyResult:
        """
        Check consistency between two hypotheses.

        Parameters
        ----------
        id1, id2 : str
            Hypothesis IDs, e.g. "H1", "H3".

        Returns
        -------
        LLMConsistencyResult
        """
        h1 = self._registry[id1]
        h2 = self._registry[id2]

        prompt = render_consistency_prompt(h1, h2)
        raw    = self._client.complete(PHYSICS_EXPERT_SYSTEM, prompt)
        data   = self._client.complete_json(PHYSICS_EXPERT_SYSTEM, prompt)

        # complete_json calls complete internally — avoid double call
        # Re-use raw from first call and parse separately
        from einstein.llm.client import _extract_json
        data = _extract_json(raw)

        return LLMConsistencyResult(
            h1=id1,
            h2=id2,
            verdict=str(data.get("verdict", "undetermined")),
            rationale=str(data.get("rationale", "")),
            confidence=float(data.get("confidence", 0.5)),
            open_questions=list(data.get("open_questions", [])),
            supporting_theories=list(data.get("supporting_theories", [])),
            raw_response=raw,
        )

    # ── Full 15-pair matrix ────────────────────────────────────────────────

    def full_matrix(self) -> list[LLMConsistencyResult]:
        """
        Run consistency checks for all 15 pairs of H1–H6.

        Returns
        -------
        list[LLMConsistencyResult]
            15 results, one per unique pair.
        """
        results = []
        pairs   = list(itertools.combinations(self.ALL_IDS, 2))
        for id1, id2 in pairs:
            try:
                result = self.check_pair(id1, id2)
            except Exception as e:
                result = LLMConsistencyResult(
                    h1=id1, h2=id2,
                    verdict="undetermined",
                    rationale=f"Error during LLM call: {e}",
                    confidence=0.0,
                    raw_response=str(e),
                )
            results.append(result)
        return results

    def matrix_as_dict(self) -> dict[str, dict[str, str]]:
        """
        Return full matrix as nested dict[h1][h2] = verdict string.
        Symmetric: both (H1,H2) and (H2,H1) are filled.
        """
        results = self.full_matrix()
        mat: dict[str, dict[str, str]] = {}
        for r in results:
            mat.setdefault(r.h1, {})[r.h2] = r.verdict
            mat.setdefault(r.h2, {})[r.h1] = r.verdict
        return mat

    def tensions_and_conflicts(self) -> list[LLMConsistencyResult]:
        """Return only results with verdict 'tension' or 'conflict'."""
        return [
            r for r in self.full_matrix()
            if r.verdict in ("tension", "conflict")
        ]

    def summary(self) -> dict[str, int]:
        """Count results by verdict type."""
        from collections import Counter
        results = self.full_matrix()
        return dict(Counter(r.verdict for r in results))

    # ── New hypothesis validation ──────────────────────────────────────────

    def validate_new_hypothesis(
        self,
        statement: str,
        assumptions: list[str] | None = None,
    ) -> ValidationResult:
        """
        Validate a new user-supplied hypothesis against H1–H6.

        Parameters
        ----------
        statement : str
            The hypothesis statement to validate.
        assumptions : list[str], optional
            Explicit assumptions. Defaults to empty list.

        Returns
        -------
        ValidationResult
        """
        assumptions = assumptions or []
        prompt = render_validation_prompt(statement, assumptions, self._registry)
        raw    = self._client.complete(PHYSICS_EXPERT_SYSTEM, prompt)

        from einstein.llm.client import _extract_json
        data = _extract_json(raw)

        return ValidationResult(
            statement=statement,
            verdict=str(data.get("verdict", "partial")),
            consistent_with=list(data.get("consistent_with", [])),
            tensions_with=list(data.get("tensions_with", [])),
            conflicts_with=list(data.get("conflicts_with", [])),
            rationale=str(data.get("rationale", "")),
            confidence=float(data.get("confidence", 0.5)),
            suggestions=list(data.get("suggestions", [])),
            novelty=str(data.get("novelty", "complementary")),
            raw_response=raw,
        )
