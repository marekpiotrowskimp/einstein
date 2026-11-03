"""
einstein.hypotheses.specification
===================================
Formal specification of the six core hypotheses H1–H6.

Each hypothesis is a structured object with:
- statement         — precise claim in natural language
- assumptions       — explicit prerequisites
- predictions       — testable/computable consequences
- falsification     — conditions under which the hypothesis is refuted
- consistency_with  — relationship to established theories (GR, QFT, SR, QM, Cosmology)
- status            — dynamically evaluated: "supported" | "partial" | "falsified" | "untested"

HypothesisRegistry provides access to all six hypotheses and a summary view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class HypothesisStatus(str, Enum):
    SUPPORTED   = "supported"    # all testable predictions pass
    PARTIAL     = "partial"      # some predictions pass, some untested or in tension
    FALSIFIED   = "falsified"    # at least one prediction definitively fails
    UNTESTED    = "untested"     # no quantitative test available yet


class ConsistencyVerdict(str, Enum):
    CONSISTENT   = "consistent"    # hypothesis is compatible with the theory
    TENSION      = "tension"       # hypothesis is in tension, requires extension
    CONFLICT     = "conflict"      # hypothesis directly contradicts the theory
    SUBSUMES     = "subsumes"      # hypothesis generalises the theory
    UNDETERMINED = "undetermined"  # relationship unclear without further work


@dataclass
class TheoryRelation:
    """Relationship between a hypothesis and an established theory."""
    theory: str
    verdict: ConsistencyVerdict
    notes: str


@dataclass
class Prediction:
    """A single testable prediction derived from a hypothesis."""
    id: str                          # e.g. "H1-P1"
    description: str                 # human-readable
    computable: bool                 # can we evaluate it numerically?
    evaluator: Callable | None       # function() -> (passed: bool, value, expected, notes)
    references: list[str] = field(default_factory=list)


@dataclass
class FalsificationCondition:
    """A condition that, if met, falsifies the hypothesis."""
    description: str
    evaluator: Callable | None       # function() -> bool  (True = condition met = falsified)


@dataclass
class Hypothesis:
    """
    Formal specification of a research hypothesis.

    Attributes
    ----------
    id : str
        Short identifier, e.g. "H1".
    name : str
        Descriptive name.
    statement : str
        Precise claim.
    assumptions : list[str]
        Explicit prerequisites that must hold for the hypothesis to make sense.
    predictions : list[Prediction]
        Testable consequences.
    falsification_conditions : list[FalsificationCondition]
        Conditions that would definitively refute the hypothesis.
    theory_relations : list[TheoryRelation]
        Compatibility with established theories.
    status : HypothesisStatus
        Dynamically evaluated.
    notes : str
        Additional context.
    """
    id: str
    name: str
    statement: str
    assumptions: list[str]
    predictions: list[Prediction]
    falsification_conditions: list[FalsificationCondition]
    theory_relations: list[TheoryRelation]
    notes: str = ""
    _status: HypothesisStatus = field(default=HypothesisStatus.UNTESTED, init=False, repr=False)

    @property
    def status(self) -> HypothesisStatus:
        """
        Dynamically evaluate status from prediction results.

        Logic:
        - Any falsification condition met → FALSIFIED
        - All computable predictions pass → SUPPORTED
        - Some computable predictions pass → PARTIAL
        - No computable predictions → UNTESTED
        """
        # Check falsification conditions first
        for fc in self.falsification_conditions:
            if fc.evaluator is not None:
                try:
                    if fc.evaluator():
                        return HypothesisStatus.FALSIFIED
                except Exception:
                    pass

        computable = [p for p in self.predictions if p.computable and p.evaluator is not None]
        if not computable:
            return HypothesisStatus.UNTESTED

        results = []
        for p in computable:
            try:
                passed, *_ = p.evaluator()
                results.append(bool(passed))
            except Exception:
                results.append(False)

        n_passed = sum(results)
        if n_passed == len(results):
            return HypothesisStatus.SUPPORTED
        elif n_passed > 0:
            return HypothesisStatus.PARTIAL
        else:
            return HypothesisStatus.FALSIFIED

    def evaluate_predictions(self) -> list[dict]:
        """
        Run all computable predictions and return structured results.

        Returns
        -------
        list of dicts with keys: id, description, passed, value, expected, notes
        """
        results = []
        for p in self.predictions:
            if not p.computable or p.evaluator is None:
                results.append({
                    "id": p.id,
                    "description": p.description,
                    "passed": None,
                    "value": None,
                    "expected": None,
                    "notes": "Not yet computable",
                })
                continue
            try:
                outcome = p.evaluator()
                passed = outcome[0]
                value = outcome[1] if len(outcome) > 1 else None
                expected = outcome[2] if len(outcome) > 2 else None
                notes = outcome[3] if len(outcome) > 3 else ""
            except Exception as e:
                passed, value, expected, notes = False, None, None, str(e)

            results.append({
                "id": p.id,
                "description": p.description,
                "passed": passed,
                "value": value,
                "expected": expected,
                "notes": notes,
            })
        return results


class HypothesisRegistry:
    """
    Registry of all project hypotheses.

    Usage
    -----
    registry = HypothesisRegistry()
    h1 = registry["H1"]
    for h in registry:
        print(h.id, h.status)
    registry.summary()
    """

    def __init__(self, hypotheses: list[Hypothesis]) -> None:
        self._hypotheses: dict[str, Hypothesis] = {h.id: h for h in hypotheses}

    def __getitem__(self, key: str) -> Hypothesis:
        return self._hypotheses[key]

    def __iter__(self):
        return iter(self._hypotheses.values())

    def __len__(self) -> int:
        return len(self._hypotheses)

    def all(self) -> list[Hypothesis]:
        return list(self._hypotheses.values())

    def by_status(self, status: HypothesisStatus) -> list[Hypothesis]:
        return [h for h in self if h.status == status]

    def summary(self) -> dict:
        """Return a summary dict of all hypotheses and their current status."""
        return {
            h.id: {
                "name": h.name,
                "status": h.status.value,
                "n_predictions": len(h.predictions),
                "n_computable": sum(1 for p in h.predictions if p.computable),
                "n_theory_relations": len(h.theory_relations),
            }
            for h in self
        }

    def consistency_matrix(self) -> dict[str, dict[str, str]]:
        """
        Return a matrix of hypothesis ↔ theory consistency verdicts.

        Returns
        -------
        dict[hypothesis_id, dict[theory_name, verdict]]
        """
        matrix = {}
        for h in self:
            matrix[h.id] = {
                rel.theory: rel.verdict.value
                for rel in h.theory_relations
            }
        return matrix
