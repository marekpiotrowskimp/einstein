"""
einstein.llm
=============
LLM integration module — Phase 6.

Provides:
- LLM client abstraction (mock, Copilot, Ollama)
- Hypothesis consistency checker (pairwise + new hypothesis validation)
- Literature mapper (theory mapping + BibTeX bibliography)

Quick start
-----------
from einstein.llm import get_client, LLMConsistencyChecker, LiteratureMapper
from einstein.hypotheses import build_registry

client   = get_client()          # default: mock (no API key needed)
registry = build_registry()

# Pairwise consistency
checker = LLMConsistencyChecker(client, registry)
result  = checker.check_pair("H1", "H3")
print(result.verdict, result.rationale)

# Full 15-pair matrix
matrix = checker.full_matrix()

# Validate a new hypothesis
vr = checker.validate_new_hypothesis(
    statement="Dark matter is a stable φ⁶ soliton at the electroweak scale.",
    assumptions=["Three degenerate vacuum states", "Soliton mass ~ 100 GeV"],
)
print(vr.verdict, vr.consistent_with)

# Literature mapping
mapper  = LiteratureMapper(client, registry)
mapping = mapper.map_hypothesis("H4")
bib     = mapper.generate_bibliography("H4")
print(bib.to_bibtex())

# Switch to real LLM (requires GITHUB_TOKEN env var):
# client = get_client(provider="copilot")
"""

from einstein.llm.client import (
    LLMClient,
    MockLLMClient,
    CopilotLLMClient,
    OllamaLLMClient,
    get_client,
)
from einstein.llm.consistency import (
    LLMConsistencyChecker,
    LLMConsistencyResult,
    ValidationResult,
)
from einstein.llm.literature import (
    LiteratureMapper,
    TheoryEntry,
    TheoryMapping,
    BibEntry,
    Bibliography,
    LiteratureReport,
)

__all__ = [
    # client
    "LLMClient",
    "MockLLMClient",
    "CopilotLLMClient",
    "OllamaLLMClient",
    "get_client",
    # consistency
    "LLMConsistencyChecker",
    "LLMConsistencyResult",
    "ValidationResult",
    # literature
    "LiteratureMapper",
    "TheoryEntry",
    "TheoryMapping",
    "BibEntry",
    "Bibliography",
    "LiteratureReport",
]
