"""
einstein.llm.client
====================
Provider-agnostic LLM client abstraction.

Two implementations:
- MockLLMClient     — deterministic, no API key, for tests and offline use
- CopilotLLMClient  — GitHub Copilot Models API via openai SDK

Provider selection
------------------
Controlled by the LLM_PROVIDER environment variable (or constructor argument):
    LLM_PROVIDER=mock      → MockLLMClient (default)
    LLM_PROVIDER=copilot   → CopilotLLMClient  (requires GITHUB_TOKEN or COPILOT_TOKEN)
    LLM_PROVIDER=openai    → OpenAI-compatible (requires OPENAI_API_KEY)
    LLM_PROVIDER=ollama    → Ollama local (requires Ollama running on localhost:11434)

Usage
-----
from einstein.llm.client import get_client

client = get_client()                   # uses LLM_PROVIDER env var, default=mock
response = client.complete(
    system="You are a physicist.",
    user="What is the Casimir effect?"
)
print(response)
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        """
        Send a chat completion request.

        Parameters
        ----------
        system : str
            System prompt.
        user : str
            User message.
        temperature : float
            Sampling temperature (0=deterministic, 1=creative).

        Returns
        -------
        str
            Model response text.
        """

    def complete_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
    ) -> dict:
        """
        Request a JSON response and parse it.

        The user prompt should instruct the model to respond with valid JSON.
        Falls back to an empty dict if parsing fails.
        """
        response = self.complete(system, user, temperature)
        return _extract_json(response)

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier string."""


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Extract JSON from a model response that may contain markdown code blocks."""
    # Try to find a ```json ... ``` block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find bare JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


# ---------------------------------------------------------------------------
# Mock client — deterministic, no API required
# ---------------------------------------------------------------------------

# Pre-defined mock responses keyed by (task_type, hypothesis_ids)
_MOCK_CONSISTENCY_VERDICTS = {
    ("H1", "H2"): ("consistent", 0.85, "Both describe the active vacuum and energy-space unity."),
    ("H1", "H3"): ("consistent", 0.80, "Active vacuum enables field excitations that become matter."),
    ("H1", "H4"): ("consistent", 0.75, "Vacuum energy contributes to entropy and emergent gravity."),
    ("H1", "H5"): ("tension", 0.60, "Vacuum energy is timeless; emergent time needs a clock substrate."),
    ("H1", "H6"): ("consistent", 0.70, "Vacuum energy drives cosmological expansion via Λ."),
    ("H2", "H3"): ("consistent", 0.88, "If space=energy, localized excitations are matter (H3)."),
    ("H2", "H4"): ("consistent", 0.78, "Energy-space unification supports entropic gravity."),
    ("H2", "H5"): ("consistent", 0.72, "Time as configuration rate is natural if space=energy."),
    ("H2", "H6"): ("consistent", 0.80, "Dynamic geometry is the large-scale manifestation of H2."),
    ("H3", "H4"): ("consistent", 0.82, "Solitonic matter creates entropy gradients → gravity (H4)."),
    ("H3", "H5"): ("consistent", 0.65, "Excitation dynamics define the rate of configuration change."),
    ("H3", "H6"): ("tension", 0.55, "Expansion dilutes soliton density; stability mechanism unclear."),
    ("H4", "H5"): ("consistent", 0.70, "Emergent time and emergent gravity share thermodynamic origin."),
    ("H4", "H6"): ("consistent", 0.75, "Expansion modifies holographic screens and entropic force."),
    ("H5", "H6"): ("consistent", 0.80, "Emergent time and expanding geometry are tightly linked."),
}

_MOCK_LITERATURE = {
    "H1": [
        {"name": "Quantum Field Theory (Weinberg)", "connection": "Zero-point energy and vacuum fluctuations", "overlap": "direct"},
        {"name": "Casimir Effect (Casimir 1948)", "connection": "Measurable force from vacuum energy", "overlap": "direct"},
        {"name": "Cosmological Constant Problem (Weinberg 1989)", "connection": "Discrepancy between QFT and observed Λ", "overlap": "direct"},
    ],
    "H2": [
        {"name": "Loop Quantum Gravity (Rovelli)", "connection": "Space as discrete quantum geometry = quantum energy", "overlap": "strong"},
        {"name": "Causal Dynamical Triangulations", "connection": "Spacetime as dynamical structure", "overlap": "moderate"},
        {"name": "Einstein Field Equations", "connection": "G_μν = 8πG T_μν unifies geometry and energy", "overlap": "strong"},
    ],
    "H3": [
        {"name": "Soliton Theory (Rajaraman 1982)", "connection": "Topological stability of field excitations", "overlap": "direct"},
        {"name": "Standard Model QFT", "connection": "Particles as field quanta", "overlap": "strong"},
        {"name": "Skyrme Model (Skyrme 1961)", "connection": "Baryons as topological solitons of pion fields", "overlap": "strong"},
        {"name": "String Theory", "connection": "Fundamental excitations as matter", "overlap": "moderate"},
    ],
    "H4": [
        {"name": "Entropic Gravity (Verlinde 2011)", "connection": "Gravity from entropy gradient on holographic screens", "overlap": "direct"},
        {"name": "Thermodynamics of Spacetime (Jacobson 1995)", "connection": "EFE derived from Clausius relation", "overlap": "direct"},
        {"name": "AdS/CFT Correspondence", "connection": "Gravity emergent from boundary field theory", "overlap": "strong"},
        {"name": "Bekenstein-Hawking Entropy", "connection": "S = A/4L_P² — area law for BH entropy", "overlap": "direct"},
    ],
    "H5": [
        {"name": "Page-Wootters Mechanism (1983)", "connection": "Time from entanglement between clock and system", "overlap": "direct"},
        {"name": "Relational Quantum Mechanics (Rovelli)", "connection": "Time as relational, not absolute", "overlap": "strong"},
        {"name": "Causal Sets (Bombelli et al. 1987)", "connection": "Discrete causal structure replaces continuous time", "overlap": "moderate"},
        {"name": "Wheeler-DeWitt Equation", "connection": "Timeless quantum cosmology", "overlap": "strong"},
    ],
    "H6": [
        {"name": "FLRW Cosmology", "connection": "Expanding geometry as exact GR solution", "overlap": "direct"},
        {"name": "Inflationary Cosmology (Guth 1981)", "connection": "Rapid phase transition of early universe geometry", "overlap": "strong"},
        {"name": "Dark Energy / Λ (Planck 2018)", "connection": "Observed accelerating expansion", "overlap": "direct"},
        {"name": "Ekpyrotic Scenario (Khoury et al. 2001)", "connection": "Big Bang as collision of branes — geometry transition", "overlap": "moderate"},
    ],
}

_MOCK_BIBLIOGRAPHY = {
    "H1": [
        {"authors": "Casimir, H. B. G.", "year": 1948, "title": "On the attraction between two perfectly conducting plates", "journal": "Proc. Kon. Ned. Akad. Wetensch.", "volume": "51", "pages": "793", "doi": ""},
        {"authors": "Weinberg, S.", "year": 1989, "title": "The cosmological constant problem", "journal": "Rev. Mod. Phys.", "volume": "61", "pages": "1", "doi": "10.1103/RevModPhys.61.1"},
        {"authors": "Lamoreaux, S. K.", "year": 1997, "title": "Demonstration of the Casimir Force in the 0.6 to 6 μm Range", "journal": "Phys. Rev. Lett.", "volume": "78", "pages": "5", "doi": "10.1103/PhysRevLett.78.5"},
    ],
    "H2": [
        {"authors": "Rovelli, C.", "year": 2004, "title": "Quantum Gravity", "journal": "Cambridge University Press", "volume": "", "pages": "", "doi": "10.1017/CBO9780511755804"},
        {"authors": "Ambjorn, J., Jurkiewicz, J., Loll, R.", "year": 2004, "title": "Emergence of a 4D World from Causal Quantum Gravity", "journal": "Phys. Rev. Lett.", "volume": "93", "pages": "131301", "doi": "10.1103/PhysRevLett.93.131301"},
    ],
    "H3": [
        {"authors": "Rajaraman, R.", "year": 1982, "title": "Solitons and Instantons", "journal": "North-Holland", "volume": "", "pages": "", "doi": ""},
        {"authors": "Skyrme, T. H. R.", "year": 1961, "title": "A Non-Linear Field Theory", "journal": "Proc. R. Soc. London A", "volume": "260", "pages": "127", "doi": "10.1098/rspa.1961.0018"},
        {"authors": "Zabusky, N. J., Kruskal, M. D.", "year": 1965, "title": "Interaction of Solitons in a Collisionless Plasma", "journal": "Phys. Rev. Lett.", "volume": "15", "pages": "240", "doi": "10.1103/PhysRevLett.15.240"},
    ],
    "H4": [
        {"authors": "Verlinde, E.", "year": 2011, "title": "On the Origin of Gravity and the Laws of Newton", "journal": "JHEP", "volume": "04", "pages": "029", "doi": "10.1007/JHEP04(2011)029"},
        {"authors": "Jacobson, T.", "year": 1995, "title": "Thermodynamics of Spacetime: The Einstein Equation of State", "journal": "Phys. Rev. Lett.", "volume": "75", "pages": "1260", "doi": "10.1103/PhysRevLett.75.1260"},
        {"authors": "Bekenstein, J. D.", "year": 1973, "title": "Black holes and entropy", "journal": "Phys. Rev. D", "volume": "7", "pages": "2333", "doi": "10.1103/PhysRevD.7.2333"},
        {"authors": "Maldacena, J.", "year": 1998, "title": "The Large N limit of superconformal field theories and supergravity", "journal": "Adv. Theor. Math. Phys.", "volume": "2", "pages": "231", "doi": "10.4310/ATMP.1998.v2.n2.a1"},
    ],
    "H5": [
        {"authors": "Page, D. N., Wootters, W. K.", "year": 1983, "title": "Evolution without evolution: Dynamics described by stationary observables", "journal": "Phys. Rev. D", "volume": "27", "pages": "2885", "doi": "10.1103/PhysRevD.27.2885"},
        {"authors": "Rovelli, C.", "year": 1996, "title": "Relational quantum mechanics", "journal": "Int. J. Theor. Phys.", "volume": "35", "pages": "1637", "doi": "10.1007/BF02302261"},
        {"authors": "DeWitt, B. S.", "year": 1967, "title": "Quantum Theory of Gravity I: The Canonical Theory", "journal": "Phys. Rev.", "volume": "160", "pages": "1113", "doi": "10.1103/PhysRev.160.1113"},
    ],
    "H6": [
        {"authors": "Guth, A. H.", "year": 1981, "title": "Inflationary universe: A possible solution to the horizon and flatness problems", "journal": "Phys. Rev. D", "volume": "23", "pages": "347", "doi": "10.1103/PhysRevD.23.347"},
        {"authors": "Planck Collaboration", "year": 2020, "title": "Planck 2018 results VI: Cosmological parameters", "journal": "A&A", "volume": "641", "pages": "A6", "doi": "10.1051/0004-6361/201833910"},
        {"authors": "Riess, A. G. et al.", "year": 1998, "title": "Observational Evidence from Supernovae for an Accelerating Universe", "journal": "AJ", "volume": "116", "pages": "1009", "doi": "10.1086/300499"},
        {"authors": "Perlmutter, S. et al.", "year": 1999, "title": "Measurements of Ω and Λ from 42 High-Redshift Supernovae", "journal": "ApJ", "volume": "517", "pages": "565", "doi": "10.1086/307221"},
    ],
}


class MockLLMClient(LLMClient):
    """
    Deterministic mock client for testing and offline use.

    Responses are pre-defined and do not require any API key.
    The mock understands the task type from the user prompt and returns
    structured JSON matching what the real client would produce.
    """

    def __init__(self, model: str = "mock-v1") -> None:
        self._model = model
        self._call_count = 0

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def call_count(self) -> int:
        return self._call_count

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        self._call_count += 1

        # Route by keywords in the user prompt
        if "CONSISTENCY_CHECK" in user or "consistency between" in user.lower():
            return self._mock_consistency(user)
        if "VALIDATE_HYPOTHESIS" in user or "validate" in user.lower():
            return self._mock_validate(user)
        if "LITERATURE_MAP" in user or "literature" in user.lower():
            return self._mock_literature(user)
        if "BIBLIOGRAPHY" in user or "bibliography" in user.lower():
            return self._mock_bibliography(user)

        # Generic fallback
        return json.dumps({
            "verdict": "consistent",
            "rationale": "Mock response: no specific task detected.",
            "confidence": 0.5,
        })

    def _extract_ids(self, text: str) -> list[str]:
        return re.findall(r"\bH[1-6]\b", text)

    def _mock_consistency(self, user: str) -> str:
        ids = self._extract_ids(user)
        if len(ids) >= 2:
            key = (ids[0], ids[1])
            rev = (ids[1], ids[0])
            verdict, conf, rationale = _MOCK_CONSISTENCY_VERDICTS.get(
                key, _MOCK_CONSISTENCY_VERDICTS.get(rev, ("consistent", 0.5, "Mock: no entry found."))
            )
        else:
            verdict, conf, rationale = "undetermined", 0.3, "Could not identify hypothesis IDs."

        return json.dumps({
            "verdict": verdict,
            "rationale": rationale,
            "confidence": conf,
            "open_questions": [
                "What mathematical structure unifies both hypotheses?",
                "Is there a measurable prediction that distinguishes them?",
            ],
            "supporting_theories": ["GR", "QFT"],
        })

    def _mock_validate(self, user: str) -> str:
        return json.dumps({
            "verdict": "partial",
            "consistent_with": ["H3", "H1"],
            "tensions_with": ["H4"],
            "rationale": (
                "Mock validation: The new hypothesis shares topological features with H3 "
                "and is consistent with the active vacuum (H1). Tension with H4 because "
                "the mass scale is not derived from entropic arguments."
            ),
            "confidence": 0.6,
            "suggestions": [
                "Specify the field parameters connecting to the Standard Model mass scale.",
                "Check whether the proposed soliton produces an entropy gradient consistent with H4.",
            ],
        })

    def _mock_literature(self, user: str) -> str:
        ids = self._extract_ids(user)
        hid = ids[0] if ids else "H1"
        theories = _MOCK_LITERATURE.get(hid, _MOCK_LITERATURE["H1"])
        return json.dumps({"hypothesis": hid, "theories": theories})

    def _mock_bibliography(self, user: str) -> str:
        ids = self._extract_ids(user)
        hid = ids[0] if ids else "H1"
        entries = _MOCK_BIBLIOGRAPHY.get(hid, _MOCK_BIBLIOGRAPHY["H1"])
        return json.dumps({"hypothesis": hid, "entries": entries})


# ---------------------------------------------------------------------------
# Copilot / OpenAI-compatible client
# ---------------------------------------------------------------------------

class CopilotLLMClient(LLMClient):
    """
    GitHub Copilot Models API client (OpenAI-compatible).

    Requires:
    - openai Python package (`pip install openai`)
    - GITHUB_TOKEN or COPILOT_TOKEN environment variable

    The Copilot Models API endpoint: https://models.inference.ai.azure.com
    (GitHub Copilot uses Azure OpenAI under the hood via GitHub token auth.)
    """

    COPILOT_BASE_URL = "https://models.inference.ai.azure.com"
    DEFAULT_MODEL    = "gpt-4o"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError(
                "openai package required for CopilotLLMClient. "
                "Install with: pip install openai"
            ) from e

        token = (
            api_key
            or os.environ.get("GITHUB_TOKEN")
            or os.environ.get("COPILOT_TOKEN")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not token:
            raise ValueError(
                "No API token found. Set GITHUB_TOKEN, COPILOT_TOKEN, or OPENAI_API_KEY "
                "environment variable."
            )

        self._model = model or os.environ.get("LLM_MODEL", self.DEFAULT_MODEL)
        self._base_url = base_url or os.environ.get("LLM_BASE_URL", self.COPILOT_BASE_URL)
        self._client = OpenAI(api_key=token, base_url=self._base_url)

    @property
    def provider_name(self) -> str:
        return "copilot"

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Ollama client (OpenAI-compatible local)
# ---------------------------------------------------------------------------

class OllamaLLMClient(LLMClient):
    """
    Ollama local server via OpenAI-compatible API.

    Requires Ollama running at http://localhost:11434.
    Default model: qwen3:14b (as configured in opencode.json).
    """

    DEFAULT_BASE_URL = "http://localhost:11434/v1"
    DEFAULT_MODEL    = "qwen3:14b"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError("openai package required. Install with: pip install openai") from e

        self._model    = model    or os.environ.get("LLM_MODEL", self.DEFAULT_MODEL)
        self._base_url = base_url or os.environ.get("LLM_BASE_URL", self.DEFAULT_BASE_URL)
        self._client   = OpenAI(api_key="ollama", base_url=self._base_url)

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_client(
    provider: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> LLMClient:
    """
    Create and return an LLM client.

    Parameters
    ----------
    provider : str, optional
        One of: "mock", "copilot", "openai", "ollama".
        Defaults to the LLM_PROVIDER environment variable, then "mock".
    model : str, optional
        Model name override. Defaults to provider default.
    **kwargs
        Passed to the client constructor.

    Returns
    -------
    LLMClient
    """
    provider = provider or os.environ.get("LLM_PROVIDER", "mock")
    provider = provider.lower().strip()

    if provider == "mock":
        return MockLLMClient(model=model or "mock-v1")
    elif provider in ("copilot", "openai"):
        return CopilotLLMClient(model=model, **kwargs)
    elif provider == "ollama":
        return OllamaLLMClient(model=model, **kwargs)
    else:
        raise ValueError(
            f"Unknown provider: {provider!r}. "
            "Valid options: mock, copilot, openai, ollama."
        )
