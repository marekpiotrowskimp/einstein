"""
einstein.llm.literature
========================
LLM-powered literature mapper and bibliography generator.

Features
--------
- Map each hypothesis to related theories / research programs
- Generate structured bibliography entries with BibTeX export
- Combine into a full literature report for all H1–H6

Usage
-----
from einstein.llm.client import get_client
from einstein.llm.literature import LiteratureMapper
from einstein.hypotheses import build_registry

client   = get_client()
registry = build_registry()
mapper   = LiteratureMapper(client, registry)

# Map one hypothesis to related theories
mapping = mapper.map_hypothesis("H4")
for t in mapping.theories:
    print(t.name, "—", t.overlap, "—", t.connection)

# Generate bibliography
bib = mapper.generate_bibliography("H4")
print(bib.to_markdown())
print(bib.to_bibtex())

# Full report for all hypotheses
report = mapper.full_report()
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from einstein.llm.client import LLMClient, _extract_json, get_client
from einstein.llm.prompts import (
    LITERATURE_EXPERT_SYSTEM,
    PHYSICS_EXPERT_SYSTEM,
    render_bibliography_prompt,
    render_literature_map_prompt,
)
from einstein.hypotheses import build_registry
from einstein.hypotheses.specification import HypothesisRegistry


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

VALID_OVERLAP = {"direct", "strong", "moderate", "weak"}


@dataclass
class TheoryEntry:
    """A single theory or research program related to a hypothesis."""
    name: str
    connection: str
    overlap: str                        # "direct" | "strong" | "moderate" | "weak"
    key_authors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.overlap not in VALID_OVERLAP:
            self.overlap = "moderate"


@dataclass
class TheoryMapping:
    """
    Mapping of a hypothesis to related theories.

    Attributes
    ----------
    hypothesis_id : str
    theories : list[TheoryEntry]
    summary : str
        1-2 sentence overall assessment.
    raw_response : str
    """
    hypothesis_id: str
    theories: list[TheoryEntry]
    summary: str = ""
    raw_response: str = ""

    def to_markdown(self) -> str:
        lines = [f"### Theory Mapping — {self.hypothesis_id}", ""]
        for t in self.theories:
            lines.append(f"**{t.name}** ({t.overlap})")
            lines.append(f"> {t.connection}")
            if t.key_authors:
                lines.append(f"> Authors: {', '.join(t.key_authors)}")
            lines.append("")
        if self.summary:
            lines += [f"**Summary:** {self.summary}", ""]
        return "\n".join(lines)


@dataclass
class BibEntry:
    """A single bibliography entry."""
    authors: str
    year: int
    title: str
    journal: str
    volume: str = ""
    pages: str = ""
    doi: str = ""
    relevance: str = ""

    # Auto-generated BibTeX key
    @property
    def key(self) -> str:
        """Generate a BibTeX citation key: FirstAuthorLastNameYear."""
        first_author = self.authors.split(",")[0].split("and")[0].strip()
        # Keep only letters
        name_part = re.sub(r"[^A-Za-z]", "", first_author)[:12]
        return f"{name_part}{self.year}"

    def to_bibtex(self) -> str:
        entry_type = "book" if not self.volume and not self.pages else "article"
        lines = [f"@{entry_type}{{{self.key},"]
        lines.append(f"  author  = {{{self.authors}}},")
        lines.append(f"  year    = {{{self.year}}},")
        lines.append(f"  title   = {{{self.title}}},")
        lines.append(f"  journal = {{{self.journal}}},")
        if self.volume:
            lines.append(f"  volume  = {{{self.volume}}},")
        if self.pages:
            lines.append(f"  pages   = {{{self.pages}}},")
        if self.doi:
            lines.append(f"  doi     = {{{self.doi}}},")
        lines.append("}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        parts = [f"**{self.authors}** ({self.year})."]
        parts.append(f"*{self.title}.*")
        parts.append(f"{self.journal}")
        if self.volume:
            parts.append(f"**{self.volume}**")
        if self.pages:
            parts.append(f":{self.pages}.")
        if self.doi:
            parts.append(f"DOI: [{self.doi}](https://doi.org/{self.doi})")
        if self.relevance:
            parts.append(f"— *{self.relevance}*")
        return " ".join(parts)


@dataclass
class Bibliography:
    """
    Bibliography for a single hypothesis.

    Attributes
    ----------
    hypothesis_id : str
    entries : list[BibEntry]
    raw_response : str
    """
    hypothesis_id: str
    entries: list[BibEntry]
    raw_response: str = ""

    def to_bibtex(self) -> str:
        """Export all entries as a BibTeX string."""
        return "\n\n".join(e.to_bibtex() for e in self.entries)

    def to_markdown(self) -> str:
        """Export as a Markdown reference list."""
        lines = [f"### Bibliography — {self.hypothesis_id}", ""]
        for i, e in enumerate(self.entries, 1):
            lines.append(f"{i}. {e.to_markdown()}")
        return "\n".join(lines)


@dataclass
class LiteratureReport:
    """
    Full literature report for all hypotheses H1–H6.

    Attributes
    ----------
    mappings : dict[str, TheoryMapping]
    bibliographies : dict[str, Bibliography]
    """
    mappings: dict[str, TheoryMapping]
    bibliographies: dict[str, Bibliography]

    def to_markdown(self) -> str:
        sections = ["# Literature Report — Hypotheses H1–H6", ""]
        for hid in sorted(self.mappings.keys()):
            sections.append(f"## {hid}")
            sections.append("")
            sections.append(self.mappings[hid].to_markdown())
            sections.append(self.bibliographies[hid].to_markdown())
            sections.append("")
        return "\n".join(sections)

    def to_bibtex(self) -> str:
        """Export all bibliographies as one combined .bib file."""
        seen_keys: set[str] = set()
        entries: list[str] = []
        for bib in self.bibliographies.values():
            for e in bib.entries:
                k = e.key
                if k not in seen_keys:
                    seen_keys.add(k)
                    entries.append(e.to_bibtex())
        return "\n\n".join(entries)


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------

def _parse_theory_mapping(data: dict, hypothesis_id: str, raw: str) -> TheoryMapping:
    theories = []
    for t in data.get("theories", []):
        if isinstance(t, dict):
            theories.append(TheoryEntry(
                name=str(t.get("name", "Unknown")),
                connection=str(t.get("connection", "")),
                overlap=str(t.get("overlap", "moderate")),
                key_authors=list(t.get("key_authors", [])),
            ))
    return TheoryMapping(
        hypothesis_id=hypothesis_id,
        theories=theories,
        summary=str(data.get("summary", "")),
        raw_response=raw,
    )


def _parse_bibliography(data: dict, hypothesis_id: str, raw: str) -> Bibliography:
    entries = []
    for e in data.get("entries", []):
        if isinstance(e, dict):
            try:
                year = int(e.get("year", 0))
            except (ValueError, TypeError):
                year = 0
            entries.append(BibEntry(
                authors=str(e.get("authors", "")),
                year=year,
                title=str(e.get("title", "")),
                journal=str(e.get("journal", "")),
                volume=str(e.get("volume", "")),
                pages=str(e.get("pages", "")),
                doi=str(e.get("doi", "")),
                relevance=str(e.get("relevance", "")),
            ))
    return Bibliography(
        hypothesis_id=hypothesis_id,
        entries=entries,
        raw_response=raw,
    )


# ---------------------------------------------------------------------------
# LiteratureMapper
# ---------------------------------------------------------------------------

class LiteratureMapper:
    """
    Map hypotheses to related theories and generate bibliographies.

    Parameters
    ----------
    client : LLMClient
        LLM client to use.
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

    # ── Theory mapping ─────────────────────────────────────────────────────

    def map_hypothesis(self, hypothesis_id: str) -> TheoryMapping:
        """
        Map a single hypothesis to related theories.

        Parameters
        ----------
        hypothesis_id : str
            E.g. "H3".

        Returns
        -------
        TheoryMapping
        """
        h      = self._registry[hypothesis_id]
        prompt = render_literature_map_prompt(h)
        raw    = self._client.complete(PHYSICS_EXPERT_SYSTEM, prompt)
        data   = _extract_json(raw)
        return _parse_theory_mapping(data, hypothesis_id, raw)

    def map_all(self) -> dict[str, TheoryMapping]:
        """Map all six hypotheses."""
        return {hid: self.map_hypothesis(hid) for hid in self.ALL_IDS}

    # ── Bibliography ───────────────────────────────────────────────────────

    def generate_bibliography(self, hypothesis_id: str) -> Bibliography:
        """
        Generate a bibliography for a single hypothesis.

        Parameters
        ----------
        hypothesis_id : str

        Returns
        -------
        Bibliography
        """
        h      = self._registry[hypothesis_id]
        prompt = render_bibliography_prompt(h)
        raw    = self._client.complete(LITERATURE_EXPERT_SYSTEM, prompt)
        data   = _extract_json(raw)
        return _parse_bibliography(data, hypothesis_id, raw)

    def generate_all_bibliographies(self) -> dict[str, Bibliography]:
        """Generate bibliographies for all six hypotheses."""
        return {hid: self.generate_bibliography(hid) for hid in self.ALL_IDS}

    # ── Full report ────────────────────────────────────────────────────────

    def full_report(self) -> LiteratureReport:
        """
        Generate a full literature report for all H1–H6.

        Returns
        -------
        LiteratureReport
            Contains theory mappings and bibliographies for all hypotheses.
        """
        mappings       = self.map_all()
        bibliographies = self.generate_all_bibliographies()
        return LiteratureReport(mappings=mappings, bibliographies=bibliographies)

    def save_bibtex(self, path: str) -> None:
        """Generate all bibliographies and save as a .bib file."""
        report = self.full_report()
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(report.to_bibtex(), encoding="utf-8")
