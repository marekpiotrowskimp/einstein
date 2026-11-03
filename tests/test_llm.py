"""
Tests for einstein.llm — Phase 6 LLM integration.

All tests use MockLLMClient — no API key required.

Test coverage:
- client.py: MockLLMClient, get_client factory, _extract_json
- prompts.py: prompt rendering functions
- consistency.py: LLMConsistencyChecker (pair + validation)
- literature.py: LiteratureMapper, Bibliography, BibTeX
- report.py: generate_report with include_llm=True
"""

from __future__ import annotations

import json
import pytest

from einstein.llm.client import (
    MockLLMClient,
    get_client,
    _extract_json,
    LLMClient,
)
from einstein.llm.consistency import (
    LLMConsistencyChecker,
    LLMConsistencyResult,
    ValidationResult,
    VALID_VERDICTS,
    VALID_VALIDATION_VERDICTS,
)
from einstein.llm.literature import (
    LiteratureMapper,
    TheoryMapping,
    Bibliography,
    BibEntry,
    LiteratureReport,
    VALID_OVERLAP,
)
from einstein.llm import prompts
from einstein.hypotheses import build_registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def registry():
    return build_registry()


@pytest.fixture(scope="module")
def mock_client():
    return MockLLMClient()


@pytest.fixture(scope="module")
def consistency_checker(mock_client, registry):
    return LLMConsistencyChecker(mock_client, registry)


@pytest.fixture(scope="module")
def literature_mapper(mock_client, registry):
    return LiteratureMapper(mock_client, registry)


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_bare_json(self):
        text = '{"verdict": "consistent", "confidence": 0.8}'
        result = _extract_json(text)
        assert result["verdict"] == "consistent"
        assert result["confidence"] == 0.8

    def test_json_in_code_block(self):
        text = '```json\n{"verdict": "tension"}\n```'
        result = _extract_json(text)
        assert result["verdict"] == "tension"

    def test_json_in_unlabelled_block(self):
        text = '```\n{"verdict": "conflict"}\n```'
        result = _extract_json(text)
        assert result["verdict"] == "conflict"

    def test_json_surrounded_by_prose(self):
        text = 'Here is my analysis:\n{"verdict": "consistent", "confidence": 0.9}\nEnd.'
        result = _extract_json(text)
        assert result["verdict"] == "consistent"

    def test_invalid_json_returns_empty(self):
        result = _extract_json("This is not JSON at all.")
        assert result == {}

    def test_empty_string_returns_empty(self):
        result = _extract_json("")
        assert result == {}


# ---------------------------------------------------------------------------
# MockLLMClient
# ---------------------------------------------------------------------------

class TestMockLLMClient:
    def test_is_llm_client(self, mock_client):
        assert isinstance(mock_client, LLMClient)

    def test_provider_name(self, mock_client):
        assert mock_client.provider_name == "mock"

    def test_model_name(self, mock_client):
        assert mock_client.model_name == "mock-v1"

    def test_complete_returns_string(self, mock_client):
        result = mock_client.complete("system", "user message")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_complete_json_returns_dict(self, mock_client):
        result = mock_client.complete_json("system", "CONSISTENCY_CHECK H1 H2")
        assert isinstance(result, dict)

    def test_consistency_response_has_verdict(self, mock_client):
        raw = mock_client.complete("system", "CONSISTENCY_CHECK between H1 and H3")
        data = _extract_json(raw)
        assert "verdict" in data
        assert data["verdict"] in VALID_VERDICTS

    def test_validation_response_has_verdict(self, mock_client):
        raw = mock_client.complete("system", "VALIDATE_HYPOTHESIS: dark matter is a soliton")
        data = _extract_json(raw)
        assert "verdict" in data

    def test_literature_response_has_theories(self, mock_client):
        raw = mock_client.complete("system", "LITERATURE_MAP for H3")
        data = _extract_json(raw)
        assert "theories" in data

    def test_bibliography_response_has_entries(self, mock_client):
        raw = mock_client.complete("system", "BIBLIOGRAPHY for H4")
        data = _extract_json(raw)
        assert "entries" in data

    def test_call_count_increments(self):
        client = MockLLMClient()
        assert client.call_count == 0
        client.complete("s", "u")
        assert client.call_count == 1
        client.complete("s", "u")
        assert client.call_count == 2


# ---------------------------------------------------------------------------
# get_client factory
# ---------------------------------------------------------------------------

class TestGetClient:
    def test_default_is_mock(self):
        client = get_client()
        assert client.provider_name == "mock"

    def test_explicit_mock(self):
        client = get_client("mock")
        assert client.provider_name == "mock"

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_client("invalid_provider_xyz")

    def test_model_override(self):
        client = get_client("mock", model="test-model")
        assert client.model_name == "test-model"


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_consistency_prompt_contains_ids(self, registry):
        h1 = registry["H1"]
        h2 = registry["H3"]
        p = prompts.render_consistency_prompt(h1, h2)
        assert "H1" in p
        assert "H3" in p
        assert "CONSISTENCY_CHECK" in p

    def test_consistency_prompt_contains_statement(self, registry):
        h1 = registry["H1"]
        h2 = registry["H2"]
        p = prompts.render_consistency_prompt(h1, h2)
        assert h1.statement[:30] in p
        assert h2.statement[:30] in p

    def test_validation_prompt_contains_statement(self, registry):
        stmt = "Dark matter is a stable soliton of a new scalar field."
        p = prompts.render_validation_prompt(stmt, [], registry)
        assert "VALIDATE_HYPOTHESIS" in p
        assert stmt in p
        assert "H1" in p

    def test_validation_prompt_contains_all_hypotheses(self, registry):
        p = prompts.render_validation_prompt("Test hypothesis", [], registry)
        for hid in ["H1", "H2", "H3", "H4", "H5", "H6"]:
            assert hid in p

    def test_literature_prompt_contains_hypothesis(self, registry):
        h = registry["H4"]
        p = prompts.render_literature_map_prompt(h)
        assert "LITERATURE_MAP" in p
        assert "H4" in p
        assert h.name in p

    def test_bibliography_prompt_contains_hypothesis(self, registry):
        h = registry["H5"]
        p = prompts.render_bibliography_prompt(h)
        assert "BIBLIOGRAPHY" in p
        assert "H5" in p

    def test_system_prompts_are_non_empty(self):
        assert len(prompts.PHYSICS_EXPERT_SYSTEM) > 100
        assert len(prompts.LITERATURE_EXPERT_SYSTEM) > 100


# ---------------------------------------------------------------------------
# LLMConsistencyChecker — single pair
# ---------------------------------------------------------------------------

class TestLLMConsistencyCheckerPair:
    def test_check_pair_returns_result(self, consistency_checker):
        result = consistency_checker.check_pair("H1", "H2")
        assert isinstance(result, LLMConsistencyResult)

    def test_result_ids_correct(self, consistency_checker):
        result = consistency_checker.check_pair("H2", "H4")
        assert result.h1 == "H2"
        assert result.h2 == "H4"

    def test_verdict_is_valid(self, consistency_checker):
        for h1, h2 in [("H1", "H2"), ("H3", "H4"), ("H5", "H6")]:
            r = consistency_checker.check_pair(h1, h2)
            assert r.verdict in VALID_VERDICTS

    def test_confidence_in_range(self, consistency_checker):
        result = consistency_checker.check_pair("H1", "H3")
        assert 0.0 <= result.confidence <= 1.0

    def test_rationale_is_string(self, consistency_checker):
        result = consistency_checker.check_pair("H2", "H3")
        assert isinstance(result.rationale, str)

    def test_open_questions_is_list(self, consistency_checker):
        result = consistency_checker.check_pair("H4", "H5")
        assert isinstance(result.open_questions, list)

    def test_raw_response_is_string(self, consistency_checker):
        result = consistency_checker.check_pair("H1", "H6")
        assert isinstance(result.raw_response, str)
        assert len(result.raw_response) > 0

    @pytest.mark.parametrize("h1,h2", [
        ("H1", "H2"), ("H1", "H5"), ("H3", "H6"),
        ("H4", "H5"), ("H2", "H6"),
    ])
    def test_specific_pairs_have_valid_verdict(self, consistency_checker, h1, h2):
        r = consistency_checker.check_pair(h1, h2)
        assert r.verdict in VALID_VERDICTS


# ---------------------------------------------------------------------------
# LLMConsistencyChecker — full matrix
# ---------------------------------------------------------------------------

class TestLLMConsistencyCheckerMatrix:
    def test_full_matrix_returns_15_results(self, consistency_checker):
        results = consistency_checker.full_matrix()
        assert len(results) == 15

    def test_all_results_valid_verdicts(self, consistency_checker):
        for r in consistency_checker.full_matrix():
            assert r.verdict in VALID_VERDICTS

    def test_all_confidence_in_range(self, consistency_checker):
        for r in consistency_checker.full_matrix():
            assert 0.0 <= r.confidence <= 1.0

    def test_matrix_as_dict_has_all_ids(self, consistency_checker):
        mat = consistency_checker.matrix_as_dict()
        for hid in ["H1", "H2", "H3", "H4", "H5", "H6"]:
            assert hid in mat

    def test_matrix_is_symmetric(self, consistency_checker):
        mat = consistency_checker.matrix_as_dict()
        for h1 in mat:
            for h2, v in mat[h1].items():
                assert mat[h2][h1] == v, f"Asymmetric: {h1}↔{h2}"

    def test_summary_counts_sum_to_15(self, consistency_checker):
        counts = consistency_checker.summary()
        assert sum(counts.values()) == 15

    def test_some_consistent_pairs(self, consistency_checker):
        counts = consistency_checker.summary()
        assert counts.get("consistent", 0) > 0


# ---------------------------------------------------------------------------
# LLMConsistencyChecker — validation
# ---------------------------------------------------------------------------

class TestValidateNewHypothesis:
    def test_returns_validation_result(self, consistency_checker):
        vr = consistency_checker.validate_new_hypothesis(
            "Dark matter consists of stable φ⁶ solitons."
        )
        assert isinstance(vr, ValidationResult)

    def test_verdict_is_valid(self, consistency_checker):
        vr = consistency_checker.validate_new_hypothesis("Space has a discrete lattice structure.")
        assert vr.verdict in VALID_VALIDATION_VERDICTS

    def test_consistent_with_is_list_of_ids(self, consistency_checker):
        vr = consistency_checker.validate_new_hypothesis(
            "Gravity emerges from quantum entanglement entropy.",
            assumptions=["Holographic principle holds"],
        )
        for hid in vr.consistent_with:
            assert hid in {"H1", "H2", "H3", "H4", "H5", "H6"}

    def test_confidence_in_range(self, consistency_checker):
        vr = consistency_checker.validate_new_hypothesis("Test hypothesis.")
        assert 0.0 <= vr.confidence <= 1.0

    def test_suggestions_is_list(self, consistency_checker):
        vr = consistency_checker.validate_new_hypothesis("New hypothesis with assumptions.",
                                                          assumptions=["Assumption A"])
        assert isinstance(vr.suggestions, list)

    def test_novelty_is_valid(self, consistency_checker):
        from einstein.llm.consistency import VALID_NOVELTY
        vr = consistency_checker.validate_new_hypothesis("The universe is a simulation.")
        assert vr.novelty in VALID_NOVELTY

    def test_statement_preserved(self, consistency_checker):
        stmt = "This is a unique test hypothesis statement."
        vr = consistency_checker.validate_new_hypothesis(stmt)
        assert vr.statement == stmt


# ---------------------------------------------------------------------------
# LiteratureMapper — theory mapping
# ---------------------------------------------------------------------------

class TestLiteratureMapperMapping:
    @pytest.mark.parametrize("hid", ["H1", "H2", "H3", "H4", "H5", "H6"])
    def test_map_hypothesis_returns_mapping(self, literature_mapper, hid):
        mapping = literature_mapper.map_hypothesis(hid)
        assert isinstance(mapping, TheoryMapping)

    def test_mapping_has_correct_id(self, literature_mapper):
        mapping = literature_mapper.map_hypothesis("H4")
        assert mapping.hypothesis_id == "H4"

    def test_mapping_has_theories(self, literature_mapper):
        mapping = literature_mapper.map_hypothesis("H3")
        assert len(mapping.theories) >= 1

    def test_theory_overlap_is_valid(self, literature_mapper):
        mapping = literature_mapper.map_hypothesis("H1")
        for t in mapping.theories:
            assert t.overlap in VALID_OVERLAP

    def test_theory_has_name_and_connection(self, literature_mapper):
        mapping = literature_mapper.map_hypothesis("H2")
        for t in mapping.theories:
            assert len(t.name) > 0
            assert len(t.connection) > 0

    def test_map_all_returns_six(self, literature_mapper):
        all_mappings = literature_mapper.map_all()
        assert set(all_mappings.keys()) == {"H1", "H2", "H3", "H4", "H5", "H6"}

    def test_to_markdown_contains_hypothesis_id(self, literature_mapper):
        mapping = literature_mapper.map_hypothesis("H5")
        md = mapping.to_markdown()
        assert "H5" in md


# ---------------------------------------------------------------------------
# LiteratureMapper — bibliography
# ---------------------------------------------------------------------------

class TestLiteratureMapperBibliography:
    @pytest.mark.parametrize("hid", ["H1", "H2", "H3", "H4", "H5", "H6"])
    def test_generate_bibliography_returns_bib(self, literature_mapper, hid):
        bib = literature_mapper.generate_bibliography(hid)
        assert isinstance(bib, Bibliography)

    def test_bibliography_has_entries(self, literature_mapper):
        bib = literature_mapper.generate_bibliography("H4")
        assert len(bib.entries) >= 1

    def test_entry_has_required_fields(self, literature_mapper):
        bib = literature_mapper.generate_bibliography("H3")
        for e in bib.entries:
            assert isinstance(e, BibEntry)
            assert len(e.authors) > 0
            assert isinstance(e.year, int)
            assert len(e.title) > 0

    def test_bibtex_output_has_at_article(self, literature_mapper):
        bib = literature_mapper.generate_bibliography("H1")
        bibtex = bib.to_bibtex()
        assert "@" in bibtex

    def test_bibtex_contains_author(self, literature_mapper):
        bib = literature_mapper.generate_bibliography("H4")
        bibtex = bib.to_bibtex()
        assert "author" in bibtex

    def test_markdown_contains_hypothesis_id(self, literature_mapper):
        bib = literature_mapper.generate_bibliography("H6")
        md = bib.to_markdown()
        assert "H6" in md

    def test_bib_entry_key_format(self, literature_mapper):
        bib = literature_mapper.generate_bibliography("H4")
        for e in bib.entries:
            key = e.key
            assert len(key) > 0
            assert key.isalnum()

    def test_generate_all_bibliographies(self, literature_mapper):
        bibs = literature_mapper.generate_all_bibliographies()
        assert set(bibs.keys()) == {"H1", "H2", "H3", "H4", "H5", "H6"}


# ---------------------------------------------------------------------------
# LiteratureReport
# ---------------------------------------------------------------------------

class TestLiteratureReport:
    @pytest.fixture(scope="class")
    def full_report(self, literature_mapper):
        return literature_mapper.full_report()

    def test_full_report_has_all_mappings(self, full_report):
        assert set(full_report.mappings.keys()) == {"H1", "H2", "H3", "H4", "H5", "H6"}

    def test_full_report_has_all_bibliographies(self, full_report):
        assert set(full_report.bibliographies.keys()) == {"H1", "H2", "H3", "H4", "H5", "H6"}

    def test_to_markdown_contains_all_ids(self, full_report):
        md = full_report.to_markdown()
        for hid in ["H1", "H2", "H3", "H4", "H5", "H6"]:
            assert hid in md

    def test_to_bibtex_contains_entries(self, full_report):
        bib = full_report.to_bibtex()
        assert "@" in bib

    def test_bibtex_no_duplicate_keys(self, full_report):
        bibtex = full_report.to_bibtex()
        import re
        keys = re.findall(r"@\w+\{(\w+),", bibtex)
        assert len(keys) == len(set(keys)), "Duplicate BibTeX keys found"

    def test_save_bibtex(self, literature_mapper, tmp_path):
        out = tmp_path / "test.bib"
        literature_mapper.save_bibtex(str(out))
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "@" in content


# ---------------------------------------------------------------------------
# Report integration with LLM section
# ---------------------------------------------------------------------------

class TestReportWithLLM:
    def test_generate_report_with_llm_default_mock(self):
        from einstein.hypotheses.report import generate_report
        md = generate_report(include_llm=True)
        assert "LLM Analysis" in md
        assert "Pairwise Consistency Matrix" in md
        assert "Theory Mapping" in md
        assert "Bibliography" in md

    def test_generate_report_with_explicit_mock(self):
        from einstein.hypotheses.report import generate_report
        client = get_client("mock")
        md = generate_report(include_llm=True, llm_client=client)
        assert "Provider: `mock`" in md

    def test_generate_report_without_llm_has_no_llm_section(self):
        from einstein.hypotheses.report import generate_report
        md = generate_report(include_llm=False)
        assert "LLM Analysis" not in md

    def test_report_saves_with_llm_section(self, tmp_path):
        from einstein.hypotheses.report import generate_report
        out = tmp_path / "report_with_llm.md"
        md = generate_report(include_llm=True, output_path=str(out))
        assert out.exists()
        assert "LLM Analysis" in out.read_text(encoding="utf-8")

    def test_report_saves_bibtex(self, tmp_path):
        from einstein.hypotheses.report import generate_report
        bib_out = tmp_path / "refs.bib"
        generate_report(
            include_llm=True,
            bibtex_path=str(bib_out),
        )
        assert bib_out.exists()
        content = bib_out.read_text(encoding="utf-8")
        assert "@" in content
