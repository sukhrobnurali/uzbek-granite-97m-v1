from __future__ import annotations

import pytest

from utils.dataset_card import build_card


@pytest.fixture
def stats() -> dict:
    return {
        "repo_id": "sukhrobnurali/uzbek-embedding-pairs",
        "train_rows": 356278,
        "validation_rows": 1994,
        "retrieval_rows": 5000,
        "smoke_rows": 100,
        "source_distribution": {
            "parallel_opus": 121442,
            "wiki": 199208,
            "mixscript": 35628,
        },
        "mixscript_ratio": 0.10,
        "validation_cyrl_source": "transliterated",
        "near_dedup_threshold": 0.9,
        "generation_date": "2026-05-17",
    }


REQUIRED_HEADERS = [
    "## Dataset Structure",
    "## Schema",
    "## Source Datasets",
    "## Processing Pipeline",
    "## Validation Split",
    "## Intended Use",
    "## Limitations",
    "## Citations",
    "## License",
    "## Reproduction",
]


def test_yaml_frontmatter(stats):
    card = build_card(stats)
    assert card.startswith("---\n")
    assert "license: apache-2.0" in card
    assert "language:" in card
    assert "task_categories:" in card
    assert "sentence-similarity" in card


def test_required_headers_present(stats):
    card = build_card(stats)
    for header in REQUIRED_HEADERS:
        assert header in card, f"missing required header: {header}"


def test_bibtex_citations_present(stats):
    card = build_card(stats)
    assert "@inproceedings" in card or "@article" in card or "@misc" in card
    assert "opus-100" in card.lower() or "opus 100" in card.lower()
    assert "flores" in card.lower()


def test_reports_native_cyrl_source():
    stats = {
        "repo_id": "x/y", "train_rows": 1, "validation_rows": 1,
        "retrieval_rows": 1, "smoke_rows": 1,
        "source_distribution": {"parallel_opus": 1},
        "mixscript_ratio": 0.10, "validation_cyrl_source": "native",
        "near_dedup_threshold": 0.9, "generation_date": "2026-05-17",
    }
    card = build_card(stats)
    assert "native" in card.lower()


def test_reports_translit_cyrl_source():
    stats = {
        "repo_id": "x/y", "train_rows": 1, "validation_rows": 1,
        "retrieval_rows": 1, "smoke_rows": 1,
        "source_distribution": {"parallel_opus": 1},
        "mixscript_ratio": 0.10, "validation_cyrl_source": "transliterated",
        "near_dedup_threshold": 0.9, "generation_date": "2026-05-17",
    }
    card = build_card(stats)
    assert "transliterated" in card.lower()


def test_row_counts_consistent(stats):
    card = build_card(stats)
    assert "356,278" in card or "356278" in card
    assert "1,994" in card or "1994" in card
    assert "5,000" in card or "5000" in card


def test_limitations_section_lists_known_issues(stats):
    card = build_card(stats)
    limitations = card.split("## Limitations", 1)[1].split("##", 1)[0]
    assert "Е" in limitations or "E/" in limitations or "one-way" in limitations.lower()
    assert "opus" in limitations.lower() and "license" in limitations.lower()


def test_repo_id_appears(stats):
    card = build_card(stats)
    assert stats["repo_id"] in card
