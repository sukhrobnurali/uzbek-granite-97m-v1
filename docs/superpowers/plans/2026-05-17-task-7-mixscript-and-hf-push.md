# Task 7 — Mix-script Augmentation + HF Dataset Push Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-05-17-task-7-mixscript-and-hf-push-design.md` (commit 29d42a2)

**Goal:** Add 10% Cyrillic↔Latin mix-script augmentation to the training pool, build a FLORES-dev validation split, and ship `sukhrobnurali/uzbek-embedding-pairs` to Hugging Face with three configs (`default`, `wiki_retrieval_eval`, `smoke_100`) and a populated dataset card.

**Architecture:** Extend `prepare_data.py` with three pure builder functions (`augment_mixscript`, `build_validation`, `build_smoke_100`) plus a `push_to_hf` orchestrator. Extract the dataset card markdown into `utils/dataset_card.py` (keeps `prepare_data.py` under the 300-line threshold). Two-flag gate (`--push` + interactive `yes` confirm; `--allow-overwrite` for existing repos) makes the irreversible HF call hard to fire by accident. All new code is TDD-driven; tests mock HF and `huggingface_hub.HfApi` so the suite stays offline.

**Tech Stack:** Python 3.12, `datasets>=3.0`, `huggingface_hub>=0.26`, `pytest`, `ruff`. `utils.translit.to_latin` / `to_cyrillic` (already built in Task 2) for the transliteration. `utils.dataset_io.load_source` for FLORES dev.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `prepare_data.py` | extend | Add `augment_mixscript`, `build_validation`, `build_smoke_100`, `_print_preflight`, `_perform_pushes`, `push_to_hf`, new CLI flags. Target <300 lines after changes. |
| `utils/dataset_card.py` | new | `build_card(stats: dict) -> str` — renders the dataset card markdown from computed stats. |
| `tests/test_prepare_data.py` | extend | Mixscript, validation, smoke_100, push-preflight tests. |
| `tests/test_dataset_card.py` | new | Card structure + dynamic-content tests. |

---

## Task 1: Mix-script augmentation

**Files:**
- Modify: `prepare_data.py` (add `augment_mixscript` function near top, after imports)
- Test: `tests/test_prepare_data.py` (add new test class `TestMixscriptAug`)

- [ ] **Step 1.1: Write the failing tests**

Add to `tests/test_prepare_data.py`:

```python
from datasets import Dataset

import prepare_data


class TestMixscriptAug:
    @staticmethod
    def _pool(n: int, anchor_lang: str = "uz_Latn", source: str = "parallel_opus") -> Dataset:
        return Dataset.from_list([
            {
                "anchor": f"Toshkent shahar markazi {i}",
                "positive": f"Tashkent city center {i}",
                "source": source,
                "anchor_lang": anchor_lang,
                "positive_lang": "en",
            }
            for i in range(n)
        ])

    def test_target_ratio_about_10pct(self):
        pool = self._pool(100)
        aug = prepare_data.augment_mixscript(pool, target_ratio=0.10, seed=42)
        # n / (100 + n) ≈ 0.10 → n ≈ 11
        assert 9 <= len(aug) <= 13

    def test_deterministic(self):
        pool = self._pool(100)
        aug1 = prepare_data.augment_mixscript(pool, target_ratio=0.10, seed=42)
        aug2 = prepare_data.augment_mixscript(pool, target_ratio=0.10, seed=42)
        assert list(aug1["anchor"]) == list(aug2["anchor"])
        assert list(aug1["positive"]) == list(aug2["positive"])

    def test_flips_script_and_tags_source(self):
        pool = self._pool(100)
        aug = prepare_data.augment_mixscript(pool, target_ratio=0.10, seed=42)
        assert len(aug) > 0
        for row in aug:
            assert row["source"] == "mixscript"
            assert row["anchor_lang"] != row["positive_lang"]
            assert row["anchor_lang"] in {"uz_Latn", "uz_Cyrl"}
            assert row["positive_lang"] in {"uz_Latn", "uz_Cyrl"}

    def test_excludes_non_uzbek_anchors(self):
        # anchor_lang="en" rows are not eligible for augmentation
        pool = self._pool(100, anchor_lang="en")
        aug = prepare_data.augment_mixscript(pool, target_ratio=0.10, seed=42)
        assert len(aug) == 0

    def test_handles_cyrillic_anchors(self):
        pool = Dataset.from_list([
            {
                "anchor": "Тошкент шаҳар",
                "positive": "Tashkent city",
                "source": "wiki",
                "anchor_lang": "uz_Cyrl",
                "positive_lang": "en",
            }
            for _ in range(100)
        ])
        aug = prepare_data.augment_mixscript(pool, target_ratio=0.10, seed=42)
        assert len(aug) > 0
        for row in aug:
            # Cyrillic anchors should produce Latin positives
            assert row["anchor_lang"] == "uz_Cyrl"
            assert row["positive_lang"] == "uz_Latn"
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m pytest tests/test_prepare_data.py::TestMixscriptAug -v`

Expected: 5 tests fail with `AttributeError: module 'prepare_data' has no attribute 'augment_mixscript'`.

- [ ] **Step 1.3: Implement `augment_mixscript`**

Add to `prepare_data.py` after the existing imports and before `SOURCES_DEFAULT`:

```python
import random as _random

from utils.translit import to_cyrillic, to_latin, auto_detect_script


def augment_mixscript(
    pool: Dataset,
    target_ratio: float = 0.10,
    seed: int = 42,
    oversample_factor: float = 1.05,
) -> Dataset:
    """Generate mix-script training pairs by transliterating Uzbek anchors.

    Samples eligible rows (anchor_lang in {uz_Latn, uz_Cyrl}) stratified by source,
    transliterates the anchor to the opposite script, and emits
    {anchor: original, positive: transliterated, source: "mixscript", ...}.

    Target satisfies n_aug / (len(pool) + n_aug) ≈ target_ratio.
    """
    eligible_indices = [
        i for i, lang in enumerate(pool["anchor_lang"]) if lang in {"uz_Latn", "uz_Cyrl"}
    ]
    if not eligible_indices:
        return Dataset.from_list([])

    n_target = round(len(pool) * target_ratio / (1.0 - target_ratio))
    n_to_sample = min(int(n_target * oversample_factor), len(eligible_indices))

    rng = _random.Random(seed)
    by_source: dict[str, list[int]] = {}
    for idx in eligible_indices:
        src = pool["source"][idx]
        by_source.setdefault(src, []).append(idx)

    sampled: list[int] = []
    for src, idxs in by_source.items():
        per_source = round(n_to_sample * len(idxs) / len(eligible_indices))
        per_source = min(per_source, len(idxs))
        sampled.extend(rng.sample(idxs, per_source))
    rng.shuffle(sampled)

    rows: list[dict] = []
    for idx in sampled:
        anchor = pool["anchor"][idx]
        anchor_lang = pool["anchor_lang"][idx]
        if anchor_lang == "uz_Latn":
            positive = to_cyrillic(anchor)
        else:
            positive = to_latin(anchor)
        if not positive or positive == anchor:
            continue
        new_anchor_lang = auto_detect_script(anchor)
        new_positive_lang = auto_detect_script(positive)
        if new_anchor_lang == new_positive_lang:
            continue
        if new_anchor_lang not in {"uz_Latn", "uz_Cyrl"} or new_positive_lang not in {"uz_Latn", "uz_Cyrl"}:
            continue
        rows.append({
            "anchor": anchor,
            "positive": positive,
            "source": "mixscript",
            "anchor_lang": new_anchor_lang,
            "positive_lang": new_positive_lang,
        })
        if len(rows) >= n_target:
            break

    return Dataset.from_list(rows)
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m pytest tests/test_prepare_data.py::TestMixscriptAug -v`

Expected: 5 passed.

- [ ] **Step 1.5: Run the full suite + ruff**

Run: `c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m pytest tests/ -q && c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m ruff check .`

Expected: 109 prior tests + 5 new = 114 passed, ruff clean.

- [ ] **Step 1.6: Commit**

```bash
cd c:/projects/portfolio/HF/1-embedding
git add prepare_data.py tests/test_prepare_data.py
git commit -m "feat(prepare_data): 10% mix-script augmentation via utils.translit"
```

---

## Task 2: Validation split builder

**Files:**
- Modify: `prepare_data.py` (add `build_validation` function after `augment_mixscript`)
- Test: `tests/test_prepare_data.py` (add `TestBuildValidation` class)

- [ ] **Step 2.1: Write the failing tests**

Add to `tests/test_prepare_data.py`:

```python
class TestBuildValidation:
    @staticmethod
    def _latn() -> Dataset:
        return Dataset.from_list([
            {
                "anchor": f"Toshkent shahar {i}",
                "positive": f"Tashkent city {i}",
                "source": "flores_dev_latn",
                "anchor_lang": "uz_Latn",
                "positive_lang": "en",
            }
            for i in range(100)
        ])

    @staticmethod
    def _cyrl() -> Dataset:
        return Dataset.from_list([
            {
                "anchor": f"Тошкент шаҳар {i}",
                "positive": f"Tashkent city {i}",
                "source": "flores_dev_cyrl",
                "anchor_lang": "uz_Cyrl",
                "positive_lang": "en",
            }
            for i in range(100)
        ])

    def test_native_cyrl_when_available(self, monkeypatch):
        latn = self._latn()
        cyrl = self._cyrl()

        def fake_load(name, **kwargs):
            return {"flores_dev_latn": latn, "flores_dev_cyrl": cyrl}.get(name)

        monkeypatch.setattr(prepare_data.dataset_io, "load_source", fake_load)
        val, cyrl_source = prepare_data.build_validation()

        assert cyrl_source == "native"
        sources = set(val["source"])
        assert "flores_dev_latn" in sources
        assert "flores_dev_cyrl" in sources
        assert "flores_dev_cyrl_translit" not in sources
        assert len(val) == 200

    def test_translit_fallback_when_cyrl_missing(self, monkeypatch):
        latn = self._latn()

        def fake_load(name, **kwargs):
            if name == "flores_dev_latn":
                return latn
            if name == "flores_dev_cyrl":
                return None
            raise KeyError(name)

        monkeypatch.setattr(prepare_data.dataset_io, "load_source", fake_load)
        val, cyrl_source = prepare_data.build_validation()

        assert cyrl_source == "transliterated"
        sources = set(val["source"])
        assert "flores_dev_latn" in sources
        assert "flores_dev_cyrl_translit" in sources
        assert "flores_dev_cyrl" not in sources
        translit_rows = val.filter(lambda r: r["source"] == "flores_dev_cyrl_translit")
        assert len(translit_rows) == 100
        for row in translit_rows:
            assert row["anchor_lang"] == "uz_Cyrl"

    def test_raises_when_latn_missing(self, monkeypatch):
        def fake_load(name, **kwargs):
            return None

        monkeypatch.setattr(prepare_data.dataset_io, "load_source", fake_load)
        with pytest.raises(RuntimeError, match="flores_dev_latn"):
            prepare_data.build_validation()
```

Make sure `pytest` is imported at the top of the test file (it likely already is).

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m pytest tests/test_prepare_data.py::TestBuildValidation -v`

Expected: 3 tests fail with `AttributeError: module 'prepare_data' has no attribute 'build_validation'`.

- [ ] **Step 2.3: Implement `build_validation`**

Add to `prepare_data.py` after `augment_mixscript`:

```python
def build_validation() -> tuple[Dataset, str]:
    """Build the FLORES-dev validation split.

    Returns (validation_dataset, cyrl_source) where cyrl_source is "native"
    if FLORES uzn_Cyrl-eng_Latn loaded, else "transliterated" (we synthesize
    Cyrillic anchors from the Latn dev split using utils.translit.to_cyrillic).
    """
    latn = dataset_io.load_source("flores_dev_latn")
    if latn is None:
        raise RuntimeError(
            "flores_dev_latn (Muennighoff/flores200, uzn_Latn-eng_Latn dev) failed to load; "
            "cannot build validation split."
        )
    cyrl = dataset_io.load_source("flores_dev_cyrl")
    if cyrl is not None:
        validation = concatenate_datasets([latn, cyrl])
        return validation, "native"

    def _translit_row(row: dict) -> dict:
        new_anchor = to_cyrillic(row["anchor"])
        return {
            "anchor": new_anchor,
            "positive": row["positive"],
            "source": "flores_dev_cyrl_translit",
            "anchor_lang": auto_detect_script(new_anchor),
            "positive_lang": row["positive_lang"],
        }

    translit = latn.map(_translit_row)
    validation = concatenate_datasets([latn, translit])
    return validation, "transliterated"
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m pytest tests/test_prepare_data.py::TestBuildValidation -v`

Expected: 3 passed.

- [ ] **Step 2.5: Run full suite + ruff**

Run: `c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m pytest tests/ -q && c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m ruff check .`

Expected: 117 passed, ruff clean.

- [ ] **Step 2.6: Commit**

```bash
cd c:/projects/portfolio/HF/1-embedding
git add prepare_data.py tests/test_prepare_data.py
git commit -m "feat(prepare_data): FLORES-dev validation split with translit fallback"
```

---

## Task 3: `smoke_100` stratified sample

**Files:**
- Modify: `prepare_data.py` (add `build_smoke_100`)
- Test: `tests/test_prepare_data.py` (add `TestSmoke100`)

- [ ] **Step 3.1: Write the failing test**

Add to `tests/test_prepare_data.py`:

```python
class TestSmoke100:
    def test_size_and_strata(self):
        pool = Dataset.from_list(
            [{"anchor": f"a{i}", "positive": f"b{i}", "source": "parallel_opus",
              "anchor_lang": "uz_Latn", "positive_lang": "en"} for i in range(341)] +
            [{"anchor": f"c{i}", "positive": f"d{i}", "source": "wiki",
              "anchor_lang": "uz_Latn", "positive_lang": "uz_Latn"} for i in range(559)] +
            [{"anchor": f"e{i}", "positive": f"f{i}", "source": "mixscript",
              "anchor_lang": "uz_Latn", "positive_lang": "uz_Cyrl"} for i in range(100)]
        )
        smoke = prepare_data.build_smoke_100(pool, seed=42)
        assert len(smoke) == 100
        counts: dict[str, int] = {}
        for r in smoke:
            counts[r["source"]] = counts.get(r["source"], 0) + 1
        # proportional targets: 34 parallel_opus, 56 wiki, 10 mixscript
        assert abs(counts.get("parallel_opus", 0) - 34) <= 3
        assert abs(counts.get("wiki", 0) - 56) <= 3
        assert abs(counts.get("mixscript", 0) - 10) <= 3

    def test_deterministic(self):
        pool = Dataset.from_list(
            [{"anchor": f"a{i}", "positive": f"b{i}", "source": "parallel_opus",
              "anchor_lang": "uz_Latn", "positive_lang": "en"} for i in range(500)] +
            [{"anchor": f"c{i}", "positive": f"d{i}", "source": "wiki",
              "anchor_lang": "uz_Latn", "positive_lang": "uz_Latn"} for i in range(500)]
        )
        s1 = prepare_data.build_smoke_100(pool, seed=42)
        s2 = prepare_data.build_smoke_100(pool, seed=42)
        assert list(s1["anchor"]) == list(s2["anchor"])

    def test_pool_smaller_than_100_returns_all(self):
        pool = Dataset.from_list(
            [{"anchor": f"a{i}", "positive": f"b{i}", "source": "parallel_opus",
              "anchor_lang": "uz_Latn", "positive_lang": "en"} for i in range(50)]
        )
        smoke = prepare_data.build_smoke_100(pool, seed=42)
        assert len(smoke) == 50
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m pytest tests/test_prepare_data.py::TestSmoke100 -v`

Expected: 3 tests fail with `AttributeError`.

- [ ] **Step 3.3: Implement `build_smoke_100`**

Add to `prepare_data.py` after `build_validation`:

```python
def build_smoke_100(pool: Dataset, seed: int = 42, target_size: int = 100) -> Dataset:
    """Return a stratified-by-source sample of `target_size` rows (or all rows if smaller)."""
    if len(pool) <= target_size:
        return pool

    by_source: dict[str, list[int]] = {}
    for i, src in enumerate(pool["source"]):
        by_source.setdefault(src, []).append(i)

    rng = _random.Random(seed)
    sampled: list[int] = []
    for src, idxs in by_source.items():
        n = max(1, round(target_size * len(idxs) / len(pool)))
        n = min(n, len(idxs))
        sampled.extend(rng.sample(idxs, n))

    if len(sampled) > target_size:
        sampled = rng.sample(sampled, target_size)
    elif len(sampled) < target_size:
        remaining = [i for i in range(len(pool)) if i not in set(sampled)]
        sampled.extend(rng.sample(remaining, target_size - len(sampled)))

    sampled.sort()
    return pool.select(sampled)
```

- [ ] **Step 3.4: Run tests to verify they pass**

Run: `c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m pytest tests/test_prepare_data.py::TestSmoke100 -v`

Expected: 3 passed.

- [ ] **Step 3.5: Run full suite + ruff**

Run: `c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m pytest tests/ -q && c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m ruff check .`

Expected: 120 passed, ruff clean.

- [ ] **Step 3.6: Commit**

```bash
cd c:/projects/portfolio/HF/1-embedding
git add prepare_data.py tests/test_prepare_data.py
git commit -m "feat(prepare_data): stratified smoke_100 sample"
```

---

## Task 4: Dataset card builder

**Files:**
- Create: `utils/dataset_card.py`
- Test: `tests/test_dataset_card.py`

- [ ] **Step 4.1: Write the failing tests**

Create `tests/test_dataset_card.py`:

```python
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
    # Numbers may render with or without thousands separators
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
```

- [ ] **Step 4.2: Run tests to verify they fail**

Run: `c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m pytest tests/test_dataset_card.py -v`

Expected: All tests fail with `ModuleNotFoundError: No module named 'utils.dataset_card'`.

- [ ] **Step 4.3: Implement `utils/dataset_card.py`**

Create `utils/dataset_card.py`:

```python
"""Dataset card builder for sukhrobnurali/uzbek-embedding-pairs.

Pure function: stats dict in, markdown string out. Kept in its own module so the
~200-line template doesn't bloat prepare_data.py.
"""

from __future__ import annotations


def _fmt(n: int) -> str:
    return f"{n:,}"


def build_card(stats: dict) -> str:
    repo = stats["repo_id"]
    train = stats["train_rows"]
    val = stats["validation_rows"]
    retr = stats["retrieval_rows"]
    smoke = stats["smoke_rows"]
    dist = stats["source_distribution"]
    mix_ratio_pct = stats["mixscript_ratio"] * 100
    cyrl_source = stats["validation_cyrl_source"]
    dedup_threshold = stats["near_dedup_threshold"]
    gen_date = stats["generation_date"]

    dist_rows = "\n".join(
        f"| `{src}` | {_fmt(count)} | {count / train * 100:.1f}% |"
        for src, count in sorted(dist.items(), key=lambda x: -x[1])
    )

    if cyrl_source == "native":
        cyrl_note = (
            "Cyrillic validation rows are **native FLORES-200** `uzn_Cyrl-eng_Latn` dev. "
            "Both Latin and Cyrillic anchors are professional translations."
        )
    else:
        cyrl_note = (
            "Cyrillic validation rows are **transliterated** from the Latin dev split via "
            "`utils.translit.to_cyrillic` (the public FLORES-200 release did not include "
            "`uzn_Cyrl-eng_Latn` at build time). Rows have `source=\"flores_dev_cyrl_translit\"` "
            "for transparency."
        )

    return f"""---
license: apache-2.0
language:
  - uz
  - en
multilinguality: multilingual
size_categories:
  - 100K<n<1M
task_categories:
  - sentence-similarity
  - feature-extraction
tags:
  - uzbek
  - embedding
  - sentence-embedding
  - parallel-corpus
  - mixed-script
pretty_name: Uzbek Embedding Pairs
configs:
  - config_name: default
    data_files:
      - split: train
        path: default/train-*
      - split: validation
        path: default/validation-*
  - config_name: wiki_retrieval_eval
    data_files:
      - split: test
        path: wiki_retrieval_eval/test-*
  - config_name: smoke_100
    data_files:
      - split: train
        path: smoke_100/train-*
---

# Uzbek Embedding Pairs

Parallel and monolingual Uzbek sentence pairs for training sentence-embedding models.
Combines OPUS-100, Uzbek Wikipedia, FLORES-200 dev, and 10% Cyrillic↔Latin mix-script
augmentation. Generated {gen_date}.

Repository: `{repo}`

## Dataset Structure

| Config | Split | Rows | Use |
|---|---|---|---|
| `default` | `train` | {_fmt(train)} | Training pool (parallel + monolingual + mixscript aug) |
| `default` | `validation` | {_fmt(val)} | In-loop eval during training (FLORES-200 dev) |
| `wiki_retrieval_eval` | `test` | {_fmt(retr)} | Held-out Wikipedia title↔paragraph pairs for retrieval eval |
| `smoke_100` | `train` | {_fmt(smoke)} | Stratified 100-row sample for pipeline integrity tests |

## Schema

Every row has the same five columns:

| Column | Type | Description |
|---|---|---|
| `anchor` | string | First text in the pair (typically Uzbek for parallel sources, title for wiki) |
| `positive` | string | Paired text (typically English translation, or wiki paragraph, or transliterated Uzbek for mixscript) |
| `source` | string | Origin tag — see "Source Datasets" below |
| `anchor_lang` | string | `uz_Latn` \\| `uz_Cyrl` \\| `en` \\| `mixed` \\| `empty` (auto-detected) |
| `positive_lang` | string | Same vocabulary as `anchor_lang` |

## Source Datasets

| Source | HF ID | License | Role |
|---|---|---|---|
| OPUS-100 en-uz | `Helsinki-NLP/opus-100` | unknown (flagged below) | Parallel sentences (subset, after dedup) |
| Parallel-sentences-OPUS | `sentence-transformers/parallel-sentences-opus-100` | unknown | Cleaner OPUS variant (preferred on dedup collisions) |
| Uzbek Wikipedia | `yakhyo/uz-wiki` | MIT | Title↔first-80-words pairs |
| FLORES-200 dev | `Muennighoff/flores200` | CC-BY-SA-4.0 | Validation split only (uzn_Latn-eng_Latn + uzn_Cyrl-eng_Latn or transliterated) |

### Source distribution in `default/train`

| Source | Rows | Share |
|---|---|---|
{dist_rows}

## Processing Pipeline

1. Load each source with `utils.dataset_io.load_source(name)` → unified 5-column schema.
2. Normalize Unicode to NFC; canonicalize apostrophes (`oʻ`, `gʻ` markers).
3. Filter: length 5–512 chars per side (2000 for wiki paragraphs), word count ≥ 2, length ratio 0.4–2.5 for parallel sources, drop identity rows, Uzbek-side script-sanity check.
4. Exact dedup on `(anchor.lower(), positive.lower())`.
5. Near dedup via MinHash LSH (joint key `anchor || positive`, threshold {dedup_threshold:.2f}, num_perm 64). Joint key chosen over anchor-only after the anchor-only variant collapsed OPUS-100 (median 6-word anchors) too aggressively.
6. Mix-script augmentation: sample eligible Uzbek anchors stratified by source, transliterate to the opposite script via `utils.translit.to_cyrillic` / `to_latin`, emit (original, transliterated) pairs with `source="mixscript"`. Target: {mix_ratio_pct:.1f}% of final pool.
7. Validation split assembled from FLORES-200 `uzn_Latn-eng_Latn` dev plus the Cyrillic counterpart (see below).
8. Wiki retrieval holdout: 5,000 random articles (seed=42) split off before training and exposed as the `wiki_retrieval_eval` config.

## Validation Split

The `validation` split of the `default` config is FLORES-200 dev — held out from training.

{cyrl_note}

`flores_devtest` is **never** included in any config — it is reserved for the
downstream model evaluation script (`eval.py`).

## Intended Use

- Training Uzbek sentence-embedding / retrieval models with contrastive losses (e.g., `MultipleNegativesRankingLoss`).
- Compatible with `sentence-transformers` `SentenceTransformerTrainer`.
- Anchor↔positive is treated symmetrically — no `query:` / `passage:` prefix required for the IBM Granite r2 base.

## Limitations

- **Cyrillic Е/Э one-way mapping (rule-based fallback).** The rule-based transliterator maps Latin `E` → Cyrillic `Е` only, never `Э`. Native Uzbek words overwhelmingly use `Е`; rare loanwords using `Э` lose the distinction on round-trip. `UzTransliterator==0.0.36` is the primary path and handles this correctly when installed.
- **OPUS-100 en-uz license is unknown.** Included under common research-redistribution conventions. Downstream users in restrictive jurisdictions should verify the source license themselves.
- **Wiki title overlaps in the retrieval holdout.** Two articles in the 5,000-row holdout share a title with a training article (Wikipedia redirects/disambig pages). Minor leakage; flagged for transparency.
- **Wiki pairing is heuristic.** Title↔first-80-words after dropping disambig pages and title-in-first-sentence rows. Some pairs are weakly grounded.
- **`auto_detect_script` is heuristic.** Latin text containing `q`/`x` is tagged `uz_Latn` (rare in English roots); pure English with these letters may be misclassified.

## Citations

Please cite the source datasets you transitively use:

```bibtex
@inproceedings{{zhang-etal-2020-improving,
  title     = {{Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation}},
  author    = {{Zhang, Biao and Williams, Philip and Titov, Ivan and Sennrich, Rico}},
  booktitle = {{Proceedings of ACL}},
  year      = {{2020}}
}}

@article{{nllb2022,
  title   = {{No Language Left Behind: Scaling Human-Centered Machine Translation}},
  author  = {{NLLB Team}},
  journal = {{arXiv preprint arXiv:2207.04672}},
  year    = {{2022}}
}}

@misc{{uzwiki2024,
  title  = {{Uzbek Wikipedia dump}},
  author = {{Yakhyo, A.}},
  year   = {{2024}},
  url    = {{https://huggingface.co/datasets/yakhyo/uz-wiki}}
}}

@misc{{granite_embedding_r2,
  title  = {{Granite Embedding 97m Multilingual r2}},
  author = {{IBM Granite Team}},
  year   = {{2026}},
  url    = {{https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2}}
}}
```

## License

Apache-2.0 for the **processing pipeline and curation choices** in this dataset. Source-row licenses are preserved per the row's `source` tag — consult the table above for upstream terms.

## Reproduction

```bash
git clone https://github.com/sukhrobnurali/uzbek-embedding-pairs
cd uzbek-embedding-pairs
pip install -r requirements.txt
python prepare_data.py --push --repo-id <your-fork>
```

## See Also

- Model trained on this dataset (forthcoming): `sukhrobnurali/uzbek-granite-97m-v1`
- Live demo Space (forthcoming): `sukhrobnurali/uzbek-embedding-demo`
- Base embedding model: `ibm-granite/granite-embedding-97m-multilingual-r2`
"""
```

- [ ] **Step 4.4: Run tests to verify they pass**

Run: `c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m pytest tests/test_dataset_card.py -v`

Expected: 8 passed.

- [ ] **Step 4.5: Run full suite + ruff**

Run: `c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m pytest tests/ -q && c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m ruff check .`

Expected: 128 passed, ruff clean.

- [ ] **Step 4.6: Commit**

```bash
cd c:/projects/portfolio/HF/1-embedding
git add utils/dataset_card.py tests/test_dataset_card.py
git commit -m "feat(dataset_card): card builder with dynamic stats and honest cyrl-source reporting"
```

---

## Task 5: Push orchestrator + CLI flags

**Files:**
- Modify: `prepare_data.py` (add `push_to_hf`, `_print_preflight`, `_perform_pushes`, extend `main`)
- Test: `tests/test_prepare_data.py` (add `TestPushPreflight` class)

- [ ] **Step 5.1: Write the failing tests**

Add to the top of `tests/test_prepare_data.py` (if not already there):

```python
from unittest.mock import MagicMock
```

Add at the end of `tests/test_prepare_data.py`:

```python
class TestPushPreflight:
    @staticmethod
    def _tiny_dataset() -> Dataset:
        return Dataset.from_list([
            {"anchor": "a", "positive": "b", "source": "parallel_opus",
             "anchor_lang": "uz_Latn", "positive_lang": "en"}
        ])

    @staticmethod
    def _stats() -> dict:
        return {
            "repo_id": "user/test", "train_rows": 1, "validation_rows": 1,
            "retrieval_rows": 1, "smoke_rows": 1,
            "source_distribution": {"parallel_opus": 1},
            "mixscript_ratio": 0.10, "validation_cyrl_source": "native",
            "near_dedup_threshold": 0.9, "generation_date": "2026-05-17",
        }

    def test_aborts_on_no_response(self, monkeypatch):
        ds = self._tiny_dataset()
        fake_api = MagicMock()
        fake_api.repo_exists.return_value = False
        fake_api.whoami.return_value = {"name": "tester"}
        perform = MagicMock()
        monkeypatch.setenv("HF_TOKEN", "fake-token")
        monkeypatch.setattr(prepare_data, "HfApi", lambda: fake_api)
        monkeypatch.setattr(prepare_data, "_perform_pushes", perform)
        monkeypatch.setattr("builtins.input", lambda _prompt: "no")

        with pytest.raises(SystemExit):
            prepare_data.push_to_hf(
                train=ds, validation=ds, retrieval=ds, smoke=ds,
                repo_id="user/test", stats=self._stats(),
                confirm=False, allow_overwrite=False,
            )
        perform.assert_not_called()
        # repo didn't exist, so create_repo was called
        fake_api.create_repo.assert_called_once()

    def test_aborts_on_existing_repo_without_overwrite(self, monkeypatch):
        ds = self._tiny_dataset()
        fake_api = MagicMock()
        fake_api.repo_exists.return_value = True
        perform = MagicMock()
        monkeypatch.setenv("HF_TOKEN", "fake-token")
        monkeypatch.setattr(prepare_data, "HfApi", lambda: fake_api)
        monkeypatch.setattr(prepare_data, "_perform_pushes", perform)

        with pytest.raises(SystemExit, match="exists"):
            prepare_data.push_to_hf(
                train=ds, validation=ds, retrieval=ds, smoke=ds,
                repo_id="user/test", stats=self._stats(),
                confirm=True, allow_overwrite=False,
            )
        perform.assert_not_called()
        fake_api.create_repo.assert_not_called()

    def test_proceeds_with_confirm_flag(self, monkeypatch):
        ds = self._tiny_dataset()
        fake_api = MagicMock()
        fake_api.repo_exists.return_value = False
        fake_api.whoami.return_value = {"name": "tester"}
        perform = MagicMock()
        push_card = MagicMock()
        monkeypatch.setenv("HF_TOKEN", "fake-token")
        monkeypatch.setattr(prepare_data, "HfApi", lambda: fake_api)
        monkeypatch.setattr(prepare_data, "_perform_pushes", perform)
        monkeypatch.setattr(prepare_data, "_push_card", push_card)

        prepare_data.push_to_hf(
            train=ds, validation=ds, retrieval=ds, smoke=ds,
            repo_id="user/test", stats=self._stats(),
            confirm=True, allow_overwrite=False,
        )
        perform.assert_called_once()
        push_card.assert_called_once()

    def test_aborts_without_hf_token(self, monkeypatch):
        ds = self._tiny_dataset()
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with pytest.raises(SystemExit, match="HF_TOKEN"):
            prepare_data.push_to_hf(
                train=ds, validation=ds, retrieval=ds, smoke=ds,
                repo_id="user/test", stats=self._stats(),
                confirm=True, allow_overwrite=False,
            )

    def test_skips_create_repo_when_existing_and_overwrite(self, monkeypatch):
        ds = self._tiny_dataset()
        fake_api = MagicMock()
        fake_api.repo_exists.return_value = True
        fake_api.whoami.return_value = {"name": "tester"}
        perform = MagicMock()
        push_card = MagicMock()
        monkeypatch.setenv("HF_TOKEN", "fake-token")
        monkeypatch.setattr(prepare_data, "HfApi", lambda: fake_api)
        monkeypatch.setattr(prepare_data, "_perform_pushes", perform)
        monkeypatch.setattr(prepare_data, "_push_card", push_card)

        prepare_data.push_to_hf(
            train=ds, validation=ds, retrieval=ds, smoke=ds,
            repo_id="user/test", stats=self._stats(),
            confirm=True, allow_overwrite=True,
        )
        fake_api.create_repo.assert_not_called()
        perform.assert_called_once()


class TestCLISmokePushGuard:
    def test_smoke_push_requires_repo_override(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["prepare_data.py", "--smoke", "--push"])
        with pytest.raises(SystemExit):
            prepare_data.main()
```

- [ ] **Step 5.2: Run tests to verify they fail**

Run: `c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m pytest tests/test_prepare_data.py::TestPushPreflight tests/test_prepare_data.py::TestCLISmokePushGuard -v`

Expected: All tests fail with `AttributeError` for `push_to_hf` / `HfApi` / etc.

- [ ] **Step 5.3: Implement `push_to_hf` and helpers**

Add to the top of `prepare_data.py`, after existing imports:

```python
import os
import sys

from huggingface_hub import HfApi
from huggingface_hub.repocard import DatasetCard

from utils.dataset_card import build_card
```

Add at module scope (after the existing constants):

```python
CANONICAL_REPO_ID = "sukhrobnurali/uzbek-embedding-pairs"
```

Add these functions after `build_smoke_100`:

```python
def _print_preflight(api: HfApi, repo_id: str, stats: dict, card: str) -> None:
    user = api.whoami().get("name", "<unknown>")
    log.info("=== HF PUSH PRE-FLIGHT ===")
    log.info("Repo:          https://huggingface.co/datasets/%s", repo_id)
    log.info("Token user:    %s", user)
    log.info("Configs:")
    log.info("  default              train=%s  validation=%s", f"{stats['train_rows']:,}", f"{stats['validation_rows']:,}")
    log.info("  wiki_retrieval_eval  test=%s", f"{stats['retrieval_rows']:,}")
    log.info("  smoke_100            train=%s", f"{stats['smoke_rows']:,}")
    log.info("Source distribution: %s", stats["source_distribution"])
    log.info("Mixscript ratio:     %.1f%%", stats["mixscript_ratio"] * 100)
    log.info("Validation cyrl:     %s", stats["validation_cyrl_source"])
    log.info("--- card preview (first 30 lines) ---")
    for line in card.splitlines()[:30]:
        log.info(line)
    log.info("--- card preview (last 10 lines) ---")
    for line in card.splitlines()[-10:]:
        log.info(line)
    log.info("=== END PRE-FLIGHT ===")


def _perform_pushes(
    train: Dataset,
    validation: Dataset,
    retrieval: Dataset,
    smoke: Dataset,
    repo_id: str,
) -> None:
    from datasets import DatasetDict

    log.info("Pushing config 'default' ...")
    DatasetDict({"train": train, "validation": validation}).push_to_hub(
        repo_id, config_name="default"
    )
    log.info("Pushing config 'wiki_retrieval_eval' ...")
    DatasetDict({"test": retrieval}).push_to_hub(
        repo_id, config_name="wiki_retrieval_eval"
    )
    log.info("Pushing config 'smoke_100' ...")
    DatasetDict({"train": smoke}).push_to_hub(
        repo_id, config_name="smoke_100"
    )


def _push_card(api: HfApi, repo_id: str, card: str) -> None:
    log.info("Pushing dataset card ...")
    DatasetCard(card).push_to_hub(repo_id, repo_type="dataset")


def push_to_hf(
    train: Dataset,
    validation: Dataset,
    retrieval: Dataset,
    smoke: Dataset,
    repo_id: str,
    stats: dict,
    confirm: bool = False,
    allow_overwrite: bool = False,
) -> None:
    """Push the dataset to HF after pre-flight checks.

    Calls SystemExit on any abort path so the caller can be safely scripted.
    """
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set in the environment; refusing to push.")

    api = HfApi()
    exists = api.repo_exists(repo_id, repo_type="dataset")
    if exists and not allow_overwrite:
        raise SystemExit(
            f"Dataset repo {repo_id} already exists; pass --allow-overwrite to push anyway."
        )
    if not exists:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=False, private=False)
        log.info("Created dataset repo %s", repo_id)

    card = build_card(stats)
    _print_preflight(api, repo_id, stats, card)

    if not confirm:
        response = input("Type 'yes' to push: ").strip().lower()
        if response != "yes":
            raise SystemExit("Push aborted by user.")

    _perform_pushes(train, validation, retrieval, smoke, repo_id)
    _push_card(api, repo_id, card)
    log.info("Done. https://huggingface.co/datasets/%s", repo_id)
```

Modify `main()` in `prepare_data.py` to wire up the new flags. Replace the existing `main()` with:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="Load 50 rows per source")
    parser.add_argument("--max-rows", type=int, default=None, help="Cap rows per source")
    parser.add_argument("--output-dir", default="data/processed", help="Where to write parquet")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(SOURCES_DEFAULT),
        help="Source names to load (see utils.dataset_io._LOADERS)",
    )
    parser.add_argument("--near-dedup-threshold", type=float, default=0.9)
    parser.add_argument("--near-dedup-num-perm", type=int, default=64)
    parser.add_argument("--wiki-holdout-size", type=int, default=5000)
    parser.add_argument("--wiki-seed", type=int, default=42)
    parser.add_argument("--mixscript-ratio", type=float, default=0.10)
    parser.add_argument("--mixscript-seed", type=int, default=42)
    parser.add_argument("--push", action="store_true", help="Push to Hugging Face after building")
    parser.add_argument("--confirm", action="store_true", help="Skip interactive 'yes' prompt")
    parser.add_argument("--allow-overwrite", action="store_true", help="Allow pushing to an existing repo")
    parser.add_argument("--repo-id", default=CANONICAL_REPO_ID, help="Override target repo (testing/forks)")
    args = parser.parse_args()

    if args.smoke and args.push and args.repo_id == CANONICAL_REPO_ID:
        raise SystemExit(
            "Refusing to push a --smoke build to the canonical repo. "
            "Pass --repo-id <your-test-repo> if you really want to push the smoke build."
        )

    combined, wiki_holdout = build_pool(
        sources=tuple(args.sources),
        smoke=args.smoke,
        max_rows=args.max_rows,
        near_dedup_threshold=args.near_dedup_threshold,
        near_dedup_num_perm=args.near_dedup_num_perm,
        wiki_holdout_size=args.wiki_holdout_size,
        wiki_seed=args.wiki_seed,
    )

    log.info("Building mix-script augmentation ...")
    mix_aug = augment_mixscript(combined, target_ratio=args.mixscript_ratio, seed=args.mixscript_seed)
    log.info("Mixscript rows: %d (%.1f%% of final)", len(mix_aug), len(mix_aug) / (len(combined) + len(mix_aug)) * 100)
    train_pool = concatenate_datasets([combined, mix_aug]) if len(mix_aug) > 0 else combined

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_pool.to_parquet(str(output_dir / "default.parquet"))
    log.info("Wrote %d rows to %s", len(train_pool), output_dir / "default.parquet")

    if wiki_holdout is not None:
        wiki_holdout.to_parquet(str(output_dir / "wiki_retrieval_eval.parquet"))

    if not args.push:
        return

    log.info("Building validation split ...")
    validation, cyrl_source = build_validation()
    log.info("Validation rows: %d (cyrl_source=%s)", len(validation), cyrl_source)

    log.info("Building smoke_100 ...")
    smoke = build_smoke_100(train_pool)

    stats = {
        "repo_id": args.repo_id,
        "train_rows": len(train_pool),
        "validation_rows": len(validation),
        "retrieval_rows": len(wiki_holdout) if wiki_holdout is not None else 0,
        "smoke_rows": len(smoke),
        "source_distribution": _source_distribution(train_pool),
        "mixscript_ratio": args.mixscript_ratio,
        "validation_cyrl_source": cyrl_source,
        "near_dedup_threshold": args.near_dedup_threshold,
        "generation_date": __import__("datetime").date.today().isoformat(),
    }

    if wiki_holdout is None:
        raise SystemExit("Wiki holdout was not built; cannot push wiki_retrieval_eval config.")

    push_to_hf(
        train=train_pool,
        validation=validation,
        retrieval=wiki_holdout,
        smoke=smoke,
        repo_id=args.repo_id,
        stats=stats,
        confirm=args.confirm,
        allow_overwrite=args.allow_overwrite,
    )
```

- [ ] **Step 5.4: Run tests to verify they pass**

Run: `c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m pytest tests/test_prepare_data.py::TestPushPreflight tests/test_prepare_data.py::TestCLISmokePushGuard -v`

Expected: 6 passed (5 preflight + 1 CLI guard).

- [ ] **Step 5.5: Run full suite + ruff**

Run: `c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m pytest tests/ -q && c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m ruff check .`

Expected: 134 passed, ruff clean.

- [ ] **Step 5.6: Verify `prepare_data.py` stays under 300 lines**

Run: `c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -c "print(sum(1 for _ in open('prepare_data.py')))"`

Expected: <300. If over, extract the push orchestrator into `utils/hf_push.py` and re-run tests.

- [ ] **Step 5.7: Commit**

```bash
cd c:/projects/portfolio/HF/1-embedding
git add prepare_data.py tests/test_prepare_data.py
git commit -m "feat(prepare_data): push orchestrator with two-flag gate and pre-flight"
```

---

## Task 6: Test-repo dry run

**This task does not modify code. It de-risks the irreversible canonical push by pushing to a throwaway repo first and inspecting the result on HF.**

- [ ] **Step 6.1: Verify HF_TOKEN is set**

Run: `c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -c "import os; print('HF_TOKEN set:', bool(os.environ.get('HF_TOKEN')))"`

Expected: `HF_TOKEN set: True`. If False, set it before continuing.

- [ ] **Step 6.2: Install push-time dependencies if missing**

The local `.venv/` was kept light for Tasks 1-6. Push needs `huggingface_hub`. Run:

```
c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m pip install -r requirements.txt
```

If installs fail on `torch`, `sentence-transformers`, or `faiss-cpu` (Colab-deferred), strip them from the install command for this push:

```
c:/projects/portfolio/HF/1-embedding/.venv/Scripts/python.exe -m pip install "huggingface_hub>=0.26.0,<2.0.0" "datasets>=3.0.0,<5.0.0" "datasketch>=1.6.0" "pandas>=2.2.0" "pyarrow>=17.0.0"
```

- [ ] **Step 6.3: Push to the test repo**

Run from `c:/projects/portfolio/HF/1-embedding`:

```
.venv/Scripts/python.exe prepare_data.py --push --repo-id sukhrobnurali/uzbek-embedding-pairs-test
```

Expected:
- Builds train pool (~356K rows after mixscript).
- Builds validation (~1,994 rows, `cyrl_source` reported).
- Builds smoke_100 (100 rows).
- Creates the test repo.
- Prints pre-flight summary.
- Prompts: `Type 'yes' to push:`. Type `yes` and press Enter.
- Pushes three configs + card.
- Prints `https://huggingface.co/datasets/sukhrobnurali/uzbek-embedding-pairs-test`.

- [ ] **Step 6.4: Inspect the test repo on HF**

Open `https://huggingface.co/datasets/sukhrobnurali/uzbek-embedding-pairs-test` in a browser. Verify:

- Card renders with all 13 sections (Dataset Structure, Schema, Source Datasets, Processing Pipeline, Validation Split, Intended Use, Limitations, Citations, License, Reproduction).
- YAML frontmatter parses (HF should show the configs widget).
- Three configs are visible: `default`, `wiki_retrieval_eval`, `smoke_100`.
- Row counts in the card match what HF reports.
- Validation Split section names `native` or `transliterated` correctly.

Then verify programmatically:

```
.venv/Scripts/python.exe -c "from datasets import load_dataset; ds = load_dataset('sukhrobnurali/uzbek-embedding-pairs-test'); print(ds)"
.venv/Scripts/python.exe -c "from datasets import load_dataset; ds = load_dataset('sukhrobnurali/uzbek-embedding-pairs-test', 'smoke_100'); print(len(ds['train']))"
.venv/Scripts/python.exe -c "from datasets import load_dataset; ds = load_dataset('sukhrobnurali/uzbek-embedding-pairs-test', 'wiki_retrieval_eval'); print(len(ds['test']))"
```

Expected: All three load cleanly; row counts match expectations (smoke_100=100, wiki_retrieval_eval=5000, default train ≈356K + validation ≈1994).

- [ ] **Step 6.5: Delete the test repo**

```
.venv/Scripts/python.exe -c "from huggingface_hub import HfApi; HfApi().delete_repo('sukhrobnurali/uzbek-embedding-pairs-test', repo_type='dataset')"
```

Expected: no error. Verify in the browser the repo is gone.

---

## Task 7: Real push + verification + memory update

- [ ] **Step 7.1: Confirm canonical repo does not already exist**

```
.venv/Scripts/python.exe -c "from huggingface_hub import HfApi; print(HfApi().repo_exists('sukhrobnurali/uzbek-embedding-pairs', repo_type='dataset'))"
```

Expected: `False`. If `True`, stop and check whether a prior session pushed there — do NOT use `--allow-overwrite` without confirming the existing contents are yours.

- [ ] **Step 7.2: Push to the canonical repo**

```
.venv/Scripts/python.exe prepare_data.py --push
```

Expected:
- Same build as Step 6.3.
- Pre-flight summary shows `repo_id=sukhrobnurali/uzbek-embedding-pairs`.
- Prompt `Type 'yes' to push:` — type `yes`.
- Pushes three configs + card.
- Final line prints the canonical URL.

- [ ] **Step 7.3: Verify the canonical repo**

Open `https://huggingface.co/datasets/sukhrobnurali/uzbek-embedding-pairs` and run:

```
.venv/Scripts/python.exe -c "from datasets import load_dataset; print(load_dataset('sukhrobnurali/uzbek-embedding-pairs'))"
```

Expected: `default` config loads with `train` (~356K) and `validation` (~1,994) splits.

- [ ] **Step 7.4: Update memory file**

Edit `C:\Users\User\.claude\projects\c--projects-portfolio-HF-1-embedding\memory\project_1_embedding.md` — under the "Progress" section, change:

- `⏸ Tasks 7–15 pending` → `✅ Task 7: mix-script aug + HF dataset push.` + `⏸ Tasks 8–15 pending`.
- Add a bullet under "Local artifacts" with the live dataset URL: `https://huggingface.co/datasets/sukhrobnurali/uzbek-embedding-pairs`.
- Update "Next session — resume from here" to point to Task 8 (`train.py` on Colab).

- [ ] **Step 7.5: Commit**

```bash
cd c:/projects/portfolio/HF/1-embedding
git add prepare_data.py utils/dataset_card.py tests/
git status                                              # confirm nothing surprising
git commit -m "feat(task-7): publish sukhrobnurali/uzbek-embedding-pairs (mixscript + 3 configs)"
```

(Most code was committed in Tasks 1-5; this commit is for any leftover formatting or doc updates surfaced during the actual push.)

- [ ] **Step 7.6: Final verification**

```
.venv/Scripts/python.exe -m pytest tests/ -q
.venv/Scripts/python.exe -m ruff check .
git log --oneline -10
```

Expected: tests pass, ruff clean, recent commits show Tasks 1-5 increments + Task 7 finalization.

---

## Risks & Mitigations

1. **Push partially fails mid-run.** The repo exists with partial configs. Re-run with `--allow-overwrite`. The abort message reminds the user. Step 7.1 catches the prior-attempt state.
2. **`HF_TOKEN` is read-only or wrong account.** Pre-flight prints the `whoami` user — verify it's `sukhrobnurali` before typing `yes`.
3. **FLORES uzn_Cyrl actually exists.** The card honestly reports `native` and an extra ~997 rows land in the validation split. Tests cover both branches.
4. **`huggingface_hub` API drift between minors.** Pinned `>=0.26.0,<2.0.0` in `requirements.txt`. The `DatasetCard(content).push_to_hub` and `DatasetDict.push_to_hub` paths used here have been stable since 0.25.
5. **prepare_data.py exceeds 300 lines after Task 5.** Step 5.6 checks and instructs to extract `utils/hf_push.py` if so.
