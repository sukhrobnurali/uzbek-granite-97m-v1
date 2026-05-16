# Task 7 — Mix-script Augmentation + HF Dataset Push

**Date:** 2026-05-17
**Project:** Uzbek sentence embedding (Project 1 of HF portfolio)
**Plan reference:** `C:\Users\User\.claude\plans\project-context-applies-shimmying-pony.md` — Task 7
**Predecessor commits:** Tasks 1–6 complete on `main` (6 commits, 109 tests, ruff clean).
**Working dir:** `c:\projects\portfolio\HF\1-embedding`

## Goal

Ship the `sukhrobnurali/uzbek-embedding-pairs` HF dataset, with three configs (`default`, `wiki_retrieval_eval`, `smoke_100`), a populated dataset card, and a validation split sourced from FLORES-200 dev. Add 10% Cyrillic↔Latin mix-script augmentation to the training pool so the downstream `cosine(uz_Latn, uz_Cyrl) > 0.9` invariant is optimized directly.

The HF push is irreversible. The CLI is structured so that the canonical repo cannot be touched without two explicit flags and an interactive `yes`.

## Locked Decisions

1. **Mix-script ratio: 10% of the final pool.** Add `n_mix` rows such that `n_mix / (len(pool) + n_mix) ≈ 0.10`. Concretely: 320,650 + ~35,628 → 356,278 rows, mixscript share = 10.0%.
2. **Push safety: two-flag gate.** Default behavior unchanged (local parquet only). `--push` triggers an interactive pre-flight that requires `yes` on stdin before any HF call. `--confirm` skips the prompt for scripted use. `create_repo(exist_ok=False)` is the default; `--allow-overwrite` is required to touch an existing repo.
3. **Validation Cyrl strategy: try Cyrl, transliterate as fallback.** If `Muennighoff/flores200:uzn_Cyrl-eng_Latn` loads, use it (997 rows). Otherwise transliterate the Latn anchors via `utils.translit.to_cyrillic` and tag `source = "flores_dev_cyrl_translit"`. Either way validation is ~1,994 rows. The dataset card reports the actual cyrl_source honestly.

## Architecture

### 1. Mix-script augmentation

New function in `prepare_data.py`:

```python
def augment_mixscript(
    pool: Dataset,
    target_ratio: float = 0.10,
    seed: int = 42,
    oversample_factor: float = 1.05,
) -> Dataset:
    """Sample ~target_ratio of pool, transliterate the Uzbek anchor to the opposite
    script, emit (original_uz, transliterated_uz) pairs with source='mixscript'.
    Both directions emerge naturally from the input script distribution.
    """
```

- **Eligibility:** rows whose `anchor_lang ∈ {"uz_Latn", "uz_Cyrl"}`.
- **Target count:** `n_target = round(len(eligible) * target_ratio / (1 - target_ratio))` (solves `n/(N+n)=target`).
- **Sampling:** stratified by `source`; seeded `random.Random(seed)`.
- **Per-row transform:**
  - If `anchor_lang == "uz_Latn"`: `positive = to_cyrillic(anchor)`.
  - If `anchor_lang == "uz_Cyrl"`: `positive = to_latin(anchor)`.
  - Emit `{anchor, positive, source: "mixscript", anchor_lang, positive_lang}` with both lang tags recomputed from the actual produced strings.
- **Sanity drops:** `auto_detect_script(positive) == auto_detect_script(anchor)` → drop. `positive == anchor` → drop. `positive` empty → drop. Over-sample by `oversample_factor` (default 1.05) so drops don't bring the final ratio below target.
- **Composition:** returned dataset is the augmentation only (caller concatenates with the original pool).

### 2. Validation split

New function in `prepare_data.py`:

```python
def build_validation() -> tuple[Dataset, str]:
    """Build the FLORES dev validation split. Returns (dataset, cyrl_source) where
    cyrl_source ∈ {'native', 'transliterated'} for honest reporting in the card.
    """
```

- Load `flores_dev_latn` (expected, 997 rows).
- Try `flores_dev_cyrl`; if `None`, transliterate `latn` anchors to Cyrl, re-tag `source = "flores_dev_cyrl_translit"`, mark `cyrl_source = "transliterated"`.
- Concat and filter via the same `filter_dataset` used for the train pool.

### 3. smoke_100 config

```python
def build_smoke_100(train_pool: Dataset, seed: int = 42) -> Dataset:
    """100 rows stratified by source, proportional to train_pool's source distribution."""
```

- Single `train` split. Useful for `python smoke_test.py` (Task 9) and for users who want to inspect the dataset shape without downloading 350K rows.

### 4. Dataset card builder

New module `utils/dataset_card.py`:

```python
def build_card(stats: dict) -> str:
    """Render the dataset card markdown from computed stats.

    stats keys: repo_id, train_rows, validation_rows, retrieval_rows, smoke_rows,
                source_distribution (dict), mixscript_ratio (float),
                validation_cyrl_source ('native'|'transliterated'),
                near_dedup_threshold, generation_date.
    """
```

The card is a long templated f-string with YAML frontmatter and body sections enumerated below. All numbers come from the `stats` dict — none hardcoded.

#### Card sections

1. **Frontmatter (YAML):** `license: apache-2.0`, `language: [uz, en]`, `multilinguality: multilingual`, `size_categories: [100K<n<1M]`, `task_categories: [sentence-similarity, feature-extraction]`, `tags: [uzbek, embedding, sentence-embedding, parallel-corpus, mixed-script]`, plus a `configs:` block listing the three configs and their data files.
2. **Title + one-line description.**
3. **Dataset Structure** — table of configs × splits × row counts, populated from `stats`.
4. **Schema** — the 5 columns documented.
5. **Source datasets table** — name, HF ID, license, role.
6. **Processing Pipeline** — NFC + apostrophe normalization, filters, exact dedup, MinHash joint-key near-dedup @ 0.9, mix-script augmentation @ 10%, validation source.
7. **Validation note** — populated from `stats["validation_cyrl_source"]`.
8. **Intended use** — training Uzbek sentence-embedding / retrieval models; compatible with `MultipleNegativesRankingLoss`.
9. **Limitations** — Cyrillic Е/Э one-way; OPUS-100 en-uz license unknown; wiki title overlaps (2 in holdout); wiki title↔paragraph pairing is heuristic; `auto_detect_script` heuristics.
10. **Citations** — BibTeX for OPUS-100, FLORES-200, Uzbek Wikipedia, IBM Granite base model.
11. **License** — Apache-2.0 for the processing pipeline; source licenses preserved per row's `source` tag.
12. **Reproduction** — `git clone <repo>; python prepare_data.py --push --repo-id <fork>`.
13. **Cross-links** — to `sukhrobnurali/uzbek-granite-97m-v1` and `sukhrobnurali/uzbek-embedding-demo` (with "forthcoming" note since Tasks 10/13 aren't shipped).

### 5. Push orchestrator

New function in `prepare_data.py`:

```python
def push_to_hf(
    train_pool: Dataset,
    validation: Dataset,
    wiki_holdout: Dataset,
    smoke_100: Dataset,
    repo_id: str,
    stats: dict,
    confirm: bool,
    allow_overwrite: bool,
) -> None:
```

Flow:

1. `assert os.environ.get("HF_TOKEN")` — fail fast if token missing.
2. `api = HfApi()`. `whoami = api.whoami()` to surface the token's identity in pre-flight.
3. `exists = api.repo_exists(repo_id, repo_type="dataset")`.
   - `exists and not allow_overwrite` → abort with explicit error (suggest `--allow-overwrite`).
   - `not exists` → `api.create_repo(repo_id, repo_type="dataset", exist_ok=False, private=False)`.
   - `exists and allow_overwrite` → skip `create_repo` (repo is already there; subsequent `push_to_hub` calls will overwrite the relevant configs).
4. Print pre-flight: repo URL, token user, configs + splits + row counts, source distribution, mixscript ratio, validation cyrl_source, card preview (first 30 + last 10 lines).
5. If `not confirm`: read stdin; require exactly `"yes"` (trimmed, case-insensitive). Any other input aborts before any push.
6. Push configs in order: `default` (with `train` + `validation` splits) → `wiki_retrieval_eval` (with `test`) → `smoke_100` (with `train`).
7. Push the card via `DatasetCard(content).push_to_hub(repo_id, repo_type="dataset")`.
8. Print final URLs.

### 6. CLI surface

New flags on `prepare_data.py` (additive — existing flags unchanged):

```
--push                  Build + push to HF (default: local parquet only).
--confirm               Skip the interactive 'yes' prompt.
--allow-overwrite       Push to a repo that already exists (default refuses).
--repo-id REPO_ID       Override the canonical repo (testing/forks).
```

**Hard guard:** `--smoke --push` errors out unless `--repo-id` is also given. A 50-row smoke build must not target the canonical repo.

## File Changes

| Path | Action | Notes |
|---|---|---|
| `prepare_data.py` | extend | Add `augment_mixscript`, `build_validation`, `build_smoke_100`, `push_to_hf`, new CLI flags. Stays <300 lines. |
| `utils/dataset_card.py` | new | `build_card(stats: dict) -> str`. |
| `tests/test_prepare_data.py` | extend | Mixscript, validation, smoke_100, push-preflight tests. |
| `tests/test_dataset_card.py` | new | Card structure + dynamic-content tests. |

## Tests (TDD — all fast, no HF network hits)

| Test | Asserts |
|---|---|
| `test_mixscript_target_ratio` | On a 100-row fixture, augmented count makes mixscript ≈ 10% ±1pp. |
| `test_mixscript_deterministic` | Same seed → same emitted rows. |
| `test_mixscript_flips_script` | Every row: `anchor_lang != positive_lang`, both in `{uz_Latn, uz_Cyrl}`. |
| `test_mixscript_drops_no_op` | Row whose transliteration is a no-op (all-digit anchor) is not emitted. |
| `test_mixscript_stratified` | Source proportions in augmentation match input pool ±20%. |
| `test_build_validation_native_cyrl` | When `load_source("flores_dev_cyrl")` returns rows: validation has Latn + native Cyrl, `cyrl_source == "native"`, no `*_translit` source tag. |
| `test_build_validation_translit_fallback` | When `load_source("flores_dev_cyrl")` returns `None`: validation has Latn + transliterated rows tagged `flores_dev_cyrl_translit`, `cyrl_source == "transliterated"`. |
| `test_smoke_100_size_and_strata` | Exactly 100 rows; source distribution within 3 rows of proportional target. |
| `test_card_contains_required_sections` | Card has YAML frontmatter, all expected H2 headers, BibTeX blocks. |
| `test_card_reports_cyrl_source` | Card text reflects `stats["validation_cyrl_source"]` (native vs transliterated). |
| `test_card_row_counts_consistent` | Numbers in the card match `stats` dict — no hardcoded counts. |
| `test_preflight_aborts_on_no` | Pre-flight reads `"no"` from stdin → no `HfApi` push methods called (mocked). |
| `test_preflight_aborts_on_existing_repo` | `repo_exists` returns True and `allow_overwrite=False` → push functions not called. |
| `test_smoke_push_requires_repo_override` | `--smoke --push` without `--repo-id` → SystemExit before any HF call. |

## Implementation Order

Strict TDD per `feedback_iteration_rules.md`:

1. Write all tests above (red).
2. Implement `augment_mixscript`, `build_validation`, `build_smoke_100` → green.
3. Implement `utils/dataset_card.py::build_card` → green.
4. Implement `push_to_hf` orchestrator + new CLI flags → green.
5. Local dry run: `python prepare_data.py --push --repo-id sukhrobnurali/uzbek-embedding-pairs-test` (TEST repo).
6. Inspect the test repo on HF: verify all three configs load via `load_dataset(...)`, card renders, validation Cyrl tag matches reality.
7. Delete the test repo via `huggingface_hub.HfApi().delete_repo(...)`.
8. Real push: `python prepare_data.py --push` (no `--repo-id`), type `yes` at the prompt.
9. Verify canonical repo; commit.

Step 5 is the safety net for the irreversible step 8. HF has no `--dry-run` mode for pushes, so a test-repo cycle is the closest equivalent.

## Risks & Mitigations

1. **FLORES-200 `uzn_Cyrl-eng_Latn` not on HF.** Mitigated by the transliteration fallback path; honestly tagged in the card.
2. **Mix-script sampling over-counts mixed-script anchors.** `auto_detect_script` returns `"mixed"` for rows containing both alphabets — those are excluded from eligibility, so no double-counting.
3. **Push partially fails mid-run.** Repo exists with partial configs. No auto-cleanup; user must re-run with `--allow-overwrite`. Documented in the abort message.
4. **HF rate-limits during large push.** `push_to_hub` has retry/resume built in; if it fails the user re-runs with `--allow-overwrite`.
5. **Test-repo step costs nothing but disk on HF** — free tier handles 350K-row dataset uploads fine.
6. **Determinism drift if `to_cyrillic` / `to_latin` change.** Mitigated by pinning `UzTransliterator==0.0.36` in `requirements.txt` and unit-testing the rule fallback. Mixscript seed is also stored in CLI args for full reproducibility.

## Done-criterion

- `load_dataset("sukhrobnurali/uzbek-embedding-pairs")` returns the `default` config with `train` and `validation` splits.
- `load_dataset("sukhrobnurali/uzbek-embedding-pairs", "wiki_retrieval_eval")["test"]` has 5,000 rows.
- `load_dataset("sukhrobnurali/uzbek-embedding-pairs", "smoke_100")["train"]` has 100 rows.
- Source distribution on `default/train` shows ~10% `mixscript`.
- Card renders on HF with all 13 sections; YAML frontmatter parses.
- All new tests pass; existing 109 tests still pass; `ruff check .` clean.
- Repo URL recorded in `project_1_embedding.md` memory under "Progress".
