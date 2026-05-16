"""Dataset card builder for sukhrobnurali/uzbek-embedding-pairs."""

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
Combines OPUS-100, Uzbek Wikipedia, FLORES-200 dev, and 10% Cyrillic-Latin mix-script
augmentation. Generated {gen_date}.

Repository: `{repo}`

## Dataset Structure

| Config | Split | Rows | Use |
|---|---|---|---|
| `default` | `train` | {_fmt(train)} | Training pool (parallel + monolingual + mixscript aug) |
| `default` | `validation` | {_fmt(val)} | In-loop eval during training (FLORES-200 dev) |
| `wiki_retrieval_eval` | `test` | {_fmt(retr)} | Held-out Wikipedia title-paragraph pairs for retrieval eval |
| `smoke_100` | `train` | {_fmt(smoke)} | Stratified 100-row sample for pipeline integrity tests |

## Schema

Every row has the same five columns:

| Column | Type | Description |
|---|---|---|
| `anchor` | string | First text in the pair (typically Uzbek for parallel sources, title for wiki) |
| `positive` | string | Paired text (typically English translation, or wiki paragraph, or transliterated Uzbek for mixscript) |
| `source` | string | Origin tag - see "Source Datasets" below |
| `anchor_lang` | string | `uz_Latn` \\| `uz_Cyrl` \\| `en` \\| `mixed` \\| `empty` (auto-detected) |
| `positive_lang` | string | Same vocabulary as `anchor_lang` |

## Source Datasets

| Source | HF ID | License | Role |
|---|---|---|---|
| OPUS-100 en-uz | `Helsinki-NLP/opus-100` | unknown (flagged below) | Parallel sentences (subset, after dedup) |
| Parallel-sentences-OPUS | `sentence-transformers/parallel-sentences-opus-100` | unknown | Cleaner OPUS variant (preferred on dedup collisions) |
| Uzbek Wikipedia | `yakhyo/uz-wiki` | MIT | Title-first-80-words pairs |
| FLORES+ dev | `openlanguagedata/flores_plus` | CC-BY-SA-4.0 (gated) | Validation split only (uzn_Latn-eng_Latn; Cyrillic side transliterated) |

### Source distribution in `default/train`

| Source | Rows | Share |
|---|---|---|
{dist_rows}

## Processing Pipeline

1. Load each source with `utils.dataset_io.load_source(name)` to a unified 5-column schema.
2. Normalize Unicode to NFC; canonicalize apostrophes (`o'`, `g'` markers).
3. Filter: length 5-512 chars per side (2000 for wiki paragraphs), word count >= 2, length ratio 0.4-2.5 for parallel sources, drop identity rows, Uzbek-side script-sanity check.
4. Exact dedup on `(anchor.lower(), positive.lower())`.
5. Near dedup via MinHash LSH (joint key `anchor || positive`, threshold {dedup_threshold:.2f}, num_perm 64). Joint key chosen over anchor-only after the anchor-only variant collapsed OPUS-100 (median 6-word anchors) too aggressively.
6. Mix-script augmentation: sample eligible Uzbek anchors stratified by source, transliterate to the opposite script via `utils.translit.to_cyrillic` / `to_latin`, emit (original, transliterated) pairs with `source="mixscript"`. Target: {mix_ratio_pct:.1f}% of final pool.
7. Validation split assembled from FLORES-200 `uzn_Latn-eng_Latn` dev plus the Cyrillic counterpart (see below).
8. Wiki retrieval holdout: 5,000 random articles (seed=42) split off before training and exposed as the `wiki_retrieval_eval` config.

## Validation Split

The `validation` split of the `default` config is FLORES-200 dev - held out from training.

{cyrl_note}

`flores_devtest` is **never** included in any config - it is reserved for the
downstream model evaluation script (`eval.py`).

## Intended Use

- Training Uzbek sentence-embedding / retrieval models with contrastive losses (e.g., `MultipleNegativesRankingLoss`).
- Compatible with `sentence-transformers` `SentenceTransformerTrainer`.
- Anchor-positive is treated symmetrically - no `query:` / `passage:` prefix required for the IBM Granite r2 base.

## Limitations

- **Cyrillic Е/Э one-way mapping (rule-based fallback).** The rule-based transliterator maps Latin `E` to Cyrillic `Е` only, never `Э`. Native Uzbek words overwhelmingly use `Е`; rare loanwords using `Э` lose the distinction on round-trip. `UzTransliterator==0.0.36` is the primary path and handles this correctly when installed.
- **OPUS-100 en-uz license is unknown.** Included under common research-redistribution conventions. Downstream users in restrictive jurisdictions should verify the source license themselves.
- **Wiki title overlaps in the retrieval holdout.** Two articles in the 5,000-row holdout share a title with a training article (Wikipedia redirects/disambig pages). Minor leakage; flagged for transparency.
- **Wiki pairing is heuristic.** Title and first-80-words after dropping disambig pages and title-in-first-sentence rows. Some pairs are weakly grounded.
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

Apache-2.0 for the **processing pipeline and curation choices** in this dataset. Source-row licenses are preserved per the row's `source` tag - consult the table above for upstream terms.

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
