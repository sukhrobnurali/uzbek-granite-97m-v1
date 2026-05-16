"""Build the uzbek-embedding-pairs dataset.

Multi-source pipeline:
  OPUS-100 + parallel-sentences-opus-100 + yakhyo/uz-wiki + Helsinki-NLP/tatoeba_mt
  -> unified schema -> filter -> exact + joint-key near dedup
  -> data/processed/default.parquet (train pool)
  -> data/processed/wiki_retrieval_eval.parquet (5000 wiki articles held out for retrieval eval)

FLORES-200 devtest is reserved for eval and is rejected here; FLORES dev becomes the
validation split when the dataset is pushed to HF (Task 7).

CLI:
    python prepare_data.py                          # default sources, full data
    python prepare_data.py --smoke                  # 50 rows per source, scaled wiki holdout
    python prepare_data.py --max-rows 5000          # cap rows per source
    python prepare_data.py --sources opus100        # subset of sources
    python prepare_data.py --wiki-holdout-size 3000
"""

from __future__ import annotations

import argparse
import datetime
import random as _random
from pathlib import Path

from datasets import Dataset, concatenate_datasets

from utils import dataset_io, dedup
from utils.hf_push import push_to_hf
from utils.logging_setup import configure
from utils.translit import auto_detect_script, to_cyrillic, to_latin

log = configure()

# parallel_opus first so exact_dedup keeps its cleaner formatting on OPUS-100
# collisions. tatoeba dropped from defaults because Helsinki-NLP/tatoeba_mt is a
# loading-script dataset, unsupported by datasets>=3.0. It can still be requested
# explicitly via --sources tatoeba if a future parquet mirror appears.
SOURCES_DEFAULT = ("parallel_opus", "opus100", "wiki")
FORBIDDEN_SOURCES = ("flores_devtest_latn", "flores_devtest_cyrl")
CANONICAL_REPO_ID = "sukhrobnurali/uzbek-embedding-pairs"


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

    Target satisfies n_aug / (len(pool) + n_aug) ~= target_ratio.
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
    for _src, idxs in by_source.items():
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
            positive_lang = "uz_Cyrl"
        else:
            positive = to_latin(anchor)
            positive_lang = "uz_Latn"
        if not positive or positive == anchor:
            continue
        rows.append({
            "anchor": anchor,
            "positive": positive,
            "source": "mixscript",
            "anchor_lang": anchor_lang,
            "positive_lang": positive_lang,
        })
        if len(rows) >= n_target:
            break

    return Dataset.from_list(rows)


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
        return concatenate_datasets([latn, cyrl]), "native"

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
    return concatenate_datasets([latn, translit]), "transliterated"


def build_smoke_100(pool: Dataset, seed: int = 42, target_size: int = 100) -> Dataset:
    """Return a stratified-by-source sample of `target_size` rows (or all rows if smaller)."""
    if len(pool) <= target_size:
        return pool

    by_source: dict[str, list[int]] = {}
    for i, src in enumerate(pool["source"]):
        by_source.setdefault(src, []).append(i)

    rng = _random.Random(seed)
    sampled: list[int] = []
    for _src, idxs in by_source.items():
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


def build_pool(
    sources: tuple[str, ...] = SOURCES_DEFAULT,
    smoke: bool = False,
    max_rows: int | None = None,
    near_dedup_threshold: float = 0.9,
    near_dedup_num_perm: int = 64,
    wiki_holdout_size: int = 5000,
    wiki_seed: int = 42,
) -> tuple[Dataset, Dataset | None]:
    """Build the training pool and (optionally) hold out wiki rows for retrieval eval.

    Returns (combined_pool, wiki_holdout_or_None). Holdout is None when "wiki" is not
    in `sources` or the wiki dataset failed to load.
    """
    for forbidden in FORBIDDEN_SOURCES:
        if forbidden in sources:
            raise ValueError(
                f"{forbidden} is FLORES-200 devtest and is reserved for eval.py — "
                f"never load it in prepare_data."
            )

    pool: list[Dataset] = []
    wiki_holdout: Dataset | None = None

    for name in sources:
        log.info("Loading %s ...", name)
        if name == "wiki":
            result = dataset_io.load_wiki_split(
                holdout_size=wiki_holdout_size,
                seed=wiki_seed,
                smoke=smoke,
                max_rows=max_rows,
            )
            if result is None:
                log.warning("wiki unavailable; skipping.")
                continue
            train_wiki, wiki_holdout = result
            log.info(
                "  wiki -> %d train rows + %d holdout rows", len(train_wiki), len(wiki_holdout)
            )
            pool.append(train_wiki)
            continue

        ds = dataset_io.load_source(name, smoke=smoke, max_rows=max_rows)
        if ds is None:
            log.warning("Source %s unavailable; skipping.", name)
            continue
        log.info("  %s -> %d rows after transform", name, len(ds))
        pool.append(ds)

    if not pool:
        raise RuntimeError("No sources loaded successfully — nothing to build.")

    combined = concatenate_datasets(pool) if len(pool) > 1 else pool[0]
    log.info("Combined pool: %d rows", len(combined))

    combined = dataset_io.filter_dataset(combined)
    log.info("After filter: %d rows", len(combined))

    combined = dedup.exact_dedup(combined)
    log.info("After exact dedup: %d rows", len(combined))

    combined = dedup.near_dedup_minhash(
        combined,
        fields=("anchor", "positive"),
        threshold=near_dedup_threshold,
        num_perm=near_dedup_num_perm,
    )
    log.info("After near dedup (threshold=%.2f, joint): %d rows", near_dedup_threshold, len(combined))

    return combined, wiki_holdout


def _source_distribution(ds: Dataset) -> dict[str, int]:
    counts: dict[str, int] = {}
    for src in ds["source"]:
        counts[src] = counts.get(src, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
    parser.add_argument("--allow-overwrite", action="store_true",
                        help="Allow pushing to an existing repo")
    parser.add_argument("--repo-id", default=CANONICAL_REPO_ID,
                        help="Override target repo (testing/forks)")
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
    mix_aug = augment_mixscript(combined, target_ratio=args.mixscript_ratio,
                                 seed=args.mixscript_seed)
    log.info("Mixscript rows: %d (%.1f%% of final)",
             len(mix_aug),
             len(mix_aug) / (len(combined) + len(mix_aug)) * 100 if len(mix_aug) > 0 else 0.0)
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

    if wiki_holdout is None:
        raise SystemExit("Wiki holdout was not built; cannot push wiki_retrieval_eval config.")

    stats = {
        "repo_id": args.repo_id,
        "train_rows": len(train_pool),
        "validation_rows": len(validation),
        "retrieval_rows": len(wiki_holdout),
        "smoke_rows": len(smoke),
        "source_distribution": _source_distribution(train_pool),
        "mixscript_ratio": args.mixscript_ratio,
        "validation_cyrl_source": cyrl_source,
        "near_dedup_threshold": args.near_dedup_threshold,
        "generation_date": datetime.date.today().isoformat(),
    }

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


if __name__ == "__main__":
    main()
