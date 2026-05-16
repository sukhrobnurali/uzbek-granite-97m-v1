"""Build the uzbek-embedding-pairs dataset.

Task 5 scope: OPUS-100 (en-uz) -> unified schema -> filter -> dedup -> local parquet.

Future expansion (Tasks 6-7):
  - Add parallel-sentences-opus-100, FLORES dev, yakhyo/uz-wiki, Tatoeba MT
  - Wiki holdout 5000 rows -> wiki_retrieval_eval config
  - Mix-script augmentation at 10% via utils.translit
  - Push to sukhrobnurali/uzbek-embedding-pairs with DatasetCard

CLI:
    python prepare_data.py                          # full OPUS-100 -> data/processed/default.parquet
    python prepare_data.py --smoke                  # 50 rows
    python prepare_data.py --max-rows 5000          # cap source size
    python prepare_data.py --output-dir mydata      # custom output dir
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, concatenate_datasets

from utils import dataset_io, dedup
from utils.logging_setup import configure

log = configure()

SOURCES_DEFAULT = ("opus100",)


def build_pool(
    sources: tuple[str, ...] = SOURCES_DEFAULT,
    smoke: bool = False,
    max_rows: int | None = None,
    near_dedup_threshold: float = 0.9,
    near_dedup_num_perm: int = 64,
) -> Dataset:
    pool: list[Dataset] = []
    for name in sources:
        log.info("Loading %s ...", name)
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
        field="anchor",
        threshold=near_dedup_threshold,
        num_perm=near_dedup_num_perm,
    )
    log.info("After near dedup (threshold=%.2f): %d rows", near_dedup_threshold, len(combined))

    return combined


def _source_distribution(ds: Dataset) -> dict[str, int]:
    counts: dict[str, int] = {}
    for src in ds["source"]:
        counts[src] = counts.get(src, 0) + 1
    return counts


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
    args = parser.parse_args()

    combined = build_pool(
        sources=tuple(args.sources),
        smoke=args.smoke,
        max_rows=args.max_rows,
        near_dedup_threshold=args.near_dedup_threshold,
        near_dedup_num_perm=args.near_dedup_num_perm,
    )

    distribution = _source_distribution(combined)
    log.info("Source distribution: %s", distribution)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "default.parquet"
    combined.to_parquet(str(output_path))
    log.info("Wrote %d rows to %s", len(combined), output_path)


if __name__ == "__main__":
    main()
