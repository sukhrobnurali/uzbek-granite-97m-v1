from __future__ import annotations

from datasets import Dataset
from datasketch import MinHash, MinHashLSH


def exact_dedup(ds: Dataset, fields: tuple[str, ...] = ("anchor", "positive")) -> Dataset:
    """Drop rows whose (anchor.lower(), positive.lower()) tuple was already seen."""
    seen: set[tuple[str, ...]] = set()
    keep: list[int] = []
    for i, row in enumerate(ds):
        key = tuple((row[f] or "").lower().strip() for f in fields)
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return ds.select(keep)


def _shingle_minhash(text: str, num_perm: int = 64, shingle_size: int = 2) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    words = (text or "").lower().split()
    if not words:
        return mh
    if len(words) < shingle_size:
        mh.update(" ".join(words).encode("utf-8"))
        return mh
    for i in range(len(words) - shingle_size + 1):
        shingle = " ".join(words[i : i + shingle_size])
        mh.update(shingle.encode("utf-8"))
    return mh


def near_dedup_minhash(
    ds: Dataset,
    field: str = "anchor",
    threshold: float = 0.9,
    num_perm: int = 64,
    shingle_size: int = 2,
) -> Dataset:
    """Drop near-duplicates on `field` using MinHashLSH (Jaccard similarity)."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep: list[int] = []
    for i, row in enumerate(ds):
        mh = _shingle_minhash(row[field], num_perm=num_perm, shingle_size=shingle_size)
        if lsh.query(mh):
            continue
        lsh.insert(str(i), mh)
        keep.append(i)
    return ds.select(keep)
