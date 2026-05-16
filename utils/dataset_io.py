from __future__ import annotations

import json
import logging
import random
import re
from typing import Any

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download

from .translit import auto_detect_script, normalize_text

log = logging.getLogger(__name__)

UNIFIED_COLS = ["anchor", "positive", "source", "anchor_lang", "positive_lang"]

# Filter thresholds (matches the plan's pipeline filters).
MIN_LEN = 5
MAX_LEN = 512
WIKI_MAX_LEN = 2000  # wiki positives are 80-word paragraphs (~600-1000 chars typical)
MIN_WORDS = 2
MIN_LEN_RATIO = 0.4
MAX_LEN_RATIO = 2.5

# Wiki title-filter regex — drops disambiguation, list, "uncertain" pages
# in both Uzbek and English markers since Uzbek Wikipedia uses both.
WIKI_TITLE_DROP_RE = re.compile(
    r"(ro[ʻ']?yxat|disambig|noaniqlik|\(disambiguation\)|list of)",
    re.IGNORECASE,
)

# Latin-script Uzbek "looks like Uzbek" sanity check: at least one vowel and
# at least one consonant. Rejects emoji/punctuation-only strings.
LATIN_VOWELS_RE = re.compile(r"[aeiouAEIOUäöü]")
LATIN_CONSONANTS_RE = re.compile(r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]")
CYR_VOWELS_RE = re.compile(r"[аеёиоуүэюяАЕЁИОУҮЭЮЯ]")
CYR_CONSONANTS_RE = re.compile(r"[бвгджзйклмнпрстфхцчшщЬЪ]")


def _word_count(s: str) -> int:
    return len(s.split())


def _make_row(anchor: str, positive: str, source: str) -> dict:
    a = normalize_text(anchor)
    p = normalize_text(positive)
    return {
        "anchor": a,
        "positive": p,
        "source": source,
        "anchor_lang": auto_detect_script(a),
        "positive_lang": auto_detect_script(p),
    }


def is_valid_uzbek_side(text: str) -> bool:
    if not text:
        return False
    has_letters = bool(re.search(r"[A-Za-zЀ-ӿ]", text))
    if not has_letters:
        return False
    if CYRILLIC_RE.search(text):
        return bool(CYR_VOWELS_RE.search(text)) and bool(CYR_CONSONANTS_RE.search(text))
    return bool(LATIN_VOWELS_RE.search(text)) and bool(LATIN_CONSONANTS_RE.search(text))


CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")


def passes_filters(row: dict, uzbek_field: str = "anchor") -> bool:
    a = row.get("anchor") or ""
    p = row.get("positive") or ""
    if not a or not p:
        return False
    is_wiki = row.get("source") == "wiki"
    # Wiki positives are 80-word paragraphs (often ~600-1000 chars), and wiki
    # anchors are titles (often 1 word like "Toshkent"). Both the per-side MAX_LEN
    # and the 0.4-2.5 length ratio are too tight for the title/paragraph geometry.
    # The 80-word cap in transform_wiki is the real upper bound for wiki positives.
    p_max_len = WIKI_MAX_LEN if is_wiki else MAX_LEN
    if not (MIN_LEN <= len(a) <= MAX_LEN and MIN_LEN <= len(p) <= p_max_len):
        return False
    min_words_anchor = 1 if is_wiki else MIN_WORDS
    if _word_count(a) < min_words_anchor or _word_count(p) < MIN_WORDS:
        return False
    if not is_wiki:
        ratio = len(a) / len(p)
        if ratio < MIN_LEN_RATIO or ratio > MAX_LEN_RATIO:
            return False
    if a.lower() == p.lower():
        return False
    uz_text = row.get(uzbek_field) or ""
    return is_valid_uzbek_side(uz_text)


def transform_opus100(ds: Dataset, source_tag: str = "opus100") -> Dataset:
    """OPUS-100 schema: {"translation": {"en": str, "uz": str}}. Uzbek -> anchor."""

    def _map(row: dict) -> dict:
        en = row["translation"].get("en", "")
        uz = row["translation"].get("uz", "")
        return _make_row(uz, en, source_tag)

    return ds.map(_map, remove_columns=ds.column_names)


def transform_parallel_opus(ds: Dataset, source_tag: str = "parallel_opus") -> Dataset:
    """parallel-sentences-opus-100 schema: {"english": str, "non_english": str}."""

    def _map(row: dict) -> dict:
        en = row.get("english", "")
        uz = row.get("non_english", "")
        return _make_row(uz, en, source_tag)

    return ds.map(_map, remove_columns=ds.column_names)


FLORES_PLUS_REPO = "openlanguagedata/flores_plus"


def _read_flores_jsonl(split: str, lang: str) -> dict[int, str]:
    """Download one flores_plus per-language jsonl and return {id: text}."""
    path = hf_hub_download(FLORES_PLUS_REPO, f"{split}/{lang}.jsonl", repo_type="dataset")
    rows: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[row["id"]] = row["text"]
    return rows


def load_flores_plus(split: str, source_tag: str) -> Dataset:
    """Load FLORES+ uzn_Latn / eng_Latn for one split, paired by row id.

    Each language is a single jsonl file; pairs are constructed by joining on `id`.
    flores_plus has no uzn_Cyrl variant, so build_validation falls back to translit.
    """
    uz_rows = _read_flores_jsonl(split, "uzn_Latn")
    en_rows = _read_flores_jsonl(split, "eng_Latn")
    paired = [
        _make_row(uz_text, en_rows[rid], source_tag)
        for rid, uz_text in uz_rows.items()
        if rid in en_rows
    ]
    return Dataset.from_list(paired)


def transform_wiki(
    ds: Dataset,
    title_field: str = "title",
    text_field: str = "text",
    paragraph_words: int = 80,
    source_tag: str = "wiki",
) -> Dataset:
    """yakhyo/uz-wiki schema: {"id", "title", "text", ...}. Pairs title <-> first N words."""

    def _map(row: dict) -> dict:
        title = (row.get(title_field) or "").strip()
        text = (row.get(text_field) or "").strip()
        if WIKI_TITLE_DROP_RE.search(title):
            return _make_row("", "", source_tag)
        words = text.split()
        if len(words) < 30:
            return _make_row("", "", source_tag)
        first_para = " ".join(words[:paragraph_words])
        first_sentence = first_para.split(".", 1)[0]
        if title and title.lower() in first_sentence.lower() and len(first_sentence) < len(title) * 2:
            return _make_row("", "", source_tag)
        return _make_row(title, first_para, source_tag)

    out = ds.map(_map, remove_columns=ds.column_names)
    return out.filter(lambda r: r["anchor"] != "" and r["positive"] != "")


def transform_tatoeba(
    ds: Dataset,
    src_field: str = "sourceString",
    trg_field: str = "targetString",
    src_lang_value: str = "uzb",
    source_tag: str = "tatoeba",
) -> Dataset:
    """Tatoeba MT schema is variable; we accept the most common columns and tag source-side."""

    def _map(row: dict) -> dict:
        src = row.get(src_field, "")
        trg = row.get(trg_field, "")
        src_lang = (row.get("sourceLang") or row.get("src_lang") or "").lower()
        if src_lang.startswith(src_lang_value):
            return _make_row(src, trg, source_tag)
        return _make_row(trg, src, source_tag)

    return ds.map(_map, remove_columns=ds.column_names)


def filter_dataset(ds: Dataset) -> Dataset:
    # load_from_cache_file=False: passes_filters is a closure over module-level
    # constants (MIN_LEN, WIKI_MAX_LEN, etc.) whose changes don't always invalidate
    # HF datasets' fingerprint. Cheap to recompute (~1s/100k rows), so always re-run.
    return ds.filter(passes_filters, load_from_cache_file=False)


_LOADERS: dict[str, dict[str, Any]] = {
    "opus100": {
        "hf_id": "Helsinki-NLP/opus-100",
        "config": "en-uz",
        "split": "train",
        "transform": transform_opus100,
    },
    "parallel_opus": {
        "hf_id": "sentence-transformers/parallel-sentences-opus-100",
        "config": "en-uz",
        "split": "train",
        "transform": transform_parallel_opus,
    },
    # FLORES via openlanguagedata/flores_plus: per-language jsonl, joined by row id.
    # flores_plus has no uzn_Cyrl, so the *_cyrl loaders return None and
    # build_validation falls through to translit fallback (documented in card).
    "flores_dev_latn": {
        "custom_loader": lambda: load_flores_plus("dev", "flores_dev_latn"),
    },
    "flores_dev_cyrl": {
        "custom_loader": lambda: None,
    },
    "flores_devtest_latn": {
        "custom_loader": lambda: load_flores_plus("devtest", "flores_devtest_latn"),
    },
    "flores_devtest_cyrl": {
        "custom_loader": lambda: None,
    },
    "wiki": {
        "hf_id": "yakhyo/uz-wiki",
        "config": None,
        "split": "train",
        "transform": transform_wiki,
    },
    "tatoeba": {
        "hf_id": "Helsinki-NLP/tatoeba_mt",
        "config": "eng-uzb",
        "split": "test",
        "transform": transform_tatoeba,
    },
}


def load_source(name: str, smoke: bool = False, max_rows: int | None = None) -> Dataset | None:
    """Load and transform a source by name. Returns None if the source is unavailable
    (e.g. flores_plus uzn_Cyrl absent), so callers can fall through to a documented
    fallback path rather than crashing.
    """
    if name not in _LOADERS:
        raise KeyError(f"Unknown source: {name}. Known: {sorted(_LOADERS.keys())}")
    spec = _LOADERS[name]

    if "custom_loader" in spec:
        try:
            ds = spec["custom_loader"]()
        except Exception as exc:
            log.warning("Could not load %s via custom_loader: %s", name, exc)
            return None
        if ds is None:
            return None
        if smoke:
            ds = ds.select(range(min(50, len(ds))))
        elif max_rows is not None:
            ds = ds.select(range(min(max_rows, len(ds))))
        return ds

    kwargs = {}
    if spec["config"] is not None:
        kwargs["name"] = spec["config"]
    try:
        ds = load_dataset(spec["hf_id"], split=spec["split"], **kwargs)
    except Exception as exc:
        log.warning("Could not load %s (%s, %s): %s", name, spec["hf_id"], spec["config"], exc)
        return None
    if smoke:
        ds = ds.select(range(min(50, len(ds))))
    elif max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))
    return spec["transform"](ds)


def _split_wiki_filtered(
    filtered: Dataset, holdout_size: int, seed: int
) -> tuple[Dataset, Dataset]:
    """Pure function: deterministic train/holdout split of a transformed+filtered wiki dataset.

    Returns (train_pool, holdout). Holdout is taken from the filtered survivors so the
    eval distribution matches training distribution; same seed always yields the same split.
    Raises ValueError if there are fewer rows than the requested holdout.
    """
    if holdout_size <= 0:
        raise ValueError(f"holdout_size must be positive, got {holdout_size}")
    if holdout_size >= len(filtered):
        raise ValueError(
            f"holdout_size {holdout_size} >= filtered wiki rows {len(filtered)}; "
            f"increase max_rows or shrink holdout."
        )
    indices = list(range(len(filtered)))
    random.Random(seed).shuffle(indices)
    holdout_idx = sorted(indices[:holdout_size])
    train_idx = sorted(indices[holdout_size:])
    return filtered.select(train_idx), filtered.select(holdout_idx)


def load_wiki_split(
    holdout_size: int = 5000,
    seed: int = 42,
    smoke: bool = False,
    max_rows: int | None = None,
) -> tuple[Dataset, Dataset] | None:
    """Load yakhyo/uz-wiki, transform+filter, then split into (train, holdout).

    Smoke mode scales the holdout down to ~10% of loaded rows so a 50-article smoke
    doesn't try to hold out 5000. Returns None if the wiki dataset is unavailable.
    """
    spec = _LOADERS["wiki"]
    try:
        raw = load_dataset(spec["hf_id"], split=spec["split"])
    except Exception as exc:
        log.warning("Could not load wiki (%s): %s", spec["hf_id"], exc)
        return None
    if smoke:
        raw = raw.select(range(min(200, len(raw))))
        holdout_size = max(1, len(raw) // 10)
    elif max_rows is not None:
        raw = raw.select(range(min(max_rows, len(raw))))

    transformed = transform_wiki(raw)
    filtered = filter_dataset(transformed)
    if len(filtered) <= holdout_size:
        # Defensive: shrink holdout to at most 10% of survivors so smoke modes
        # never bottom out on tiny corpora.
        holdout_size = max(1, len(filtered) // 10)
    return _split_wiki_filtered(filtered, holdout_size=holdout_size, seed=seed)
