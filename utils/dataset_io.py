from __future__ import annotations

import logging
import re
from typing import Any

from datasets import Dataset, load_dataset

from .translit import auto_detect_script, normalize_text

log = logging.getLogger(__name__)

UNIFIED_COLS = ["anchor", "positive", "source", "anchor_lang", "positive_lang"]

# Filter thresholds (matches the plan's pipeline filters).
MIN_LEN = 5
MAX_LEN = 512
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
    if not (MIN_LEN <= len(a) <= MAX_LEN and MIN_LEN <= len(p) <= MAX_LEN):
        return False
    if _word_count(a) < MIN_WORDS or _word_count(p) < MIN_WORDS:
        return False
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


def transform_flores(
    ds: Dataset,
    uz_field: str = "sentence_uzn_Latn",
    en_field: str = "sentence_eng_Latn",
    source_tag: str = "flores",
) -> Dataset:
    """FLORES-200 schema: {"id", "URL", "domain", "topic", "sentence_<lang>_<script>", ...}."""

    def _map(row: dict) -> dict:
        uz = row.get(uz_field, "")
        en = row.get(en_field, "")
        return _make_row(uz, en, source_tag)

    return ds.map(_map, remove_columns=ds.column_names)


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
    return ds.filter(passes_filters)


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
    "flores_dev_latn": {
        "hf_id": "Muennighoff/flores200",
        "config": "uzn_Latn-eng_Latn",
        "split": "dev",
        "transform": lambda ds: transform_flores(ds, "sentence_uzn_Latn", "sentence_eng_Latn", "flores_dev_latn"),
    },
    "flores_dev_cyrl": {
        "hf_id": "Muennighoff/flores200",
        "config": "uzn_Cyrl-eng_Latn",
        "split": "dev",
        "transform": lambda ds: transform_flores(ds, "sentence_uzn_Cyrl", "sentence_eng_Latn", "flores_dev_cyrl"),
    },
    "flores_devtest_latn": {
        "hf_id": "Muennighoff/flores200",
        "config": "uzn_Latn-eng_Latn",
        "split": "devtest",
        "transform": lambda ds: transform_flores(ds, "sentence_uzn_Latn", "sentence_eng_Latn", "flores_devtest_latn"),
    },
    "flores_devtest_cyrl": {
        "hf_id": "Muennighoff/flores200",
        "config": "uzn_Cyrl-eng_Latn",
        "split": "devtest",
        "transform": lambda ds: transform_flores(ds, "sentence_uzn_Cyrl", "sentence_eng_Latn", "flores_devtest_cyrl"),
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
    (e.g. FLORES-200 uzn_Cyrl config missing), so callers can fall through to a
    documented fallback path rather than crashing.
    """
    if name not in _LOADERS:
        raise KeyError(f"Unknown source: {name}. Known: {sorted(_LOADERS.keys())}")
    spec = _LOADERS[name]
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
