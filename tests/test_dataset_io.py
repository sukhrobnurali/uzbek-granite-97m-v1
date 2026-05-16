import pytest
from datasets import Dataset

from utils.dataset_io import (
    UNIFIED_COLS,
    _split_wiki_filtered,
    filter_dataset,
    is_valid_uzbek_side,
    load_flores_plus,
    passes_filters,
    transform_opus100,
    transform_parallel_opus,
    transform_tatoeba,
    transform_wiki,
)


def make_opus100_fixture():
    return Dataset.from_dict({
        "translation": [
            {"en": "Tashkent is the capital.", "uz": "Toshkent — poytaxt."},
            {"en": "Hello world.", "uz": "Salom dunyo."},
            {"en": "Bukhara is a city.", "uz": "Buxoro — shahar."},
        ]
    })


def make_parallel_opus_fixture():
    return Dataset.from_dict({
        "english": ["Tashkent is the capital.", "Bukhara is a city."],
        "non_english": ["Toshkent — poytaxt.", "Buxoro — shahar."],
    })


def make_wiki_fixture():
    return Dataset.from_dict({
        "id": ["1", "2", "3", "4"],
        "title": [
            "Toshkent",
            "Roʻyxat: Oʻzbekiston shaharlari",
            "Stub",
            "Andijon",
        ],
        "text": [
            (
                "Toshkent — Oʻzbekiston Respublikasining poytaxti va eng yirik shahri. "
                "Aholisi 2024-yil holatiga koʻra 2,8 million kishidan iborat. Shahar "
                "Markaziy Osiyoning eng muhim siyosiy, iqtisodiy va madaniy markazlaridan "
                "biridir. Toshkent metropoliteni 1977-yilda ochilgan va Markaziy Osiyodagi "
                "birinchi metro tizimidir. Shahar Chirchiq daryosi sohilida joylashgan."
            ),
            "Bu sahifa Oʻzbekiston shaharlarining roʻyxatini taqdim etadi.",
            "Stub artikl, juda qisqa.",
            (
                "Andijon — Oʻzbekistonning sharqida joylashgan shahar va Andijon "
                "viloyatining maʼmuriy markazi. Fargʻona vodiysining yirik shaharlaridan "
                "biri hisoblanadi. Aholisi yarim millionga yaqin. Andijon qadimiy "
                "shaharlardan boʻlib, tarixi 2500 yildan ortiqroqqa borib taqaladi. "
                "Bobur Mirzoning vatani sifatida ham mashhurdir."
            ),
        ],
    })


def make_tatoeba_fixture():
    return Dataset.from_dict({
        "sourceString": ["Salom dunyo.", "Hello world."],
        "targetString": ["Hello world.", "Salom dunyo."],
        "sourceLang": ["uzb", "eng"],
        "targetLang": ["eng", "uzb"],
    })


def test_opus100_transform_returns_unified_schema():
    ds = transform_opus100(make_opus100_fixture())
    assert set(ds.column_names) == set(UNIFIED_COLS)
    assert len(ds) == 3
    assert ds[0]["anchor"] == "Toshkent — poytaxt."
    assert ds[0]["positive"] == "Tashkent is the capital."
    assert ds[0]["source"] == "opus100"
    assert ds[0]["anchor_lang"] == "uz_Latn"


def test_parallel_opus_transform_returns_unified_schema():
    ds = transform_parallel_opus(make_parallel_opus_fixture())
    assert set(ds.column_names) == set(UNIFIED_COLS)
    assert ds[0]["anchor"] == "Toshkent — poytaxt."
    assert ds[0]["positive"] == "Tashkent is the capital."
    assert ds[0]["source"] == "parallel_opus"


def test_flores_plus_joins_languages_by_id(monkeypatch, tmp_path):
    import utils.dataset_io as dio

    uz_file = tmp_path / "uzn.jsonl"
    en_file = tmp_path / "eng.jsonl"
    uz_file.write_text(
        '{"id": 1, "text": "Bugun Toshkentda yomgir."}\n'
        '{"id": 2, "text": "Samarqand qadimiy."}\n'
        '{"id": 3, "text": "Orphan uzbek row."}\n',
        encoding="utf-8",
    )
    en_file.write_text(
        '{"id": 1, "text": "It rained in Tashkent today."}\n'
        '{"id": 2, "text": "Samarkand is ancient."}\n',
        encoding="utf-8",
    )

    def fake_download(repo_id, filename, **kwargs):
        return str(uz_file if "uzn" in filename else en_file)

    monkeypatch.setattr(dio, "hf_hub_download", fake_download)

    ds = load_flores_plus("dev", "flores_dev_latn")
    assert set(ds.column_names) == set(UNIFIED_COLS)
    assert len(ds) == 2  # row 3 has no English pair, dropped
    assert ds[0]["source"] == "flores_dev_latn"
    assert "Toshkentda" in ds[0]["anchor"]
    assert "Tashkent" in ds[0]["positive"]


def test_wiki_transform_drops_bad_articles():
    ds = transform_wiki(make_wiki_fixture())
    sources = ds.unique("source")
    assert sources == ["wiki"]
    titles_kept = [row["anchor"] for row in ds]
    assert "Toshkent" in titles_kept
    assert "Andijon" in titles_kept
    assert "Roʻyxat: Oʻzbekiston shaharlari" not in titles_kept
    assert "Stub" not in titles_kept


def test_wiki_pair_is_title_to_paragraph():
    ds = transform_wiki(make_wiki_fixture())
    toshkent_row = next(r for r in ds if r["anchor"] == "Toshkent")
    assert toshkent_row["positive"].startswith("Toshkent")
    assert len(toshkent_row["positive"].split()) <= 80


def test_tatoeba_transform_anchors_uzbek_side():
    ds = transform_tatoeba(make_tatoeba_fixture())
    assert ds[0]["anchor"] == "Salom dunyo."  # sourceLang=uzb -> anchor=src
    assert ds[1]["anchor"] == "Salom dunyo."  # sourceLang=eng -> anchor=trg (the uz one)


def test_passes_filters_accepts_valid_pair():
    row = {
        "anchor": "Toshkent — Oʻzbekistonning poytaxti.",
        "positive": "Tashkent is the capital of Uzbekistan.",
        "source": "opus100",
        "anchor_lang": "uz_Latn",
        "positive_lang": "en",
    }
    assert passes_filters(row) is True


def test_passes_filters_rejects_too_short():
    row = {"anchor": "Hi", "positive": "Salom", "source": "x", "anchor_lang": "uz_Latn", "positive_lang": "en"}
    assert passes_filters(row) is False


def test_passes_filters_rejects_one_word_sides():
    row = {"anchor": "Toshkent.", "positive": "Tashkent.", "source": "x", "anchor_lang": "uz_Latn", "positive_lang": "en"}
    assert passes_filters(row) is False


def test_passes_filters_rejects_extreme_length_ratio():
    row = {
        "anchor": "Toshkent shahri.",
        "positive": "Tashkent is the capital of Uzbekistan, a very large and historic city in Central Asia with a long history dating back many centuries.",
        "source": "x",
        "anchor_lang": "uz_Latn",
        "positive_lang": "en",
    }
    assert passes_filters(row) is False


def test_passes_filters_rejects_identity():
    row = {
        "anchor": "Hello world hello.",
        "positive": "Hello World Hello.",
        "source": "x",
        "anchor_lang": "en",
        "positive_lang": "en",
    }
    assert passes_filters(row) is False


def test_passes_filters_rejects_empty():
    row = {"anchor": "", "positive": "Tashkent is the capital.", "source": "x", "anchor_lang": "uz_Latn", "positive_lang": "en"}
    assert passes_filters(row) is False
    row2 = {"anchor": "Toshkent — poytaxt.", "positive": "", "source": "x", "anchor_lang": "uz_Latn", "positive_lang": "en"}
    assert passes_filters(row2) is False


def test_passes_filters_wiki_allows_single_word_title():
    """Wiki anchors are titles like 'Toshkent' — single words must pass."""
    row = {
        "anchor": "Toshkent",
        "positive": "Toshkent shahri Oʻzbekistonning poytaxti hisoblanadi va eng katta shahar.",
        "source": "wiki",
        "anchor_lang": "uz_Latn",
        "positive_lang": "uz_Latn",
    }
    assert passes_filters(row) is True


def test_passes_filters_wiki_allows_short_to_long_ratio():
    """Wiki title (3 words) vs paragraph (80 words) blows past the global 0.4-2.5 ratio
    AND the 512-char MAX_LEN — both must be relaxed for wiki specifically.
    """
    title = "Andijon shahri tarixi"
    # 80 words of typical Uzbek length -> well over 512 chars, matching real wiki data
    paragraph = " ".join(["Andijon"] + ["qadimiy"] * 79)
    assert len(paragraph) > 512  # sanity: this would fail the global MAX_LEN
    row = {
        "anchor": title,
        "positive": paragraph,
        "source": "wiki",
        "anchor_lang": "uz_Latn",
        "positive_lang": "uz_Latn",
    }
    assert passes_filters(row) is True


def test_passes_filters_non_wiki_still_requires_two_words():
    """Wiki exception is source-scoped — OPUS-100 single-word anchors still rejected."""
    row = {
        "anchor": "Toshkent",
        "positive": "Tashkent is great.",
        "source": "opus100",
        "anchor_lang": "uz_Latn",
        "positive_lang": "en",
    }
    assert passes_filters(row) is False


def test_passes_filters_rejects_uzbek_side_without_vowels():
    row = {
        "anchor": ".....!!!!!",
        "positive": "Tashkent is the capital.",
        "source": "x",
        "anchor_lang": "uz_Latn",
        "positive_lang": "en",
    }
    assert passes_filters(row) is False


def test_is_valid_uzbek_side_latin():
    assert is_valid_uzbek_side("Toshkent shahri") is True
    assert is_valid_uzbek_side("...!!!") is False
    assert is_valid_uzbek_side("") is False
    assert is_valid_uzbek_side("123 456") is False


def test_is_valid_uzbek_side_cyrillic():
    assert is_valid_uzbek_side("Тошкент шаҳри") is True
    assert is_valid_uzbek_side("шшшш") is False  # consonants only, no vowels


def test_filter_dataset_applies_passes_filters():
    ds = Dataset.from_list([
        {"anchor": "Toshkent shahri.", "positive": "Tashkent city.", "source": "x", "anchor_lang": "uz_Latn", "positive_lang": "en"},  # 2-word, should pass
        {"anchor": "Hi", "positive": "Salom dunyo.", "source": "x", "anchor_lang": "uz_Latn", "positive_lang": "en"},  # too short
        {"anchor": "", "positive": "Tashkent is the capital.", "source": "x", "anchor_lang": "uz_Latn", "positive_lang": "en"},  # empty
    ])
    filtered = filter_dataset(ds)
    assert len(filtered) == 1
    assert filtered[0]["anchor"] == "Toshkent shahri."


def test_apostrophe_normalized_in_transform():
    raw = Dataset.from_dict({
        "translation": [
            {"en": "Uzbekistan is in Central Asia.", "uz": "O'zbekiston Markaziy Osiyoda joylashgan."},
        ]
    })
    ds = transform_opus100(raw)
    assert "Oʻzbekiston" in ds[0]["anchor"]


def test_unknown_source_raises():
    from utils.dataset_io import load_source

    with pytest.raises(KeyError):
        load_source("does_not_exist")


def _fake_filtered_wiki(n: int) -> Dataset:
    return Dataset.from_list([
        {
            "anchor": f"Title {i}",
            "positive": f"Article {i} body text with several words to pass filters.",
            "source": "wiki",
            "anchor_lang": "uz_Latn",
            "positive_lang": "uz_Latn",
        }
        for i in range(n)
    ])


def test_split_wiki_filtered_is_reproducible_with_same_seed():
    ds = _fake_filtered_wiki(50)
    train_a, hold_a = _split_wiki_filtered(ds, holdout_size=10, seed=42)
    train_b, hold_b = _split_wiki_filtered(ds, holdout_size=10, seed=42)
    assert list(hold_a["anchor"]) == list(hold_b["anchor"])
    assert list(train_a["anchor"]) == list(train_b["anchor"])


def test_split_wiki_filtered_different_seeds_differ():
    ds = _fake_filtered_wiki(50)
    _, hold_a = _split_wiki_filtered(ds, holdout_size=10, seed=42)
    _, hold_b = _split_wiki_filtered(ds, holdout_size=10, seed=1)
    assert set(hold_a["anchor"]) != set(hold_b["anchor"])


def test_split_wiki_filtered_no_overlap():
    ds = _fake_filtered_wiki(50)
    train, holdout = _split_wiki_filtered(ds, holdout_size=10, seed=42)
    assert len(train) == 40
    assert len(holdout) == 10
    assert set(train["anchor"]) & set(holdout["anchor"]) == set()
    assert set(train["anchor"]) | set(holdout["anchor"]) == set(ds["anchor"])


def test_split_wiki_filtered_rejects_oversized_holdout():
    ds = _fake_filtered_wiki(10)
    with pytest.raises(ValueError):
        _split_wiki_filtered(ds, holdout_size=10, seed=42)
    with pytest.raises(ValueError):
        _split_wiki_filtered(ds, holdout_size=0, seed=42)
