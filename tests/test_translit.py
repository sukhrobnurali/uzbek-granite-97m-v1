import pytest

from utils.translit import (
    _rule_cyr_to_lat,
    _rule_lat_to_cyr,
    auto_detect_script,
    normalize_text,
    to_cyrillic,
    to_latin,
)

CYR_TO_LAT_PAIRS = [
    ("Тошкент", "Toshkent"),
    ("Ўзбекистон", "Oʻzbekiston"),
    ("Самарқанд", "Samarqand"),
    ("Бухоро", "Buxoro"),
    ("Хива", "Xiva"),
    ("Ғафур", "Gʻafur"),
    ("Ҳамза", "Hamza"),
    ("Қарши", "Qarshi"),
    ("Андижон", "Andijon"),
    ("Наманган", "Namangan"),
    ("Фарғона", "Fargʻona"),
    ("Чирчиқ", "Chirchiq"),
    ("Жиззах", "Jizzax"),
    ("Ёшлар", "Yoshlar"),
    ("Янги", "Yangi"),
]

LAT_TO_CYR_PAIRS = [
    ("Toshkent", "Тошкент"),
    ("Oʻzbekiston", "Ўзбекистон"),
    ("Samarqand", "Самарқанд"),
    ("Buxoro", "Бухоро"),
    ("Xiva", "Хива"),
    ("Gʻafur", "Ғафур"),
    ("Hamza", "Ҳамза"),
    ("Qarshi", "Қарши"),
    ("Andijon", "Андижон"),
    ("Namangan", "Наманган"),
    ("Fargʻona", "Фарғона"),
    ("Chirchiq", "Чирчиқ"),
    ("Jizzax", "Жиззах"),
    ("Yoshlar", "Ёшлар"),
    ("Yangi", "Янги"),
]


@pytest.mark.parametrize("cyr,lat", CYR_TO_LAT_PAIRS)
def test_cyr_to_lat_rule(cyr: str, lat: str):
    assert _rule_cyr_to_lat(cyr) == lat


@pytest.mark.parametrize("lat,cyr", LAT_TO_CYR_PAIRS)
def test_lat_to_cyr_rule(lat: str, cyr: str):
    assert _rule_lat_to_cyr(lat) == cyr


def test_uzbek_distinctive_cyr_to_lat():
    assert _rule_cyr_to_lat("Ў") == "Oʻ"
    assert _rule_cyr_to_lat("ў") == "oʻ"
    assert _rule_cyr_to_lat("Ғ") == "Gʻ"
    assert _rule_cyr_to_lat("ғ") == "gʻ"
    assert _rule_cyr_to_lat("Қ") == "Q"
    assert _rule_cyr_to_lat("қ") == "q"
    assert _rule_cyr_to_lat("Ҳ") == "H"
    assert _rule_cyr_to_lat("ҳ") == "h"


def test_uzbek_distinctive_lat_to_cyr():
    assert _rule_lat_to_cyr("Oʻ") == "Ў"
    assert _rule_lat_to_cyr("oʻ") == "ў"
    assert _rule_lat_to_cyr("Gʻ") == "Ғ"
    assert _rule_lat_to_cyr("gʻ") == "ғ"
    assert _rule_lat_to_cyr("Q") == "Қ"
    assert _rule_lat_to_cyr("q") == "қ"
    assert _rule_lat_to_cyr("H") == "Ҳ"
    assert _rule_lat_to_cyr("h") == "ҳ"


def test_ye_context_rule_at_word_start():
    # Cyrillic Е at word start palatalizes -> "Ye"
    assert _rule_cyr_to_lat("Ер") == "Yer"
    assert _rule_cyr_to_lat("ер") == "yer"


def test_ye_context_rule_after_consonant():
    # Cyrillic Е after a consonant -> plain "e"
    assert _rule_cyr_to_lat("Тенг") == "Teng"
    assert _rule_cyr_to_lat("мен") == "men"


def test_ye_context_rule_after_vowel():
    # Cyrillic Е after a vowel -> "Ye"/"ye"
    assert _rule_cyr_to_lat("Оел") == "Oyel"
    assert _rule_cyr_to_lat("ие") == "iye"


def test_apostrophe_normalization():
    assert normalize_text("O'zbekiston") == "Oʻzbekiston"
    assert normalize_text("G'ofur") == "Gʻofur"
    assert normalize_text("ma'lumot") == "maʼlumot"


def test_apostrophe_normalization_variants():
    for variant in ("'", "'", "´", "`"):
        result = normalize_text(f"O{variant}zbekiston")
        assert "Oʻ" in result, f"variant {variant!r} did not normalize to Oʻ"


def test_normalize_is_nfc():
    decomposed = "Toşhkent"
    normalized = normalize_text(decomposed)
    assert normalized == "Toşhkent" or normalized == "Toșhkent" or "̧" not in normalized


def test_to_latin_empty():
    assert to_latin("") == ""


def test_to_cyrillic_empty():
    assert to_cyrillic("") == ""


def test_auto_detect_script_cyrillic():
    assert auto_detect_script("Тошкент") == "uz_Cyrl"
    assert auto_detect_script("Ўзбекистон") == "uz_Cyrl"


def test_auto_detect_script_uz_latin():
    assert auto_detect_script("Oʻzbekiston") == "uz_Latn"
    assert auto_detect_script("Qarshi") == "uz_Latn"
    assert auto_detect_script("Toshkentdan xat keldi") == "uz_Latn"  # has 'x'


def test_auto_detect_script_english():
    assert auto_detect_script("Hello world") == "en"
    assert auto_detect_script("The cat sat on the mat") == "en"


def test_auto_detect_script_mixed():
    assert auto_detect_script("Hello Тошкент") == "mixed"


def test_auto_detect_script_empty():
    assert auto_detect_script("") == "empty"
    assert auto_detect_script("   ") == "empty"
    assert auto_detect_script("123 !@#") == "empty"


def test_to_latin_uses_rule_fallback_when_uzt_unavailable(monkeypatch):
    import utils.translit as t

    monkeypatch.setattr(t, "_uz_translit_instance", None)
    monkeypatch.setattr(t, "_uz_translit_failed", True)
    assert t.to_latin("Тошкент") == "Toshkent"
    assert t.to_latin("Ўзбекистон") == "Oʻzbekiston"


def test_to_cyrillic_uses_rule_fallback_when_uzt_unavailable(monkeypatch):
    import utils.translit as t

    monkeypatch.setattr(t, "_uz_translit_instance", None)
    monkeypatch.setattr(t, "_uz_translit_failed", True)
    assert t.to_cyrillic("Toshkent") == "Тошкент"
    assert t.to_cyrillic("Oʻzbekiston") == "Ўзбекистон"


def test_fallback_when_import_raises(monkeypatch):
    """If UzTransliterator import itself raises, the wrapper must degrade gracefully."""
    import sys

    import utils.translit as t

    monkeypatch.setitem(sys.modules, "UzTransliterator", None)
    monkeypatch.setattr(t, "_uz_translit_instance", None)
    monkeypatch.setattr(t, "_uz_translit_failed", False)
    out = t.to_latin("Қарши")
    assert out == "Qarshi"
    assert t._uz_translit_failed is True
