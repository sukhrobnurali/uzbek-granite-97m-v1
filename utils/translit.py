from __future__ import annotations

import re
import unicodedata

CYRILLIC_RE = re.compile(r"[А-Яа-яЁёЎўҒғҲҳҚқ]")
LATIN_RE = re.compile(r"[A-Za-z]")
UZ_LATIN_MARKERS = ("oʻ", "Oʻ", "gʻ", "Gʻ", "qʻ", "ʻ", "ʼ")

CYR_VOWELS = set("аеёиоуўэюяАЕЁИОУЎЭЮЯ")
APOSTROPHE_VARIANTS = ("'", "'", "'", "ʻ", "ʼ", "`", "´")
TARGET_OQUOTE = "ʻ"
TARGET_GLOTTAL = "ʼ"

_CYR_TO_LAT_SINGLE: dict[str, str] = {
    "А": "A", "а": "a",
    "Б": "B", "б": "b",
    "В": "V", "в": "v",
    "Г": "G", "г": "g",
    "Д": "D", "д": "d",
    "Ё": "Yo", "ё": "yo",
    "Ж": "J", "ж": "j",
    "З": "Z", "з": "z",
    "И": "I", "и": "i",
    "Й": "Y", "й": "y",
    "К": "K", "к": "k",
    "Л": "L", "л": "l",
    "М": "M", "м": "m",
    "Н": "N", "н": "n",
    "О": "O", "о": "o",
    "П": "P", "п": "p",
    "Р": "R", "р": "r",
    "С": "S", "с": "s",
    "Т": "T", "т": "t",
    "У": "U", "у": "u",
    "Ф": "F", "ф": "f",
    "Х": "X", "х": "x",
    "Ц": "Ts", "ц": "ts",
    "Ч": "Ch", "ч": "ch",
    "Ш": "Sh", "ш": "sh",
    "Ъ": TARGET_GLOTTAL, "ъ": TARGET_GLOTTAL,
    "Ь": "", "ь": "",
    "Э": "E", "э": "e",
    "Ю": "Yu", "ю": "yu",
    "Я": "Ya", "я": "ya",
    "Ў": "O" + TARGET_OQUOTE, "ў": "o" + TARGET_OQUOTE,
    "Ғ": "G" + TARGET_OQUOTE, "ғ": "g" + TARGET_OQUOTE,
    "Ҳ": "H", "ҳ": "h",
    "Қ": "Q", "қ": "q",
}

# Longest-match-first for Latin -> Cyrillic. Digraphs and Oʻ/Gʻ before single letters.
# Maintenance: keep this list ordered by descending length, case-specific variants
# (uppercase/lowercase/all-caps) before generic single letters.
_LAT_TO_CYR_ORDERED: list[tuple[str, str]] = [
    ("Oʻ", "Ў"), ("oʻ", "ў"), ("OʻO", "ЎО"),
    ("Gʻ", "Ғ"), ("gʻ", "ғ"),
    ("SH", "Ш"), ("Sh", "Ш"), ("sh", "ш"),
    ("CH", "Ч"), ("Ch", "Ч"), ("ch", "ч"),
    ("YO", "Ё"), ("Yo", "Ё"), ("yo", "ё"),
    ("YU", "Ю"), ("Yu", "Ю"), ("yu", "ю"),
    ("YA", "Я"), ("Ya", "Я"), ("ya", "я"),
    ("YE", "Е"), ("Ye", "Е"), ("ye", "е"),
    ("TS", "Ц"), ("Ts", "Ц"), ("ts", "ц"),
    ("A", "А"), ("a", "а"),
    ("B", "Б"), ("b", "б"),
    ("V", "В"), ("v", "в"),
    ("G", "Г"), ("g", "г"),
    ("D", "Д"), ("d", "д"),
    # Single 'E' -> Cyrillic Е (the common case in Uzbek). Cyrillic Э is
    # rare and mostly loanwords; we accept the round-trip lossiness for
    # the much more frequent Е.
    ("E", "Е"), ("e", "е"),
    ("J", "Ж"), ("j", "ж"),
    ("Z", "З"), ("z", "з"),
    ("I", "И"), ("i", "и"),
    ("Y", "Й"), ("y", "й"),
    ("K", "К"), ("k", "к"),
    ("L", "Л"), ("l", "л"),
    ("M", "М"), ("m", "м"),
    ("N", "Н"), ("n", "н"),
    ("O", "О"), ("o", "о"),
    ("P", "П"), ("p", "п"),
    ("Q", "Қ"), ("q", "қ"),
    ("R", "Р"), ("r", "р"),
    ("S", "С"), ("s", "с"),
    ("T", "Т"), ("t", "т"),
    ("U", "У"), ("u", "у"),
    ("F", "Ф"), ("f", "ф"),
    ("X", "Х"), ("x", "х"),
    ("H", "Ҳ"), ("h", "ҳ"),
    (TARGET_GLOTTAL, "ъ"),
]


def _normalize_apostrophes(text: str) -> str:
    out = text
    for variant in APOSTROPHE_VARIANTS:
        if variant in ("ʻ", "ʼ"):
            continue
        # Heuristic: an apostrophe immediately after o/O/g/G is the Oʻ/Gʻ marker.
        # Anywhere else it represents the glottal stop ʼ.
        out = re.sub(rf"([oOgG]){re.escape(variant)}", r"\1" + TARGET_OQUOTE, out)
        out = out.replace(variant, TARGET_GLOTTAL)
    return out


def _normalize(text: str) -> str:
    return _normalize_apostrophes(unicodedata.normalize("NFC", text))


def _rule_cyr_to_lat(text: str) -> str:
    text = _normalize(text)
    out: list[str] = []
    for i, ch in enumerate(text):
        if ch in ("Е", "е"):
            upper = ch == "Е"
            at_start = i == 0 or not text[i - 1].isalpha()
            after_vowel = i > 0 and text[i - 1] in CYR_VOWELS
            if at_start or after_vowel:
                out.append("Ye" if upper else "ye")
            else:
                out.append("E" if upper else "e")
        else:
            out.append(_CYR_TO_LAT_SINGLE.get(ch, ch))
    return "".join(out)


def _rule_lat_to_cyr(text: str) -> str:
    text = _normalize(text)
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        matched = False
        for src, dst in _LAT_TO_CYR_ORDERED:
            if text.startswith(src, i):
                out.append(dst)
                i += len(src)
                matched = True
                break
        if not matched:
            out.append(text[i])
            i += 1
    return "".join(out)


_uz_translit_instance = None
_uz_translit_failed = False


def _try_uztransliterator():
    global _uz_translit_instance, _uz_translit_failed
    if _uz_translit_instance is not None or _uz_translit_failed:
        return _uz_translit_instance
    try:
        from UzTransliterator import UzTransliterator  # type: ignore[import-not-found]

        _uz_translit_instance = UzTransliterator.UzTransliterator()
    except Exception:
        _uz_translit_failed = True
        _uz_translit_instance = None
    return _uz_translit_instance


def to_latin(text: str) -> str:
    if not text:
        return text
    inst = _try_uztransliterator()
    if inst is not None:
        try:
            return _normalize(inst.transliterate(text, from_="cyr", to="lat"))
        except Exception:
            pass
    return _rule_cyr_to_lat(text)


def to_cyrillic(text: str) -> str:
    if not text:
        return text
    inst = _try_uztransliterator()
    if inst is not None:
        try:
            return inst.transliterate(text, from_="lat", to="cyr")
        except Exception:
            pass
    return _rule_lat_to_cyr(text)


def auto_detect_script(text: str) -> str:
    if not text or not text.strip():
        return "empty"
    has_cyr = bool(CYRILLIC_RE.search(text))
    has_lat = bool(LATIN_RE.search(text))
    if has_cyr and has_lat:
        return "mixed"
    if has_cyr:
        return "uz_Cyrl"
    if has_lat:
        if any(marker in text for marker in UZ_LATIN_MARKERS):
            return "uz_Latn"
        # Best-effort: presence of q/x (rare in English root vocab) suggests Uzbek.
        if re.search(r"[qxQX]", text):
            return "uz_Latn"
        return "en"
    return "empty"


def normalize_text(text: str) -> str:
    return _normalize(text)
