from datasets import Dataset

from utils.dedup import _shingle_minhash, exact_dedup, near_dedup_minhash


def _make_ds(rows: list[dict]) -> Dataset:
    return Dataset.from_list(rows)


def test_exact_dedup_removes_duplicate_pairs():
    ds = _make_ds([
        {"anchor": "Toshkent shahri.", "positive": "Tashkent city.", "source": "x"},
        {"anchor": "Toshkent shahri.", "positive": "Tashkent city.", "source": "y"},  # exact dup
        {"anchor": "Buxoro shahri.", "positive": "Bukhara city.", "source": "x"},
    ])
    out = exact_dedup(ds)
    assert len(out) == 2


def test_exact_dedup_is_case_insensitive():
    ds = _make_ds([
        {"anchor": "Toshkent.", "positive": "Tashkent.", "source": "x"},
        {"anchor": "TOSHKENT.", "positive": "TASHKENT.", "source": "y"},
    ])
    out = exact_dedup(ds)
    assert len(out) == 1


def test_exact_dedup_trims_whitespace():
    ds = _make_ds([
        {"anchor": "Toshkent.", "positive": "Tashkent.", "source": "x"},
        {"anchor": "  Toshkent.  ", "positive": "  Tashkent.  ", "source": "y"},
    ])
    out = exact_dedup(ds)
    assert len(out) == 1


def test_exact_dedup_keeps_unrelated():
    ds = _make_ds([
        {"anchor": "Toshkent shahri.", "positive": "Tashkent city.", "source": "x"},
        {"anchor": "Buxoro shahri.", "positive": "Bukhara city.", "source": "x"},
        {"anchor": "Samarqand shahri.", "positive": "Samarkand city.", "source": "x"},
    ])
    out = exact_dedup(ds)
    assert len(out) == 3


def test_near_dedup_removes_near_duplicates():
    """Two anchors share most of their 2-gram shingles; near-dedup should drop one."""
    long_a = "Toshkent Oʻzbekistonning poytaxti va eng katta shahri hisoblanadi."
    near_b = "Toshkent Oʻzbekistonning poytaxti va eng katta shahri sanaladi."  # 1 word swap
    ds = _make_ds([
        {"anchor": long_a, "positive": "Tashkent is the capital of Uzbekistan."},
        {"anchor": near_b, "positive": "Tashkent is Uzbekistan's largest city."},
        {"anchor": "Buxoro qadimiy madaniyat va savdo markazi sifatida tarixda mashhur.", "positive": "Bukhara is famous as an ancient cultural and trade center."},
    ])
    out = near_dedup_minhash(ds, threshold=0.5)
    assert len(out) == 2  # near-pair removed, unrelated one kept


def test_near_dedup_preserves_unrelated():
    ds = _make_ds([
        {"anchor": "Toshkent shahri katta va gavjum.", "positive": "Tashkent is busy."},
        {"anchor": "Buxoro qadimiy shahar.", "positive": "Bukhara is ancient."},
        {"anchor": "Samarqand mashhur shahar.", "positive": "Samarkand is famous."},
    ])
    out = near_dedup_minhash(ds, threshold=0.9)
    assert len(out) == 3


def test_near_dedup_handles_short_strings():
    ds = _make_ds([
        {"anchor": "Salom", "positive": "Hello"},
        {"anchor": "Toshkent", "positive": "Tashkent"},
    ])
    out = near_dedup_minhash(ds, threshold=0.9)
    assert len(out) == 2


def test_shingle_minhash_empty_string():
    mh = _shingle_minhash("")
    assert mh.count() >= 0  # empty MinHash should not crash


def test_shingle_minhash_single_word():
    mh = _shingle_minhash("Salom")
    assert mh.count() > 0


def test_near_dedup_field_selection():
    """near_dedup_minhash operates on `field` (default anchor)."""
    ds = _make_ds([
        {"anchor": "Same uzbek text appears here once.", "positive": "Different english A here."},
        {"anchor": "Same uzbek text appears here once.", "positive": "Different english B there."},
    ])
    out = near_dedup_minhash(ds, field="anchor", threshold=0.9)
    assert len(out) == 1
