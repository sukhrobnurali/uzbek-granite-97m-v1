import pytest
from datasets import Dataset

import prepare_data


def _fake_source(source_tag: str, n: int = 5) -> Dataset:
    """Build a small valid Dataset that will survive filters.

    Includes the source tag in the text so exact_dedup doesn't collapse rows
    across sources that share the same fixture body.
    """
    return Dataset.from_list([
        {
            "anchor": f"Toshkent shahri {source_tag} {i} qadimiy va katta shahar.",
            "positive": f"Tashkent city {source_tag} number {i} is ancient and large.",
            "source": source_tag,
            "anchor_lang": "uz_Latn",
            "positive_lang": "en",
        }
        for i in range(n)
    ])


def test_build_pool_rejects_flores_devtest_latn():
    with pytest.raises(ValueError, match="reserved for eval"):
        prepare_data.build_pool(sources=("opus100", "flores_devtest_latn"))


def test_build_pool_rejects_flores_devtest_cyrl():
    with pytest.raises(ValueError, match="reserved for eval"):
        prepare_data.build_pool(sources=("flores_devtest_cyrl",))


def test_build_pool_concatenates_multiple_sources(monkeypatch):
    def fake_load_source(name, smoke=False, max_rows=None):
        return _fake_source(name, n=3)

    monkeypatch.setattr(prepare_data.dataset_io, "load_source", fake_load_source)
    combined, holdout = prepare_data.build_pool(sources=("opus100", "parallel_opus", "tatoeba"))
    assert holdout is None
    assert set(combined["source"]) == {"opus100", "parallel_opus", "tatoeba"}


def test_build_pool_returns_wiki_holdout(monkeypatch):
    def fake_load_source(name, smoke=False, max_rows=None):
        return _fake_source(name, n=3)

    def fake_load_wiki_split(holdout_size, seed, smoke, max_rows):
        train = _fake_source("wiki", n=10)
        holdout = _fake_source("wiki", n=holdout_size)
        return train, holdout

    monkeypatch.setattr(prepare_data.dataset_io, "load_source", fake_load_source)
    monkeypatch.setattr(prepare_data.dataset_io, "load_wiki_split", fake_load_wiki_split)
    combined, holdout = prepare_data.build_pool(
        sources=("opus100", "wiki"), wiki_holdout_size=4
    )
    assert holdout is not None
    assert len(holdout) == 4
    # Wiki train rows should be in the combined pool, not the holdout.
    assert "wiki" in set(combined["source"])
    assert "opus100" in set(combined["source"])


def test_build_pool_raises_if_all_sources_unavailable(monkeypatch):
    monkeypatch.setattr(prepare_data.dataset_io, "load_source", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="No sources"):
        prepare_data.build_pool(sources=("opus100",))


def test_build_pool_skips_missing_source_but_continues(monkeypatch):
    def fake_load_source(name, smoke=False, max_rows=None):
        if name == "tatoeba":
            return None
        return _fake_source(name, n=3)

    monkeypatch.setattr(prepare_data.dataset_io, "load_source", fake_load_source)
    combined, _ = prepare_data.build_pool(sources=("opus100", "tatoeba"))
    assert set(combined["source"]) == {"opus100"}
