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


class TestMixscriptAug:
    @staticmethod
    def _pool(n: int, anchor_lang: str = "uz_Latn", source: str = "parallel_opus") -> Dataset:
        return Dataset.from_list([
            {
                "anchor": f"Toshkent shahar markazi {i}",
                "positive": f"Tashkent city center {i}",
                "source": source,
                "anchor_lang": anchor_lang,
                "positive_lang": "en",
            }
            for i in range(n)
        ])

    def test_target_ratio_about_10pct(self):
        pool = self._pool(100)
        aug = prepare_data.augment_mixscript(pool, target_ratio=0.10, seed=42)
        assert 9 <= len(aug) <= 13

    def test_deterministic(self):
        pool = self._pool(100)
        aug1 = prepare_data.augment_mixscript(pool, target_ratio=0.10, seed=42)
        aug2 = prepare_data.augment_mixscript(pool, target_ratio=0.10, seed=42)
        assert list(aug1["anchor"]) == list(aug2["anchor"])
        assert list(aug1["positive"]) == list(aug2["positive"])

    def test_flips_script_and_tags_source(self):
        pool = self._pool(100)
        aug = prepare_data.augment_mixscript(pool, target_ratio=0.10, seed=42)
        assert len(aug) > 0
        for row in aug:
            assert row["source"] == "mixscript"
            assert row["anchor_lang"] != row["positive_lang"]
            assert row["anchor_lang"] in {"uz_Latn", "uz_Cyrl"}
            assert row["positive_lang"] in {"uz_Latn", "uz_Cyrl"}

    def test_excludes_non_uzbek_anchors(self):
        pool = self._pool(100, anchor_lang="en")
        aug = prepare_data.augment_mixscript(pool, target_ratio=0.10, seed=42)
        assert len(aug) == 0

    def test_handles_cyrillic_anchors(self):
        pool = Dataset.from_list([
            {
                "anchor": "Тошкент шаҳар",
                "positive": "Tashkent city",
                "source": "wiki",
                "anchor_lang": "uz_Cyrl",
                "positive_lang": "en",
            }
            for _ in range(100)
        ])
        aug = prepare_data.augment_mixscript(pool, target_ratio=0.10, seed=42)
        assert len(aug) > 0
        for row in aug:
            assert row["anchor_lang"] == "uz_Cyrl"
            assert row["positive_lang"] == "uz_Latn"


class TestBuildValidation:
    @staticmethod
    def _latn() -> Dataset:
        return Dataset.from_list([
            {
                "anchor": f"Toshkent shahar {i}",
                "positive": f"Tashkent city {i}",
                "source": "flores_dev_latn",
                "anchor_lang": "uz_Latn",
                "positive_lang": "en",
            }
            for i in range(100)
        ])

    @staticmethod
    def _cyrl() -> Dataset:
        return Dataset.from_list([
            {
                "anchor": f"Тошкент шаҳар {i}",
                "positive": f"Tashkent city {i}",
                "source": "flores_dev_cyrl",
                "anchor_lang": "uz_Cyrl",
                "positive_lang": "en",
            }
            for i in range(100)
        ])

    def test_native_cyrl_when_available(self, monkeypatch):
        latn = self._latn()
        cyrl = self._cyrl()

        def fake_load(name, **kwargs):
            return {"flores_dev_latn": latn, "flores_dev_cyrl": cyrl}.get(name)

        monkeypatch.setattr(prepare_data.dataset_io, "load_source", fake_load)
        val, cyrl_source = prepare_data.build_validation()

        assert cyrl_source == "native"
        sources = set(val["source"])
        assert "flores_dev_latn" in sources
        assert "flores_dev_cyrl" in sources
        assert "flores_dev_cyrl_translit" not in sources
        assert len(val) == 200

    def test_translit_fallback_when_cyrl_missing(self, monkeypatch):
        latn = self._latn()

        def fake_load(name, **kwargs):
            if name == "flores_dev_latn":
                return latn
            if name == "flores_dev_cyrl":
                return None
            raise KeyError(name)

        monkeypatch.setattr(prepare_data.dataset_io, "load_source", fake_load)
        val, cyrl_source = prepare_data.build_validation()

        assert cyrl_source == "transliterated"
        sources = set(val["source"])
        assert "flores_dev_latn" in sources
        assert "flores_dev_cyrl_translit" in sources
        assert "flores_dev_cyrl" not in sources
        translit_rows = val.filter(lambda r: r["source"] == "flores_dev_cyrl_translit")
        assert len(translit_rows) == 100
        for row in translit_rows:
            assert row["anchor_lang"] == "uz_Cyrl"

    def test_raises_when_latn_missing(self, monkeypatch):
        def fake_load(name, **kwargs):
            return None

        monkeypatch.setattr(prepare_data.dataset_io, "load_source", fake_load)
        with pytest.raises(RuntimeError, match="flores_dev_latn"):
            prepare_data.build_validation()
