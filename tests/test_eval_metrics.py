import numpy as np
import pytest

from eval import (
    PRIMARY_METRIC_PATHS,
    SCHEMA_VERSION,
    _normalize,
    metric_mixscript_embeddings,
    metric_parallel_cosine_embeddings,
    metric_retrieval_embeddings,
    smoke_gate,
)


def _make_aligned_embeddings(n: int, dim: int = 8, jitter: float = 0.05, seed: int = 0):
    """Build two embedding matrices A, B where B[i] is a slight perturbation of A[i].
    Both are L2-normalized so cosine == dot product."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, dim)).astype(np.float32)
    noise = rng.standard_normal((n, dim)).astype(np.float32) * jitter
    b = a + noise
    return _normalize(a), _normalize(b)


def _make_orthogonal_pair(n: int, dim: int = 8, seed: int = 0):
    """A, B where A[i] and B[i] are pairwise unrelated random vectors."""
    rng = np.random.default_rng(seed)
    a = _normalize(rng.standard_normal((n, dim)).astype(np.float32))
    b = _normalize(rng.standard_normal((n + 1, dim)).astype(np.float32))[1:]
    return a, b


def test_metric_parallel_cosine_returns_high_margin_for_aligned():
    a, b = _make_aligned_embeddings(n=20, jitter=0.05)
    out = metric_parallel_cosine_embeddings(a, b)
    assert set(out.keys()) == {"mean_cosine_parallel", "mean_cosine_random", "margin", "n"}
    assert out["n"] == 20
    assert out["mean_cosine_parallel"] > 0.9
    assert out["mean_cosine_random"] < 0.5
    assert out["margin"] > 0.5


def test_metric_parallel_cosine_low_margin_for_orthogonal():
    a, b = _make_orthogonal_pair(n=20)
    out = metric_parallel_cosine_embeddings(a, b)
    assert abs(out["mean_cosine_parallel"]) < 0.3
    assert abs(out["margin"]) < 0.4


def test_metric_parallel_cosine_random_perm_excludes_self():
    """The random baseline must never compare (uz_i, en_i) — that would be identical
    to the parallel signal and inflate the baseline."""
    n = 100
    a, _ = _make_aligned_embeddings(n=n, jitter=0.0)  # b == a after normalization
    # Run with a seeded rng; the implementation must derange to avoid self-pairs.
    # If self-pairs leaked through, mean_cosine_random would equal mean_cosine_parallel.
    a, b = _make_aligned_embeddings(n=n, jitter=0.0)
    out = metric_parallel_cosine_embeddings(a, b)
    # With identical a and b, parallel == 1.0 exactly; random must NOT equal 1.0.
    assert out["mean_cosine_parallel"] > 0.999
    assert out["mean_cosine_random"] < 0.5


def test_metric_parallel_cosine_shape_mismatch_raises():
    a = np.zeros((5, 8), dtype=np.float32)
    b = np.zeros((6, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        metric_parallel_cosine_embeddings(a, b)


def test_metric_parallel_cosine_too_few_samples():
    a = np.zeros((1, 8), dtype=np.float32)
    b = np.zeros((1, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        metric_parallel_cosine_embeddings(a, b)


def test_metric_retrieval_perfect_recall_for_identity():
    n = 50
    a, _ = _make_aligned_embeddings(n=n, jitter=0.0)
    out = metric_retrieval_embeddings(a, a)
    assert out["n"] == 50
    assert out["recall@1"] == 1.0
    assert out["recall@5"] == 1.0
    assert out["recall@10"] == 1.0


def test_metric_retrieval_recall_with_close_pairs():
    n = 30
    queries, corpus = _make_aligned_embeddings(n=n, jitter=0.1)
    out = metric_retrieval_embeddings(queries, corpus)
    assert 0.5 <= out["recall@1"] <= 1.0
    assert out["recall@5"] >= out["recall@1"]
    assert out["recall@10"] >= out["recall@5"]


def test_metric_retrieval_respects_gold_indices():
    a, _ = _make_aligned_embeddings(n=10, jitter=0.0)
    out = metric_retrieval_embeddings(
        a, a, gold_indices=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]), ks=(1,)
    )
    assert out["recall@1"] == 0.0


def test_metric_retrieval_k_exceeds_corpus_raises():
    a = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError):
        metric_retrieval_embeddings(a, a, ks=(1, 5))


def test_metric_mixscript_high_cos_when_aligned():
    a, b = _make_aligned_embeddings(n=20, jitter=0.02)
    out = metric_mixscript_embeddings(a, b)
    assert set(out.keys()) == {"mean_cosine_mixscript", "pct_above_0.9", "n", "cyrl_source"}
    assert out["n"] == 20
    assert out["mean_cosine_mixscript"] > 0.9
    assert out["pct_above_0.9"] > 0.5
    assert out["cyrl_source"] == "transliterated"


def test_metric_mixscript_low_cos_when_orthogonal():
    a, b = _make_orthogonal_pair(n=20)
    out = metric_mixscript_embeddings(a, b)
    assert out["mean_cosine_mixscript"] < 0.5
    assert out["pct_above_0.9"] < 0.2


def test_metric_mixscript_cyrl_source_passthrough():
    a, b = _make_aligned_embeddings(n=10)
    assert metric_mixscript_embeddings(a, b, cyrl_source="native")["cyrl_source"] == "native"
    assert metric_mixscript_embeddings(a, b, cyrl_source="transliterated")["cyrl_source"] == "transliterated"


def test_smoke_gate_wins_2_of_3_passes():
    base = {
        "parallel_cosine": {"mean_cosine_parallel": 0.5},
        "retrieval": {"recall@1": 0.4},
        "mixscript": {"mean_cosine_mixscript": 0.6},
    }
    tuned = {
        "parallel_cosine": {"mean_cosine_parallel": 0.6},   # +0.1 win
        "retrieval": {"recall@1": 0.5},                       # +0.1 win
        "mixscript": {"mean_cosine_mixscript": 0.55},         # -0.05 loss
    }
    passed, info = smoke_gate(base, tuned)
    assert passed is True
    assert info["wins"] == 2
    assert info["deltas"]["parallel_cosine.mean_cosine_parallel"] == pytest.approx(0.1)
    assert info["deltas"]["retrieval.recall@1"] == pytest.approx(0.1)
    assert info["deltas"]["mixscript.mean_cosine_mixscript"] == pytest.approx(-0.05)


def test_smoke_gate_wins_1_of_3_fails():
    base = {
        "parallel_cosine": {"mean_cosine_parallel": 0.5},
        "retrieval": {"recall@1": 0.4},
        "mixscript": {"mean_cosine_mixscript": 0.6},
    }
    tuned = {
        "parallel_cosine": {"mean_cosine_parallel": 0.4},   # loss
        "retrieval": {"recall@1": 0.5},                       # win
        "mixscript": {"mean_cosine_mixscript": 0.55},         # loss
    }
    passed, info = smoke_gate(base, tuned)
    assert passed is False
    assert info["wins"] == 1


def test_smoke_gate_wins_3_of_3_passes():
    base = {
        "parallel_cosine": {"mean_cosine_parallel": 0.5},
        "retrieval": {"recall@1": 0.4},
        "mixscript": {"mean_cosine_mixscript": 0.6},
    }
    tuned = {
        "parallel_cosine": {"mean_cosine_parallel": 0.7},
        "retrieval": {"recall@1": 0.6},
        "mixscript": {"mean_cosine_mixscript": 0.8},
    }
    passed, info = smoke_gate(base, tuned)
    assert passed is True
    assert info["wins"] == 3


def test_smoke_gate_ties_count_as_losses():
    base = {
        "parallel_cosine": {"mean_cosine_parallel": 0.5},
        "retrieval": {"recall@1": 0.4},
        "mixscript": {"mean_cosine_mixscript": 0.6},
    }
    tuned = {
        "parallel_cosine": {"mean_cosine_parallel": 0.5},
        "retrieval": {"recall@1": 0.4},
        "mixscript": {"mean_cosine_mixscript": 0.7},
    }
    passed, info = smoke_gate(base, tuned)
    assert passed is False  # 1 win + 2 ties; ties are NOT wins
    assert info["wins"] == 1


def test_primary_metric_paths_unchanged():
    """Pin the 3 primary metrics — if anyone changes these, the gate semantics shift."""
    assert PRIMARY_METRIC_PATHS == (
        ("parallel_cosine", "mean_cosine_parallel"),
        ("retrieval", "recall@1"),
        ("mixscript", "mean_cosine_mixscript"),
    )


def test_schema_version_pinned():
    assert SCHEMA_VERSION == 1


def test_normalize_helper_produces_unit_norm():
    a = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    out = _normalize(a)
    assert out.shape == a.shape
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, [1.0, 0.0, 1.0])
