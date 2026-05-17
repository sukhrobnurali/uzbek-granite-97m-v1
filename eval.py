"""Standalone evaluation for Uzbek sentence embedding models.

Three primary metrics gate the model push (need >= 2 wins vs base):
  1. Parallel-pair cosine: uz<->en margin on FLORES-200 devtest
  2. Retrieval Recall@1/5/10: held-out 5000 Uzbek Wiki title<->paragraph pairs
  3. Mix-script cosine: uz_Latn <-> uz_Cyrl invariance

Math lives in `_embeddings` functions (pure numpy + sklearn) so tests run without
sentence-transformers. The CLI / orchestrator functions import sentence-transformers
lazily and load FLORES + held-out Wiki at runtime.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

SCHEMA_VERSION = 1
PRIMARY_METRIC_PATHS = (
    ("parallel_cosine", "mean_cosine_parallel"),
    ("retrieval", "recall@1"),
    ("mixscript", "mean_cosine_mixscript"),
)


def metric_parallel_cosine_embeddings(
    uz_emb: np.ndarray,
    en_emb: np.ndarray,
    random_seed: int = 42,
) -> dict:
    """Cosine of (uz_i, en_i) vs (uz_i, en_pi(i)) where pi is a derangement."""
    if uz_emb.shape != en_emb.shape:
        raise ValueError(f"Embedding shape mismatch: {uz_emb.shape} vs {en_emb.shape}")
    n = len(uz_emb)
    if n < 2:
        raise ValueError("Need at least 2 pairs to compute random baseline")
    parallel = (uz_emb * en_emb).sum(axis=1)
    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(n)
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    random_en = en_emb[perm]
    random_cos = (uz_emb * random_en).sum(axis=1)
    return {
        "mean_cosine_parallel": float(parallel.mean()),
        "mean_cosine_random": float(random_cos.mean()),
        "margin": float(parallel.mean() - random_cos.mean()),
        "n": int(n),
    }


def metric_retrieval_embeddings(
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    gold_indices: np.ndarray | None = None,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Top-K cosine retrieval. gold_indices[i] = index in corpus that matches query i.
    If gold_indices is None, assume identity (corpus[i] is the gold for query[i])."""
    n = len(query_emb)
    if gold_indices is None:
        gold_indices = np.arange(n)
    k_max = max(ks)
    if k_max > len(corpus_emb):
        raise ValueError(f"k={k_max} exceeds corpus size {len(corpus_emb)}")
    nn = NearestNeighbors(n_neighbors=k_max, metric="cosine")
    nn.fit(corpus_emb)
    _, indices = nn.kneighbors(query_emb)
    out: dict[str, Any] = {"n": int(n)}
    for k in ks:
        hits = sum(1 for i in range(n) if gold_indices[i] in indices[i, :k])
        out[f"recall@{k}"] = float(hits / n)
    return out


def metric_mixscript_embeddings(
    latn_emb: np.ndarray,
    cyrl_emb: np.ndarray,
    cyrl_source: str = "transliterated",
) -> dict:
    if latn_emb.shape != cyrl_emb.shape:
        raise ValueError(f"Shape mismatch: {latn_emb.shape} vs {cyrl_emb.shape}")
    cos = (latn_emb * cyrl_emb).sum(axis=1)
    return {
        "mean_cosine_mixscript": float(cos.mean()),
        "pct_above_0.9": float((cos > 0.9).mean()),
        "n": len(cos),
        "cyrl_source": cyrl_source,
    }


def smoke_gate(base_metrics: dict, tuned_metrics: dict) -> tuple[bool, dict]:
    """True iff tuned beats base on >= 2 of 3 primary metrics."""
    wins = 0
    deltas: dict[str, float] = {}
    for category, metric in PRIMARY_METRIC_PATHS:
        base_v = base_metrics[category][metric]
        tuned_v = tuned_metrics[category][metric]
        delta = tuned_v - base_v
        deltas[f"{category}.{metric}"] = delta
        if delta > 0:
            wins += 1
    return wins >= 2, {"wins": wins, "deltas": deltas, "primary_paths": [".".join(p) for p in PRIMARY_METRIC_PATHS]}


def _normalize(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return emb / norms


def _encode(model, sentences: list[str], batch_size: int = 128) -> np.ndarray:
    emb = model.encode(
        sentences,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype=np.float32)


def metric_parallel_cosine(model, uz_sentences: list[str], en_sentences: list[str], random_seed: int = 42) -> dict:
    uz_emb = _encode(model, uz_sentences)
    en_emb = _encode(model, en_sentences)
    return metric_parallel_cosine_embeddings(uz_emb, en_emb, random_seed=random_seed)


def metric_retrieval(model, titles: list[str], paragraphs: list[str]) -> dict:
    query_emb = _encode(model, titles)
    corpus_emb = _encode(model, paragraphs)
    return metric_retrieval_embeddings(query_emb, corpus_emb)


def metric_mixscript(
    model,
    sentences_latn: list[str],
    sentences_cyrl: list[str],
    cyrl_source: str = "transliterated",
) -> dict:
    latn_emb = _encode(model, sentences_latn)
    cyrl_emb = _encode(model, sentences_cyrl)
    return metric_mixscript_embeddings(latn_emb, cyrl_emb, cyrl_source=cyrl_source)


def load_flores_devtest_pairs() -> tuple[list[str], list[str]]:
    """Returns (uzbek_Latn_sentences, english_sentences) from FLORES+ devtest (1012 rows)."""
    from utils.dataset_io import load_flores_plus

    ds = load_flores_plus("devtest", "eval_flores_devtest")
    return list(ds["anchor"]), list(ds["positive"])


def load_flores_devtest_cyrl(uz_latn_sentences: list[str]) -> tuple[list[str], str]:
    """FLORES+ has no uzn_Cyrl variant, so always transliterate from Latn.
    Tagged 'transliterated' in the JSON for honest reporting."""
    from utils.translit import to_cyrillic

    return [to_cyrillic(s) for s in uz_latn_sentences], "transliterated"


def load_wiki_retrieval_eval(dataset_id: str = "sukhrobnurali/uzbek-embedding-pairs") -> tuple[list[str], list[str]]:
    """Loads the held-out 5000 Wiki title<->paragraph pairs from the dataset's
    wiki_retrieval_eval config. Returns (titles, paragraphs)."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, name="wiki_retrieval_eval", split="test")
    return list(ds["anchor"]), list(ds["positive"])


def _compute_comparison(metrics: dict, baseline: dict) -> dict:
    base_metrics = baseline["metrics"]
    won, gate_info = smoke_gate(base_metrics, metrics)
    return {
        "baseline_model_id": baseline.get("model_id"),
        "wins": gate_info["wins"],
        "deltas": gate_info["deltas"],
        "passed_gate": won,
        "primary_paths": gate_info["primary_paths"],
    }


def run_full_eval(
    model_id_or_path: str,
    output_path: str,
    baseline_path: str | None = None,
    device: str | None = None,
    smoke: bool = False,
    dataset_id: str = "sukhrobnurali/uzbek-embedding-pairs",
) -> dict:
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = os.environ.get("DEVICE") or _detect_device()

    model = SentenceTransformer(model_id_or_path, device=device)

    uz_latn, en_latn = load_flores_devtest_pairs()
    if smoke:
        uz_latn = uz_latn[:50]
        en_latn = en_latn[:50]

    parallel = metric_parallel_cosine(model, uz_latn, en_latn)

    titles, paragraphs = load_wiki_retrieval_eval(dataset_id)
    if smoke:
        titles = titles[:50]
        paragraphs = paragraphs[:50]
    retrieval = metric_retrieval(model, titles, paragraphs)

    uz_cyrl, cyrl_source = load_flores_devtest_cyrl(uz_latn)
    mixscript = metric_mixscript(model, uz_latn, uz_cyrl, cyrl_source=cyrl_source)

    report = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id_or_path,
        "model_sha": _model_sha(model),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "smoke": smoke,
        "metrics": {
            "parallel_cosine": parallel,
            "retrieval": retrieval,
            "mixscript": mixscript,
        },
    }

    if baseline_path and Path(baseline_path).exists():
        with open(baseline_path, encoding="utf-8") as f:
            baseline = json.load(f)
        report["comparison_to_baseline"] = _compute_comparison(report["metrics"], baseline)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def _model_sha(model) -> str:
    try:
        config = model[0].auto_model.config
        return getattr(config, "_commit_hash", "") or ""
    except Exception:
        return ""


def _detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an Uzbek sentence embedding model.")
    parser.add_argument("--model", required=True, help="HF model ID or local checkpoint path")
    parser.add_argument("--output", required=True, help="Path to write the JSON report")
    parser.add_argument("--baseline", default=None, help="Optional baseline JSON to compare against")
    parser.add_argument("--device", default=None, help="cuda:0 / cpu (auto-detected if unset)")
    parser.add_argument("--smoke", action="store_true", help="Use 50-sample slices for a fast sanity run")
    parser.add_argument(
        "--dataset",
        default="sukhrobnurali/uzbek-embedding-pairs",
        help="HF dataset for the held-out Wiki retrieval eval",
    )
    args = parser.parse_args()
    report = run_full_eval(
        model_id_or_path=args.model,
        output_path=args.output,
        baseline_path=args.baseline,
        device=args.device,
        smoke=args.smoke,
        dataset_id=args.dataset,
    )
    print(json.dumps(report.get("comparison_to_baseline") or report["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
