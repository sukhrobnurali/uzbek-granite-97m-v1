"""HF Hub publishing orchestrator for uzbek-embedding-pairs.

Split out of prepare_data.py to keep the data-prep module under the size threshold.
The two-flag gate (`confirm` + interactive 'yes' + `allow_overwrite`) makes the
irreversible network call hard to fire accidentally.
"""

from __future__ import annotations

import os

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from huggingface_hub.repocard import DatasetCard

from utils.dataset_card import build_card
from utils.logging_setup import configure

log = configure()


def _print_preflight(api: HfApi, repo_id: str, stats: dict, card: str) -> None:
    user = api.whoami().get("name", "<unknown>")
    log.info("=== HF PUSH PRE-FLIGHT ===")
    log.info("Repo:          https://huggingface.co/datasets/%s", repo_id)
    log.info("Token user:    %s", user)
    log.info("Configs:")
    log.info("  default              train=%s  validation=%s",
             f"{stats['train_rows']:,}", f"{stats['validation_rows']:,}")
    log.info("  wiki_retrieval_eval  test=%s", f"{stats['retrieval_rows']:,}")
    log.info("  smoke_100            train=%s", f"{stats['smoke_rows']:,}")
    log.info("Source distribution: %s", stats["source_distribution"])
    log.info("Mixscript ratio:     %.1f%%", stats["mixscript_ratio"] * 100)
    log.info("Validation cyrl:     %s", stats["validation_cyrl_source"])
    log.info("--- card preview (first 30 lines) ---")
    for line in card.splitlines()[:30]:
        log.info(line)
    log.info("--- card preview (last 10 lines) ---")
    for line in card.splitlines()[-10:]:
        log.info(line)
    log.info("=== END PRE-FLIGHT ===")


def _perform_pushes(
    train: Dataset,
    validation: Dataset,
    retrieval: Dataset,
    smoke: Dataset,
    repo_id: str,
) -> None:
    log.info("Pushing config 'default' ...")
    DatasetDict({"train": train, "validation": validation}).push_to_hub(
        repo_id, config_name="default"
    )
    log.info("Pushing config 'wiki_retrieval_eval' ...")
    DatasetDict({"test": retrieval}).push_to_hub(
        repo_id, config_name="wiki_retrieval_eval"
    )
    log.info("Pushing config 'smoke_100' ...")
    DatasetDict({"train": smoke}).push_to_hub(
        repo_id, config_name="smoke_100"
    )


def _push_card(api: HfApi, repo_id: str, card: str) -> None:
    log.info("Pushing dataset card ...")
    DatasetCard(card).push_to_hub(repo_id, repo_type="dataset")


def push_to_hf(
    train: Dataset,
    validation: Dataset,
    retrieval: Dataset,
    smoke: Dataset,
    repo_id: str,
    stats: dict,
    confirm: bool = False,
    allow_overwrite: bool = False,
) -> None:
    """Push the dataset to HF after pre-flight checks.

    Calls SystemExit on any abort path so the caller can be safely scripted.
    """
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set in the environment; refusing to push.")

    api = HfApi()
    exists = api.repo_exists(repo_id, repo_type="dataset")
    if exists and not allow_overwrite:
        raise SystemExit(
            f"Dataset repo {repo_id} already exists; pass --allow-overwrite to push anyway."
        )
    if not exists:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=False, private=False)
        log.info("Created dataset repo %s", repo_id)

    card = build_card(stats)
    _print_preflight(api, repo_id, stats, card)

    if not confirm:
        response = input("Type 'yes' to push: ").strip().lower()
        if response != "yes":
            raise SystemExit("Push aborted by user.")

    _perform_pushes(train, validation, retrieval, smoke, repo_id)
    _push_card(api, repo_id, card)
    log.info("Done. https://huggingface.co/datasets/%s", repo_id)
