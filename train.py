"""Train Uzbek sentence embedding via MultipleNegativesRankingLoss.

Loads a SentenceTransformer base model (granite-97m by default), the HF dataset
(default or smoke_100 config), runs MNRL contrastive training with the
SentenceTransformerTrainer, and saves the final checkpoint.

T4 specifics: attn_implementation='sdpa' (no FlashAttention-2 on sm_75) and
fp16=True/bf16=False. Push is intentionally manual: gated by post-train eval.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml
from pydantic import BaseModel

log = logging.getLogger("train")


class TrainConfig(BaseModel):
    base_model: str
    dataset_id: str
    dataset_config: str
    target_repo: str
    output_dir: str
    seed: int = 42
    max_seq_length: int = 256
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 2.0e-5
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2
    logging_steps: int = 50
    push: bool = False  # informational; push is always manual, gated by eval.py

    @classmethod
    def from_yaml(cls, path: str) -> TrainConfig:
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))


def _build_training_args(cfg: TrainConfig, has_validation: bool):
    from sentence_transformers import SentenceTransformerTrainingArguments
    from sentence_transformers.training_args import BatchSamplers

    common: dict = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        fp16=True,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=cfg.logging_steps,
        report_to="none",
        push_to_hub=False,
        dataloader_num_workers=2,
        seed=cfg.seed,
    )
    if has_validation:
        common.update(
            eval_strategy="steps",
            eval_steps=cfg.eval_steps,
            save_strategy="steps",
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    else:
        # smoke_100 has no validation split: skip in-loop eval, rely on final save_pretrained.
        common.update(eval_strategy="no", save_strategy="no")
    return SentenceTransformerTrainingArguments(**common)


def train(cfg: TrainConfig) -> None:
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from transformers import set_seed

    set_seed(cfg.seed)

    log.info("Loading base model: %s", cfg.base_model)
    model = SentenceTransformer(
        cfg.base_model,
        device="cuda",
        model_kwargs={"attn_implementation": "sdpa"},
    )
    model.max_seq_length = cfg.max_seq_length

    log.info("Loading dataset: %s/%s", cfg.dataset_id, cfg.dataset_config)
    ds = load_dataset(cfg.dataset_id, name=cfg.dataset_config)
    train_ds = ds["train"].select_columns(["anchor", "positive"])
    val_ds = ds["validation"].select_columns(["anchor", "positive"]) if "validation" in ds else None
    log.info(
        "Train rows=%d, validation=%s",
        len(train_ds),
        f"{len(val_ds)} rows" if val_ds is not None else "absent",
    )

    loss = MultipleNegativesRankingLoss(model, scale=20.0)
    args = _build_training_args(cfg, has_validation=val_ds is not None)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss=loss,
    )

    log.info("Training start.")
    trainer.train()
    log.info("Training done. Saving model to %s", cfg.output_dir)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(cfg.output_dir)
    log.info("Saved.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train an Uzbek sentence embedding model.")
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/smoke.yaml)")
    args = parser.parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
