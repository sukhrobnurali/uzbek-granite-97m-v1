# uzbek-embeddings

Fine-tuned Uzbek sentence embedding model — `sukhrobnurali/uzbek-granite-97m-v1`.

The first strong open Uzbek embedding model on Hugging Face. Built on top of
[`ibm-granite/granite-embedding-97m-multilingual-r2`](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2)
with parallel data from OPUS-100, Uzbek Wikipedia, FLORES-200, and Tatoeba,
plus explicit Latin↔Cyrillic script-augmentation.

## Artifacts

- **Model:** https://huggingface.co/sukhrobnurali/uzbek-granite-97m-v1
- **Dataset:** https://huggingface.co/datasets/sukhrobnurali/uzbek-embedding-pairs
- **Demo (Gradio Space):** https://huggingface.co/spaces/sukhrobnurali/uzbek-embedding-demo
- **Base model:** [`ibm-granite/granite-embedding-97m-multilingual-r2`](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2)

## Results

Three primary metrics (★) define the 2/3 win gate vs. the base model. Filled in after the full training run.

| Metric | Base | Tuned | Δ | Win? |
|---|---|---|---|---|
| ★ Parallel mean cos (uz↔en, FLORES devtest) |  |  |  |  |
| Parallel margin (parallel − random) |  |  |  | — |
| ★ Retrieval Recall@1 (Wiki 5K) |  |  |  |  |
| Retrieval Recall@5 |  |  |  | — |
| Retrieval Recall@10 |  |  |  | — |
| ★ Mix-script cos (uz_Latn ↔ uz_Cyrl) |  |  |  |  |
| Mix-script % above 0.9 |  |  |  | — |
| **Wins (need ≥2/3)** | — | — | — | **/3** |

## Quickstart

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sukhrobnurali/uzbek-granite-97m-v1")
embeddings = model.encode(
    ["Toshkent — Oʻzbekistonning poytaxti.", "Тошкент — Ўзбекистоннинг пойтахти."],
    normalize_embeddings=True,
)
# cosine(embeddings[0], embeddings[1]) > 0.9   # script-invariant
```

## Reproducing the training

```
pip install -r requirements.txt

# 1. Eval comes first
python eval.py --model ibm-granite/granite-embedding-97m-multilingual-r2 \
               --output results/base.json --smoke

# 2. Build & push dataset
python prepare_data.py --push

# 3. Smoke test (100 samples, pipeline integrity check)
python smoke_test.py

# 4. Full training
python train.py --config configs/train.yaml

# 5. Full eval — gates the push (need 2/3 wins vs base)
python eval.py --model checkpoints/full --baseline results/base.json \
               --output results/tuned.json

# 6. (Optional) Hard-negative re-train
python mine_hard_negatives.py
python train.py --config configs/train_hardneg.yaml
```

## Training environment

- Single T4 (Colab free tier, 16GB, sm_75)
- fp16 (T4 has no bf16), SDPA attention (T4 has no FlashAttention-2)
- `sentence-transformers` v5.4+ with `SentenceTransformerTrainer`
- See [`configs/train.yaml`](configs/train.yaml) for hyperparameters

## Limitations

- Script detection is a simple character-class heuristic — mixed-script sentences are tagged `mixed` but not split.
- Wikipedia title↔paragraph pairs are noisy by construction (stubs, disambiguation pages filtered but not perfect).
- OPUS-100 `en-uz` is web-scraped (license "unknown"); pair quality varies. Length-ratio and word-count filters mitigate but don't eliminate alignment errors.
- `uzn_Cyrl-eng_Latn` config of FLORES-200 may not be public — mix-script eval falls back to transliterating `uzn_Latn` and tags `"cyrl_source": "transliterated"` in the results JSON.

## License

Apache-2.0. The processed dataset (`sukhrobnurali/uzbek-embedding-pairs`) is redistributed under Apache-2.0 with source-license caveats documented in its dataset card.
