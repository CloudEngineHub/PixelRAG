# CLAUDE.md

> ⚠️ **Dev notes may be out of date.** `docs/training_dev_notes.md` (and some
> references in this file) describe historical internals and may have drifted
> from the current code — e.g. they mention `train_colpali.py` and an old
> `training.*` package/trainer stack (`train.py`, `evaluate.py`, `dataset.py`,
> `model.py`) that are superseded by the self-contained `train_contrastors.py`.
> Treat the docs as background; trust the code and `README.md` as source of truth.

## Pinned Versions

All training and eval **must** use these exact versions — mismatches cause silent numerical divergence:

| Package | Version |
|---------|---------|
| PyTorch | **2.9.1+cu129** |
| cuDNN | **92000** |
| transformers | **4.57.1** |

`uv sync` will install the correct versions from `pyproject.toml` + lockfile.
cuDNN 9.20 is forced via `override-dependencies` in `pyproject.toml` (torch 2.9.1 ships with 9.10, but we need 9.20 for native bf16 Conv3d).

**Always use `uv run` to ensure the locked environment is used.**

## Training Pipeline (Best Config)

```bash
# 0. Install
uv sync

# 1. Mine hard negatives (requires search API at localhost:30888)
uv run python mine_hard_negatives.py \
    --input training/data/train.jsonl \
    --output training/data/train_hn.jsonl \
    --num-negatives 7 --n-docs 50 \
    --filter-mode margin --margin 0.95

# 2. Train (best config)
CUDA_VISIBLE_DEVICES=1,2 uv run torchrun --nproc_per_node=2 train_contrastors.py \
    --data-split-dir training/data/lite-query-v2-full-filtered-hn-v2-chunks/split \
    --max-steps 50 \
    --batch-size 16 \
    --grad-cache-chunk 4 \
    --num-hard-negatives 5 \
    --lr 1e-5 \
    --warmup-steps 20 \
    --test-eval-steps 50 \
    --test-max-pairs 0 \
    --eval-steps 25 \
    --save-steps 50 \
    --output-dir training/output_nvme/output_hn_best

# 3. Eval checkpoint on hard-mini-v6 (default) or v7
cd /home/user/Vis-RAG/agent && \
CUDA_VISIBLE_DEVICES=4 uv run python scripts/eval_lora_checkpoint.py \
    /home/user/wiki-screenshot-training/training/output_nvme/output_hn_best/checkpoint-50 \
    --tiles-dir tiles-hard-mini-v6

# Eval on hard-mini-v7 (400 queries, 7723 tiles — more comprehensive)
# ⚠️ v7 has known issues — prefer v6 for now
cd /home/user/Vis-RAG/agent && \
CUDA_VISIBLE_DEVICES=4 uv run python scripts/eval_lora_checkpoint.py \
    /home/user/wiki-screenshot-training/training/output_nvme/output_hn_best/checkpoint-50 \
    --tiles-dir tiles-hard-mini-v7 \
    --vllm-url http://localhost:8201/v1 \
    --vllm-model Qwen/Qwen3-VL-4B-Instruct

# Retrieval-only (skip QA, faster)
cd /home/user/Vis-RAG/agent && \
CUDA_VISIBLE_DEVICES=4 uv run python scripts/eval_lora_checkpoint.py \
    /path/to/checkpoint --tiles-dir tiles-hard-mini-v7 --retrieval-only

# Resume from checkpoint
uv run python train_contrastors.py --resume training/output_nvme/output_hn_best/checkpoint-50
```

## Query-Side Tune

`train_contrastors.py` also supports a query-only fine-tuning mode:

- `--mode query-side-tune` trains the query tower only
- the doc/image tower stays frozen at the base model
- datastore embeddings therefore stay valid across checkpoints
- checkpoint eval can query the full external datastore directly via `wiki-screenshot` search API

Recommended command:

```bash
OPENAI_API_KEY=<your-key> \
OPENAI_BASE_URL=https://us.api.openai.com/v1 \
CUDA_VISIBLE_DEVICES=2,3,4,5 uv run torchrun --master_port 29531 --nproc_per_node=4 train_contrastors.py \
    --mode query-side-tune \
    --query-side-backward direct \
    --data-split-dir training/data/lite-query-v2-full-filtered-hn-v2-chunks/split \
    --batch-size 64 \
    --num-hard-negatives 2 \
    --lr 2e-6 \
    --warmup-steps 20 \
    --eval-steps 25 \
    --test-eval-steps 50 \
    --save-steps 100 \
    --max-steps 100 \
    --max-num-visual-tokens 1024 \
    --search-api-url http://localhost:30888 \
    --qa-eval-max-examples 100 \
    --vllm-url http://localhost:8201/v1 \
    --vllm-model Qwen/Qwen3-VL-4B-Instruct \
    --output-dir training/output_nvme/output_query_side
```

Required services:
- **search API** on `:30888` — for retrieval eval (recall@1/3)
- **vLLM** on `:8201` — Qwen3-VL-4B-Instruct for VQA answering
- **OpenAI API** — GPT-4.1 as QA grader (needs `OPENAI_API_KEY`)

Important defaults / caveats:

- In `query-side-tune`, default `--query-side-backward` is `direct` (no GradCache replay).
- This is intentional: `direct` was stable in 4-GPU real-data smoke tests.
- `--query-side-backward gradcache` is currently experimental for query-side mode and may hang.
- `standard` mode is unchanged and still uses the original GradCache path.
- Query-side retrieval eval requires a **new enough** `wiki-screenshot` `search_api.py` that accepts pre-computed `embedding` queries.

## Training Scripts

| Script | Description |
|--------|-------------|
| `train_contrastors.py` | **Primary.** Standard mode uses GradCache; query-side mode freezes the doc tower and defaults to direct backward. |
| `train_swift.py` | ms-swift alternative. Simpler (one `sft_main()` call) but no GradCache / learnable temp. |
| `train_colpali.py` | Legacy HF Trainer-based. Simpler but no GradCache. |

Key args for `train_contrastors.py`:

| Arg | Default | Notes |
|-----|---------|-------|
| `--mode` | `standard` | `standard` = shared tower training; `query-side-tune` = train query side only |
| `--query-side-backward` | `direct` | Only used in `query-side-tune`; `direct` is stable default |
| `--batch-size` | 4 | Per-GPU. Effective negatives = batch × num_gpus |
| `--grad-cache-chunk` | 2 | Memory ∝ this, not batch size |
| `--lr` | 2e-5 | Cosine schedule with warmup |
| `--warmup-steps` | 50 | Linear warmup |
| `--temperature` | 0.07 | Initial temp (learnable via LogitScale) |
| `--num-hard-negatives` | 0 | Set 2–7 when using `train_hn.jsonl` |
| `--lora-r` / `--lora-alpha` | 32 / 32 | LoRA rank and alpha |
| `--lora-vit` | off | Also apply LoRA to ViT vision encoder (see below) |
| `--max-num-visual-tokens` | 1024 | Image resolution control |
| `--max-steps` | 500 | Total training steps |
| `--eval-steps` | 100 | Validation loss frequency |
| `--test-eval-steps` | 250 | Full retrieval eval (R@1/5/10, MRR) |
| `--text-warmup-steps` | 0 | Text-only warmup steps before image training |
| `--text-data-dir` | None | Directory with text-qa-pair JSONL files |
| `--text-mix-ratio` | 0 | Fraction of text batches during image phase |
| `--text-curriculum` | off | Gradual text→image transition (50%→33%→20%→0%) |
| `--test-batch-size` | 16 | Eval batch size (lower to avoid OOM on v7/v8) |
| `--hardness-alpha` | 0 | LLaVE hardness weighting (0=off, try 5–9). Upweights harder negatives in softmax |

### LoRA Target Modules

By default, LoRA only targets the **LLM backbone** attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
The ViT vision encoder is **not** tuned — its attention uses fused `qkv` naming that doesn't match.

Qwen3-VL architecture:
```
ViT (frozen by default):
  model.visual.blocks.N.attn.qkv        — fused QKV (Linear)
  model.visual.blocks.N.attn.proj        — output proj (Linear)
  model.visual.blocks.N.mlp.linear_fc1   — MLP up (Linear)
  model.visual.blocks.N.mlp.linear_fc2   — MLP down (Linear)

Merger (vision→LLM bridge):
  model.visual.merger.linear_fc1/fc2     — not matched by --lora-vit

LLM backbone (tuned by default):
  model.layers.N.self_attn.q_proj/k_proj/v_proj/o_proj
```

With `--lora-vit`, LoRA is also applied to ViT attn + MLP layers:
- Without: 224 LoRA layers, ~12.8M trainable params
- With `--lora-vit`: 416 LoRA layers, ~25M trainable params

This matches the colpali/BiQwen2 approach which uses `.*model.*` regex to target all layers including ViT.

Query-side-specific eval behavior:

- `test split`: local query embeddings are sent to `--search-api-url`, retrieve top-3, report `recall@1` / `recall@3`
- `QA eval`: current query tower retrieves top-3 from the same endpoint, reports article-level `recall@1` / `recall@3`, then grades answers with an OpenAI-compatible judge

## Data

### Source: LLM-Augmented Query-Document Pairs

Training pairs are generated by sending Wikipedia screenshot chunks to LLMs, which produce
natural-language queries that a user might ask to find that specific chunk.

Batch JSONL files live in `agent/scripts/contrastive/batches/`:

| Batch range | Model | Pairs | Notes |
|-------------|-------|-------|-------|
| `batch_000.jsonl` – `batch_057.jsonl` | Gemini 3.1 Pro | 40,402 | Higher quality, more diverse queries |
| `batch_200.jsonl` – `batch_257.jsonl` | Flash-Lite | 88,418 | Cheaper, noisier, larger volume |

Total: ~128,820 augmented pairs across 116 batch files.

### Data Format

Basic format (one JSON object per line):
```json
{"query": "What is the population of Tokyo?", "chunk_path": "/opt/dlami/nvme/kiwix_tiles/shard_000/shard_00042/350170.png.tiles/chunk_0000_00.png"}
```

Hard negative format (after mining):
```json
{"query": "...", "chunk_path": "...", "neg_chunk_paths": ["/path/to/neg1.png", "/path/to/neg2.png"]}
```

### Train / Val / Test Split

Split the combined batch data before training:

1. **Concatenate** all batch JSONL files into one pool
2. **Shuffle** with a fixed seed for reproducibility
3. **Split** by ratio:
   - **Train**: 90% — used for contrastive fine-tuning
   - **Val** (`eval.jsonl`): 5% — monitored every `--eval-steps` for loss
   - **Test** (`test.jsonl`): 5% — full retrieval eval (R@1, R@5, R@10, MRR) every `--test-eval-steps`
4. **Mine hard negatives** on the train split only:
   ```bash
   uv run python mine_hard_negatives.py \
       --input training/data/train.jsonl \
       --output training/data/train_hn.jsonl \
       --num-negatives 7 --n-docs 20 --margin 0.95
   ```

Output files in `training/data/`:
- `train.jsonl` / `train_hn.jsonl` — training (with optional hard negs)
- `eval.jsonl` — validation
- `test.jsonl` — held-out test

### External Training Datasets (MOCA / MADQA)

Additional contrastive training data from MOCA and MADQA benchmarks, with pre-mined hard negatives:

| Dataset | HF Link | Rows | Images | Upload Status |
|---------|---------|------|--------|---------------|
| MOCA ColPali | [Chrisyichuan/moca-colpali-training](https://huggingface.co/datasets/Chrisyichuan/moca-colpali-training) | 118,195 | 118,195 | ✅ done |
| MOCA PixelRAG Ind | [Chrisyichuan/moca-visrag-ind-training](https://huggingface.co/datasets/Chrisyichuan/moca-visrag-ind-training) | 122,752 | 122,752 | ✅ done |
| MOCA PixelRAG Syn | [Chrisyichuan/moca-visrag-syn-training](https://huggingface.co/datasets/Chrisyichuan/moca-visrag-syn-training) | 239,206 | 239,298 | ✅ done |
| MADQA | [Chrisyichuan/madqa-training](https://huggingface.co/datasets/Chrisyichuan/madqa-training) | 1,840 | 3,598 | ✅ done |

Images are stored as **tar shards** under `image_shards/`. After downloading, extract with the included script:

```bash
# Download all four
pip install huggingface_hub
for repo in moca-colpali-training moca-visrag-ind-training moca-visrag-syn-training madqa-training; do
    huggingface-cli download Chrisyichuan/$repo --repo-type dataset --local-dir /opt/dlami/nvme/external_data/$repo
done

# Extract images from tar shards
cd /opt/dlami/nvme/external_data/<dataset>
python extract_hf_image_shards.py --dataset-dir .
```

Each dataset contains a JSONL metadata file + `image_shards/` directory. After extraction, `images/` is created. Format matches the main training data:
```json
{"query": "...", "chunk_path": "images/...", "neg_chunk_paths": ["images/...", "..."], "source_dataset": "moca"}
```

Preparation scripts: `prepare_andy_datasets.py` (build HF folders from raw JSONL), `package_andy_shards.py` (tar shard packaging), `upload_andy_datasets.py` (upload to Hub).

### Generating Synthetic Data (for quick testing)

`fake_data.py` generates template-based queries from article titles (not LLM-augmented):
```bash
uv run python fake_data.py \
    --tiles-dir /opt/dlami/nvme/kiwix_tiles \
    --articles-json /opt/dlami/nvme/kiwix/wikipedia_en_all_maxi_2025-08.zim.articles.json \
    --output-dir training/data --num-articles 1000
```

## Evaluation

```bash
# Verify fine-tuned vs base embeddings
CUDA_VISIBLE_DEVICES=0 uv run python verify_embeddings.py \
    --adapter training/output_nvme/output_contrastors/checkpoint-500 --max-pairs 100

# Gradient correctness tests
CUDA_VISIBLE_DEVICES=0 uv run python tests/test_grad_equivalence.py
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=2 tests/test_grad_multi_gpu.py
```

## Checkpoint Hygiene

Checkpoints produced during training quickly fill up the disk. **Periodically clean up checkpoints that are not promising:**

- After each new checkpoint is produced, check whether `training/output_nvme/` contains old checkpoints that are clearly unneeded (e.g. intermediate versions where loss did not drop or R@1 did not improve).
- If you are sure they are not promising, delete them directly.
- If you are unsure whether to delete, ask the user before acting.

## Launching Training

> **⚠️ All training jobs must be launched inside a tmux session!**
> Running training in a bare terminal = everything is lost when the SSH connection drops. We have already lost a training run once because of this. **Never run torchrun directly in a bare SSH terminal.**

```bash
# Correct approach: launch inside tmux
tmux new-session -d -s train -c /home/ubuntu/wiki-screenshot-training
tmux send-keys -t train "CUDA_VISIBLE_DEVICES=1,2,3 uv run torchrun --nproc_per_node=3 train_contrastors.py \
    [args...] 2>&1 | tee training/output_nvme/<run>/train.log" Enter

# View training output
tmux attach -t train
```

**Never pipe training output through `head`/`tail`/etc.** — SIGPIPE will kill the torchrun workers.

> **⚠️ Before launching training you must confirm that `OPENAI_API_KEY` and `OPENAI_BASE_URL` are set!**
> Without these two environment variables, the QA score of the mini-v6 eval will silently return 0 (the grader swallows all exceptions), and you will think the model is bad when in fact grading never ran.
> **If you find that the current shell has no `OPENAI_API_KEY`, you must immediately remind the user to set it and not continue launching training.**

These two variables should already be configured in `~/.zshrc`. Verify before launching:

```bash
# Mandatory check before launching training
echo "OPENAI_API_KEY=${OPENAI_API_KEY:+SET}" "OPENAI_BASE_URL=${OPENAI_BASE_URL}"
```

**This API key requires `us.api.openai.com`** — using the default `api.openai.com` will 401 silently (the grader catches all exceptions, resulting in QA score = 0 with no visible error).

```bash
export OPENAI_API_KEY="sk-proj-..."
export OPENAI_BASE_URL="https://us.api.openai.com/v1"
```

## Experiment Tracking (CSV)

Every training run must write its experiment results to a CSV file for easy side-by-side comparison.

**File location:** `training/output_nvme/<run_name>/metrics.csv`

**CSV format:**
```csv
step,eval_loss,eval_acc,recall@1,recall@3,qa_score,peak_eval_acc,peak_qa_score
0,,,0.125,0.300,0.42,0.000,0.42
50,1.05,0.62,0.138,0.325,0.45,0.62,0.45
100,0.98,0.65,0.142,0.340,0.48,0.65,0.48
```

**Metrics that must be recorded:**
- `step`: training step count
- `eval_loss`: loss on the eval split
- `eval_acc`: accuracy on the eval split
- `recall@1`, `recall@3`: retrieval recall of the test eval
- `qa_score`: QA score of the test eval (the primary optimization target)
- `peak_eval_acc`: the highest eval accuracy up to the current step
- `peak_qa_score`: the highest QA score up to the current step

**Each run must also record `training/output_nvme/<run_name>/run_config.md`:**
```markdown
# Run: <run_name>
- **Ablation**: describe what this experiment is ablating (which baseline it compares against, what variable was changed)
- **Date**: launch date
- **Machine**: machine name (e.g. colin3)
- **GPUs**: the GPU IDs and count used
- **Key args**: list all non-default parameters
- **Baseline**: name of the baseline run being compared against
- **Hypothesis**: expected effect
```

**How to read:** After training finishes, or during a mid-run check, parse step-level metrics from train.log and write them to the CSV. If the CSV already exists, append new rows.

## Key Findings

- **Hard negatives** are critical for meaningful improvement beyond baseline
- **Primary optimization target is QA score**, not recall@k. Recall can drop while QA score improves (query embeddings become more semantically useful even if exact chunk match rate falls).

## v8r Ablation Results (2026-04-22)

Full stairstep ablation on `training/data/natrual_filtered_v2/split` (350 steps, bs=64, lr=7e-6, lora-vit, visual_tokens=4096), evaluated on both miniv6 (200q, 5291 tiles) and miniv8 (400q, 7426 tiles). vLLM reader: Qwen3-VL-4B-Instruct. Grader: gpt-4.1-2025-04-14.

### Peak metrics across all eval steps

| Run | Config | v6 R@1 | v6 R@3 | **v6 QA** | v8 R@1 | v8 R@3 | **v8 QA** |
|---|---|---|---|---|---|---|---|
| base | no training | 0.650 | 0.800 | 0.665 | 0.688 | 0.833 | 0.730 |
| ab1 | + in-batch only | 0.720 | 0.840 | 0.705 | 0.750 | 0.868 | 0.750 |
| ab2 | + hard negatives | 0.715 | 0.865 | 0.735 | 0.748 | 0.878 | 0.778 |
| ab3 | + text warmup | **0.730** | 0.855 | **0.755** | 0.755 | **0.893** | **0.7825** |
| ab4 | + unfreeze ViT | **0.730** | **0.860** | **0.755** | **0.760** | 0.888 | **0.7825** |

### Final (last checkpoint, step 350) metrics

| Run | Final v6 QA | Final v8 QA | Notes |
|---|---|---|---|
| ab1 | 0.705 | 0.7225 | v8 peaked @step50 (0.750), degraded to 0.7225 — overfit (acc→1.0) |
| ab2 | 0.735 | 0.7425 | v8 peaked @step50 (0.778), degraded to 0.7425 |
| ab3 | 0.745 | 0.780 | close to peak |
| ab4 | 0.755 | 0.7825 | == peak, most stable |

### Key observations

- **Perfect QA stairstep** (peak v6: 0.665→0.705→0.735→0.755=0.755; peak v8: 0.730→0.750→0.778→0.7825=0.7825)
- **ab3 ≈ ab4 in peak** — unfreeze ViT did not add peak QA under this budget (350 steps, single GPU). But ab4 **final** is closer to peak than ab3, so ViT unfreeze adds training stability, not ceiling
- **R@1 is not monotone**: ab2 (0.715) < ab1 (0.720) on v6 — hard neg trades some R@1 for R@3 / QA
- **ab1 / ab2 overfit late**: peak QA reached @step50, then degraded. Lesson: for these configs, shorter training (or early stopping) would have landed better final numbers

### Run dirs

- `training/output_nvme/v8r_base/` (no wandb, eval-only)
- `training/output_nvme/v8r_ab1_inbatch/`
- `training/output_nvme/v8r_ab2_hn2/`
- `training/output_nvme/v8r_ab3_hn2_tw50/`
- `training/output_nvme/v8r_ab4_full/`

Tile caches live next to test images (`test_miniv6/tiles/.tile_cache_*.pt`, `test_miniv8/tiles/.tile_cache_*.pt`). ⚠️ Do not launch 4 runs in parallel from cold — they race to write the cache. Warm the cache first with one eval run, then parallel launches are safe.

## Reader SFT (LlamaFactory)

SFT training of Qwen3-VL-4B to do QA on compressed images (the "reader" model).

**Separate venv**: LlamaFactory has its own dependency environment, **do not use the main project's `.venv`**.

```bash
cd sft/LlamaFactory
source .venv/bin/activate   # separate venv, already has LlamaFactory + deepspeed + wandb installed
```

Key dependencies (already installed):
- flash-linear-attention + causal-conv1d (must be installed, otherwise it silently falls back to slow attention)
- torch 2.9.1+cu129 (same as the main project, cuDNN 9.20 to work around the Conv3D fallback issue)
- deepspeed, wandb

**Data preparation** (compress images + convert to ShareGPT format):
```bash
# Run inside the main project venv
uv run python3 sft/prepare_sft_data.py \
    --dataset-dir /mnt/data/hf_datasets/screenshot-training-natural-filtered-v2 \
    --output-dir /mnt/data/sft_data/compressed_3x \
    --compress-ratio 3 --workers 32
```

**Training** (4 GPUs, launch inside tmux):
```bash
cd sft/LlamaFactory && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1,2,3,4 \
FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=4 \
llamafactory-cli train /home/ubuntu/wiki-screenshot-training/sft/train_qwen3vl_compressed.yaml
```

Key config (`sft/train_qwen3vl_compressed.yaml`):
- template: `qwen3_vl` + `enable_thinking: false` (do not use `qwen3_vl_nothink`, it is not equivalent)
- LoRA rank 32, lr 1e-5, DeepSpeed ZeRO-2
- W&B project: `llamafactory`

## vLLM Serving

**All vLLM instances must use `serving/vllm/`** — this subproject pins vLLM + transformers + torch via `uv.lock`.

Locked versions:
- vLLM 0.19.0
- transformers 4.57.6
- torch 2.10.0

```bash
cd serving/vllm
uv sync           # first time only
uv run vllm serve Qwen/Qwen3-VL-4B-Instruct \
    --dtype auto --port 8201 --max-model-len 65536 \
    --gpu-memory-utilization 0.8 --api-key dummy
```

Never use standalone venvs or other users' venvs to serve models. The `serving/vllm/uv.lock` is the single source of truth for inference-time dependencies.

**Port discovery:** Before launching training that needs vLLM, check what's already running:

```bash
# List active vLLM endpoints
ss -tlnp | grep -E ':8[0-9]{3}\b'
# Verify model at a port
curl -s http://localhost:<port>/v1/models | python3 -m json.tool
```

Use the actual port in `--vllm-url` (e.g. `http://localhost:8200/v1`). Don't assume the default 8201 — it may be on a different port.
