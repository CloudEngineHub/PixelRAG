#!/usr/bin/env bash
# v8s ablation recipe — reproducible launch commands for each run.
# ================================================================
# Evaluated on miniv6 (200q) + miniv8 (400q) test sets.
# Env required: OPENAI_API_KEY, OPENAI_BASE_URL (for QA grader).
# Pre-req: vLLM Qwen3-VL-4B serving at http://localhost:8201/v1
# Pre-req: test_miniv6/tiles + test_miniv8/tiles preprocessed cache (warm first with any run).
#
# Usage:
#   bash recipes/v8s_ablation.sh base        # Eval-only: base Qwen3-VL, no training
#   bash recipes/v8s_ablation.sh s1          # + in-batch (no lora-vit)
#   bash recipes/v8s_ablation.sh s2          # + naive HN
#   bash recipes/v8s_ablation.sh s3          # + filtered HN (Gemini false-pos filter)
#   bash recipes/v8s_ablation.sh s4          # + text warmup
#   bash recipes/v8s_ablation.sh s5          # + lora-vit (final stairstep)
#   bash recipes/v8s_ablation.sh unfiltered  # + in-batch on unfiltered data (low-quality no-filter ablation)
#
# Each adds ONE knob vs the previous. All share: 350 steps, bs=64, grad-cache-chunk 4,
# lr 7e-6, cosine, warmup 20, max-num-visual-tokens 4096, qa-eval-max-examples 1000.

set -u
: "${OPENAI_API_KEY:?missing OPENAI_API_KEY}"
: "${OPENAI_BASE_URL:?missing OPENAI_BASE_URL}"

# Run from the train/ dir (parent of recipes/). Override with REPO_ROOT=... if needed.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

CONFIG="${1:?usage: $0 <base|s1|s2|s3|s4|s5|unfiltered>}"
GPU="${GPU:-0}"

FILTERED_DIR="training/data/natrual_filtered_v2/split"
NAIVE_HN_DIR="data/screenshot-training-naive-top2-hn-ablation"
UNFILTERED_TRAIN="data/screenshot-training-natural-unfiltered-v2/train.jsonl"

COMMON=(
    --test-data test_miniv6/test_miniv6.json test_miniv8/test_miniv8.json
    --max-steps 350
    --batch-size 64
    --grad-cache-chunk 4
    --lr 7e-6
    --warmup-steps 20
    --scheduler cosine
    --test-batch-size 16
    --eval-steps 25
    --test-eval-steps 50
    --save-steps 50
    --max-num-visual-tokens 4096
    --qa-eval-max-examples 1000
    --vllm-url http://localhost:8201/v1
    --vllm-model Qwen/Qwen3-VL-4B-Instruct
)

run() {
    local TAG="$1"; shift
    local OUT="training/output_nvme/$TAG"
    mkdir -p "$OUT"
    echo "[$(date '+%F %T')] launching $TAG on GPU $GPU"
    CUDA_VISIBLE_DEVICES="$GPU" uv run python train_contrastors.py \
        "${COMMON[@]}" \
        "$@" \
        --wandb-run-name "$TAG" \
        --output-dir "$OUT" \
        2>&1 | tee "$OUT/train.log"
}

case "$CONFIG" in
    # Base eval — no training, just run miniv6/v8 eval on pretrained Qwen3-VL-4B.
    # Uses --eval-only to short-circuit training loop.
    base)
        mkdir -p training/output_nvme/v8s_base
        CUDA_VISIBLE_DEVICES="$GPU" uv run python train_contrastors.py \
            --data-split-dir "$FILTERED_DIR" \
            --test-data test_miniv6/test_miniv6.json test_miniv8/test_miniv8.json \
            --eval-only --max-steps 1 --batch-size 4 --grad-cache-chunk 1 \
            --test-batch-size 16 --max-num-visual-tokens 4096 \
            --lora-vit --qa-eval-max-examples 1000 \
            --vllm-url http://localhost:8201/v1 \
            --vllm-model Qwen/Qwen3-VL-4B-Instruct \
            --no-wandb \
            --output-dir training/output_nvme/v8s_base \
            2>&1 | tee training/output_nvme/v8s_base/eval.log
        ;;

    # s1: in-batch only. Filtered-v2 training data but --in-batch-only forces num_hard_negatives=0.
    s1)
        run v8s_inbatch \
            --data-split-dir "$FILTERED_DIR" \
            --in-batch-only \
            --no-lora-vit
        ;;

    # s2: + naive hard negatives (same queries as filtered, but HNs picked top-2 without VLM false-pos filter).
    s2)
        run v8s_naive_hn2 \
            --data-split-dir "$NAIVE_HN_DIR" \
            --num-hard-negatives 2 \
            --no-lora-vit
        ;;

    # s3: + filtered hard negatives (Gemini VLM judge removes false negatives from HN candidates).
    s3)
        run v8s_filtered_hn2 \
            --data-split-dir "$FILTERED_DIR" \
            --num-hard-negatives 2 \
            --no-lora-vit
        ;;

    # s4: + text warmup (50 steps of text-QA pair training before switching to image).
    s4)
        run v8s_filtered_hn2_tw50 \
            --data-split-dir "$FILTERED_DIR" \
            --num-hard-negatives 2 \
            --no-lora-vit \
            --text-warmup-steps 50 \
            --text-data-dir data/text-qa-pair
        ;;

    # s5: + lora-vit (apply LoRA adapters to ViT layers). ViT base weights stay frozen;
    # LoRA delta on ViT attn + MLP is trainable. NOT the same as --unfreeze-vit.
    s5)
        run v8s_filtered_hn2_tw50_lora_vit \
            --data-split-dir "$FILTERED_DIR" \
            --num-hard-negatives 2 \
            --text-warmup-steps 50 \
            --text-data-dir data/text-qa-pair \
            --lora-vit
        ;;

    # unfiltered: in-batch on unfiltered data (no Gemini filter on queries at all).
    # Uses --train-jsonl directly (not --data-split-dir) so eval/test can come from filtered-v2.
    # Eval set shared with filtered-v2 (sanity loss only; v6/v8 peak is what matters).
    unfiltered)
        run v8s_unfiltered_inbatch \
            --train-jsonl "$UNFILTERED_TRAIN" \
            --eval-jsonl data/screenshot-training-natural-filtered-v2/eval_hn.jsonl \
            --test-jsonl data/screenshot-training-natural-filtered-v2/test_hn.jsonl \
            --in-batch-only \
            --no-lora-vit
        ;;

    *)
        echo "unknown config: $CONFIG"
        echo "choose one: base s1 s2 s3 s4 s5 unfiltered"
        exit 2
        ;;
esac

echo "[$(date '+%F %T')] $CONFIG done"
