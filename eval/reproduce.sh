#!/bin/bash
# PixelRAG paper Table 1 reproduction — one cell at a time.
# Self-contained: uses this repo's eval/run_bench.py + eval/lib (no old Vis-RAG repo).
#
#   bash reproduce.sh <bench> <retrieval>
#     bench     = nq | nqt | sqa | mms | evqa | livevqa
#     retrieval = naive | traf | base | lora
#
# Runs the full pipeline (retrieve -> read -> grade) and prints the score.
# It does NOT compare to the paper and does NOT detect the GPU: run the reader on an
# H100 (see REPRODUCE.md) and the numbers naturally land within ~1pp of the paper.
#
# Env (defaults in [] — see REPRODUCE.md for the serve topology):
#   READER_URL  reader (Qwen3.5-4B, vLLM 0.19.0) OpenAI API base  [http://localhost:8010/v1]
#   BASE_PORT   base pixel search serve   [30088]
#   LORA_PORT   lora pixel search serve   [30096]
#   TEXT_PORT   trafilatura text serve    [30097]
#   NEWS_PORT   news pixel serve (livevqa)[30095]
#   TILES_DIR   local wiki kiwix tiles    [/mnt/data/yichuan/kiwix_tiles]
#   OPENAI_API_KEY / OPENAI_BASE_URL  for the LLM-judge grader (auto-loaded from ../.env)
set -euo pipefail
cd "$(dirname "$0")"

BENCH="${1:?Usage: reproduce.sh <nq|nqt|sqa|mms|evqa|livevqa> <naive|traf|base|lora>}"
RETR="${2:?Usage: reproduce.sh <bench> <naive|traf|base|lora>}"

READER_URL="${READER_URL:-http://localhost:8010/v1}"
BASE_PORT="${BASE_PORT:-30088}"; LORA_PORT="${LORA_PORT:-30096}"
TEXT_PORT="${TEXT_PORT:-30097}"; NEWS_PORT="${NEWS_PORT:-30095}"
TILES_DIR="${TILES_DIR:-/mnt/data/yichuan/kiwix_tiles}"
PY="$(pwd)/.venv/bin/python"
PIXEL_INSTR="Retrieve images or text relevant to the user's query."
TEXT_INSTR="Retrieve text relevant to the user's query."
mkdir -p eval_output
[ -f ../.env ] && { export OPENAI_API_KEY="$(grep '^OPENAI_API_KEY=' ../.env | cut -d= -f2-)"; \
                    export OPENAI_BASE_URL="$(grep '^OPENAI_BASE_URL=' ../.env | cut -d= -f2-)"; }

# --- LiveVQA: separate news pipeline (run_livevqa.py) ---------------------
if [ "$BENCH" = livevqa ]; then
  OUT="eval_output/repro_livevqa_${RETR}.jsonl"
  COMMON=(--api-base "$READER_URL" --model Qwen/Qwen3.5-4B --no-think --max-tokens 16
          --livevqa-images /mnt/data/yichuan/livevqa --output "$OUT")
  case "$RETR" in
    naive) "$PY" run_livevqa.py --mode naive "${COMMON[@]}" ;;
    base)  "$PY" run_livevqa.py --mode pixel --pixel-api "http://localhost:${NEWS_PORT}/search" \
             --pages-db /mnt/data/yichuan/news_state.db --tiles-dir /mnt/data/yichuan/news_tiles "${COMMON[@]}" ;;
    *) echo "livevqa supports: naive | base (MCQ exact-match, scored by run_livevqa.py)" >&2; exit 1 ;;
  esac
  exit 0
fi

# --- per-benchmark config (Qwen3.5-4B, rtk=5, rk=3) -----------------------
case "$BENCH" in
  nq)   TASK=nq;               GRADE=nq;               THINK=off; MAXTOK=200;   N=1000; EXTRA="" ;;
  nqt)  TASK=nq_tables;        GRADE=nq_tables;        THINK=off; MAXTOK=200;   N=1068; EXTRA="" ;;
  sqa)  TASK=simpleqa;         GRADE=simpleqa;         THINK=off; MAXTOK=200;   N=1000; EXTRA="--nprobe 2000" ;;
  mms)  TASK=mmsearch;         GRADE=mmsearch;         THINK=on;  MAXTOK=16384; N=300;  EXTRA="" ;;
  evqa) TASK=encyclopedic_vqa; GRADE=encyclopedic_vqa; THINK=off; MAXTOK=16384; N=1000;
        EXTRA="--evqa-dataset-filter landmarks --evqa-question-type-filter automatic" ;;
  *) echo "unknown bench: $BENCH" >&2; exit 1 ;;
esac
# MMS naive is the one MMS cell the paper ran no-think / max_tokens=200.
[ "$BENCH" = mms ] && [ "$RETR" = naive ] && { THINK=off; MAXTOK=200; }
N="${NUM:-$N}"   # NUM env overrides example count (handy for a quick smoke test)
THINKFLAG=""; [ "$THINK" = off ] && THINKFLAG="--no-think"

# --- retrieval condition --------------------------------------------------
case "$RETR" in
  naive) RFLAGS=() ;;
  base)  RFLAGS=(--local-api --local-api-url "http://localhost:${BASE_PORT}/search" --query-instruction "$PIXEL_INSTR") ;;
  lora)  RFLAGS=(--local-api --local-api-url "http://localhost:${LORA_PORT}/search" --query-instruction "$PIXEL_INSTR") ;;
  traf)  RFLAGS=(--text-api  --text-api-url  "http://localhost:${TEXT_PORT}/search" --query-instruction "$TEXT_INSTR") ;;
  *) echo "unknown retrieval: $RETR" >&2; exit 1 ;;
esac

OUT="eval_output/repro_${BENCH}_${RETR}.jsonl"
echo ">>> [$BENCH/$RETR] run_bench: reader=$READER_URL task=$TASK think=$THINK max_tokens=$MAXTOK n=$N"
# shellcheck disable=SC2086
"$PY" run_bench.py --task "$TASK" --model Qwen/Qwen3.5-4B \
    --api-base "$READER_URL" --api-key dummy $THINKFLAG \
    --retrieval-top-k 5 --reader-top-k 3 --num-examples "$N" --max-tokens "$MAXTOK" \
    --tiles-dir "$TILES_DIR" --output "$OUT" --force --max-concurrent 24 \
    $EXTRA "${RFLAGS[@]}"

echo ">>> [$BENCH/$RETR] grading ($GRADE)"
PYTHONPATH=. "$PY" -m lib.grader "$GRADE" "$OUT"
