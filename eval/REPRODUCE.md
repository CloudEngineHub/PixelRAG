# Reproducing PixelRAG paper Table 1 (Qwen3.5-4B, k=3)

Self-contained in this repo (`eval/run_bench.py` + `eval/lib/` + `eval/lib/grader.py`).
**No dependency on the old `Vis-RAG` / `dr-agent` repo.** The driver and grader were
migrated from it (provenance noted in the file headers); the old repo can be deleted.

The reproduction script just runs the pipeline and prints a score. It does **not** compare
to the paper and does **not** branch on hardware. Run the reader on an **H100** and the
numbers land within ~1pp of the paper (B200 systematically diverges ~0.6–1.6pp on the
greedy decode; see `gpu-hardware-reproduction`).

## 1. Environment (locked)

```bash
cd eval
uv sync --frozen        # creates eval/.venv from pyproject.toml + uv.lock (Python 3.12)
```

Grader needs an OpenAI key with access to `gpt-4.1-2025-04-14`. `reproduce.sh` auto-loads
`OPENAI_API_KEY` / `OPENAI_BASE_URL` from `../.env`.

## 2. Serve topology (must be running before `reproduce.sh`)

| role | default port | index / model | notes |
|------|------|------|------|
| **reader** | `READER_URL` :8010 | `Qwen/Qwen3.5-4B`, **vLLM 0.19.0**, **H100** | `CUDA_VISIBLE_DEVICES=0 HF_HOME=… vllm serve Qwen/Qwen3.5-4B --port 8010` on an H100; tunnel it to :8010 |
| base pixel | :30088 | `search_index_normed_v2` (wiki, 28.2M), base encoder, direct_gpu | multimodal query |
| lora pixel | :30096 | wiki lora-vit-ckpt200 index (26.3M) | multimodal query |
| traf text  | :30097 | `text_search_index_1024_normed` (wiki, 15.7M, nprobe 128) | text query |
| news pixel | :30095 | `news_image_search_index` (3.63M, nprobe 128), base, direct_gpu | LiveVQA only |

All pixel/text serves are **direct_gpu** (the reader sends the raw query; the serve encodes
it — do NOT POST precomputed embeddings). Local tiles for the reader live at
`TILES_DIR=/mnt/data/yichuan/kiwix_tiles` (wiki) and `/mnt/data/yichuan/news_tiles` (news);
EVQA query images at `/mnt/data/yichuan/{landmark,inat}_images/`. The HF datasets
(`CaraJ/MMSearch`, encyclopedic_vqa csv) are read from `~/.cache`. LiveVQA reads its QA
dataset (questions/options/GT/img_path) from `LIVEVQA_V4_PATH`
(default `/mnt/data/yichuan/livevqa_v4_multimodal.json`; retrieval is re-done live).

These data dirs are large external inputs (not vendored in the repo), same as the tile
stores and HF caches.

## 3. Run a cell

```bash
bash reproduce.sh <bench> <retrieval>
#   bench     = nq | nqt | sqa | mms | evqa | livevqa
#   retrieval = naive | traf | base | lora
# e.g.
bash reproduce.sh evqa base       # -> prints  Score: 0.4xx
bash reproduce.sh mms lora
NUM=20 bash reproduce.sh nq traf  # NUM overrides the example count for a quick smoke
```

Per-cell config is locked inside `reproduce.sh` (verified against the paper's saved
response metadata, not the experiment scripts):

| bench | think | max_tokens | n | grader | notes |
|-------|-------|-----------|---|--------|-------|
| nq / nqt | no-think | 200 | 1000 / 1068 | exact-match | |
| sqa | no-think | 200 | 1000 | SimpleQA judge | nprobe 2000 |
| mms (base/lora/traf) | **think** | 16384 | 300 | WorldVQA judge | pixel instr = V1 "Retrieve images or text relevant to the user's query." (NOT promptG) |
| mms (naive) | no-think | 200 | 300 | WorldVQA judge | |
| evqa | no-think | 16384 | 749 | WorldVQA judge | **landmarks + question_type=automatic only**; iNaturalist & templated/multi_answer excluded |
| livevqa (naive/base) | no-think | 16 | 26888 | MCQ exact-match | news pipeline `run_livevqa.py` |

## 4. Published numbers (for your own comparison — NOT used by the script)

Paper Table 1 (Qwen3.5-4B, k=3):

| | naive | Trafilatura | base | LoRA |
|---|---|---|---|---|
| NQ | 30.4 | 55.9 | 57.9 | 58.7 |
| NQ-Tables | 24.5 | 42.5 | 47.0 | 48.8 |
| SimpleQA | 7.0 | 71.6 | 73.8 | 78.8 |
| LiveVQA | 63.6 | 59.0 | 70.3 | 70.0 |
| MMSearch | 12.7 | 24.7 | 28.3 | 28.3 |
| EVQA (lm/auto) | 27.2 | 29.6 | 40.7 | 45.1 |

On H100, this harness reproduces every pixel cell (LiveVQA/MMS/EVQA base+lora) within ~1pp.
The MMS/EVQA grader (`gpt-4.1-2025-04-14`, temp 0) has ~2–6pp run-to-run noise, so re-grading
even the paper's own responses wanders by that much. EVQA-traf is the one residual (~+4 vs
published): same text index as the paper, but this serve instance encodes the text query
better → higher recall → higher accuracy (a positive, not a config error).

## 5. Grader

`eval/lib/grader.py` (migrated, byte-faithful to the paper's `evaluate.py` + `worldvqa_eval`):
- WorldVQA judge (mmsearch / encyclopedic_vqa): prompt verbatim, GT for EVQA =
  `"Any of: " + " | ".join(reference_list)` (any reference matches → correct), `<think>` stripped,
  judge gpt-4.1 temp 0 + `system="You are a helpful assistant."` + `seed=42` + `max_tokens=1000`.
- exact-match (nq / nq_tables): SQuAD-style normalize + match against the gold answer list.
- SimpleQA judge (simpleqa): the SimpleQA `GRADER_TEMPLATE` → A/B/C.

```bash
PYTHONPATH=. .venv/bin/python -m lib.grader <task> <responses.jsonl>
```
