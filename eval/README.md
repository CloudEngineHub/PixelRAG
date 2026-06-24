# Reproducing PixelRAG paper Table 1 (Qwen3.5-4B, k=3)

Everything needed lives in this directory: the benchmark driver (`run_bench.py`), the
dataset loaders (`lib/`), and the LLM-judge grader (`lib/grader.py`). No external checkout
is required.

The reproduction script just runs the pipeline and prints a score. Run the reader on an
**H100** to match the paper's greedy decode.

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

## Data sources (where each input comes from)

| input | size | source |
|-------|------|--------|
| FAISS indexes (base/lora pixel, text, news) | ~570G | HF dataset `StarTrail-org/pixelrag-faiss-indexes` (4 subdirs; `serve_up.sh` downloads them) |
| reader Qwen3.5-4B / LoRA encoder / training data / QA datasets | — | HF (`Qwen/Qwen3.5-4B`, `Chrisyichuan/*`, `CaraJ/MMSearch`, encyclopedic_vqa csv) |
| **wiki + news tiles** (reader's image evidence) | ~4T | HF dataset `StarTrail-org/pixelrag-tiles` (or regenerate from the public kiwix ZIM via the `render` stage) |
| EVQA/LiveVQA query images (landmark/inat/editorial photo) | ~6G | small; landmark=GLDv2, inat=iNaturalist, livevqa=editorial photos (note: editorial photos are copyrighted — redistribute with care) |

So: indexes, tiles, models, and QA all come straight from HF; the tile corpus can also be
regenerated from the public Wikipedia ZIM via the render pipeline.

### Three ways to run retrieval + supply the tile images

The reader always asks the serve for images (`include_images`), so the retrieved tiles can
reach it three ways — pick one:

1. **Self-hosted serve, index + tiles.** `serve_up.sh` downloads the FAISS index and the tile
   corpus (`StarTrail-org/pixelrag-tiles`, or rendered from the ZIM). The serve returns each
   retrieved tile inline as base64; or set `TILES_DIR` to read the tiles from the reader's
   local disk instead. Full self-host.
2. **Public API (no self-hosting).** Point the retrieval URL at the public endpoint
   (`api.ds-serve.org` / `api.pixelrag.ai`) instead of a local serve. It returns base64 tiles,
   so you only run the reader + grader — no index, no tile corpus.
3. **Self-hosted serve, index + on-demand render.** Run the serve with the index but **no** tile
   corpus, started with an on-demand renderer; it renders each retrieved page to tiles at query
   time and returns them as base64. Needs the kiwix ZIM, not the ~4T corpus.

In modes 2 and 3 the reader needs no local tiles, so leave `TILES_DIR` empty.

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

Before running, `reproduce.sh` runs a **preflight**: it curls the reader and the retrieval
serve(s) that *this* cell needs and checks each is up with the expected index (`/status`
`total_vectors`). If a serve is down / on the wrong port / wrong index, it prints the exact
`pixelrag serve --index-dir … --port …` command to launch it and exits (no silent empty run).

Per-cell config is locked inside `reproduce.sh`:

| bench | think | max_tokens | n | grader | notes |
|-------|-------|-----------|---|--------|-------|
| nq / nqt | no-think | 200 | 1000 / all | exact-match | |
| sqa | no-think | 200 | 1000 | SimpleQA judge | nprobe 2000 |
| mms (base/lora/traf) | **think** | 16384 | all | WorldVQA judge | pixel instr = V1 "Retrieve images or text relevant to the user's query." |
| mms (naive) | no-think | 200 | all | WorldVQA judge | |
| evqa | no-think | 16384 | all | WorldVQA judge | **landmarks + question_type=automatic only**; iNaturalist & templated/multi_answer excluded |
| livevqa (naive/base) | no-think | 16 | all | MCQ exact-match | news pipeline `run_livevqa.py` |

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

On H100, this harness reproduces the pixel cells (LiveVQA/MMS/EVQA base+lora) within ~1pp.

NOTE on traf (text retrieval): `reproduce.sh` passes `--no-query-image` to match the paper's
text-only text retrieval.

## 5. Grader

`eval/lib/grader.py` (faithful to the paper's grading procedure):
- WorldVQA judge (mmsearch / encyclopedic_vqa): prompt verbatim, GT for EVQA =
  `"Any of: " + " | ".join(reference_list)` (any reference matches → correct), `<think>` stripped,
  judge gpt-4.1 temp 0 + `system="You are a helpful assistant."` + `seed=42` + `max_tokens=1000`.
- exact-match (nq / nq_tables): SQuAD-style normalize + match against the gold answer list.
- SimpleQA judge (simpleqa): the SimpleQA `GRADER_TEMPLATE` → A/B/C.

```bash
PYTHONPATH=. .venv/bin/python -m lib.grader <task> <responses.jsonl>
```
