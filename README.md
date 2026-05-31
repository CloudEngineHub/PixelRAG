<h1 align="center">PixelRAG</h1>
<p align="center">Visual Retrieval-Augmented Generation.<br>Search any document by how it <em>looks</em>, not just the text it contains.</p>

<p align="center">
  <a href="https://github.com/StarTrail-org/PixelRAG/actions/workflows/ci.yml"><img src="https://github.com/StarTrail-org/PixelRAG/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pixelrag.ai"><img src="https://img.shields.io/badge/demo-pixelrag.ai-7c3aed" alt="Live demo"></a>
  <a href="https://status.pixelrag.ai"><img src="https://img.shields.io/badge/status-live-22c55e" alt="Status"></a>
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License">
</p>

<p align="center">
  <a href="#architecture">Architecture</a> &middot;
  <a href="#build-an-index-from-your-own-documents">Build an index</a> &middot;
  <a href="#claude-code-plugin">Claude plugin</a> &middot;
  <a href="#training">Training</a>
</p>

<!-- TODO(demo): add a hero GIF/screenshot of the search UI or pixelrag.ai here. -->

---

PixelRAG renders documents — web pages, PDFs, images — as screenshots, embeds them with a
vision-language model, builds FAISS indexes, and serves a search API. Wikipedia's 8.28M
articles ship as a pre-built index; the pipeline itself is general-purpose.

```bash
uv sync --package pixelrag-serve

# Download a pre-built index (8.28M Wikipedia pages)
aws s3 sync s3://wiki-screenshot-tiles-backup/kiwix_tiles/text_search_index_1024/ ./index/

# Serve it
pixelrag-serve --index-dir ./index --port 30001

# Query
curl -X POST http://localhost:30001/search \
  -H "Content-Type: application/json" \
  -d '{"queries": [{"text": "What is the capital of France?"}], "n_docs": 5}'
```

That's retrieval over 8.28M pages indexed as images. Try it live at [pixelrag.ai](https://pixelrag.ai).

## Architecture

Five packages, each independently installable:

| Package             | What it does                                                    | Install                             |
| ------------------- | --------------------------------------------------------------- | ----------------------------------- |
| **pixelrag-render** | Document → image tiles (Playwright CDP, PDF)                    | `uv sync --package pixelrag-render` |
| **pixelrag-embed**  | Tiles → vectors → FAISS index (three independent tools)         | `uv sync --package pixelrag-embed`  |
| **pixelrag-index**  | Orchestrates the full pipeline: source → ingest → embed → index | `uv sync --package pixelrag-index`  |
| **pixelrag-serve**  | FAISS search API (FastAPI, CPU or GPU)                          | `uv sync --package pixelrag-serve`  |
| **pixelrag-train**  | LoRA fine-tuning for Qwen3-VL-Embedding                         | `cd train && uv sync`               |

```
render ←── index ──→ embed       serve (independent)       train → serve (HTTP)
```

`render`/`embed`/`index`/`serve` share the root workspace. **`train` is a separate
uv project** with its own pinned env (`torch==2.9.1+cu129`, `transformers==4.57.1`,
cuDNN 9.20) — install it from inside `train/`, not from the root.

## Build an index from your own documents

```bash
uv sync --package pixelrag-index

# Create pixelrag.yaml
cat > pixelrag.yaml << 'EOF'
source:
  type: local
  path: ./my_docs

embed:
  model: Qwen/Qwen3-VL-Embedding-2B
  device: cuda
  gpu_ids: [0]

output: ./my_index
EOF

# Build, then serve
pixelrag-index build
pixelrag-serve --index-dir ./my_index --port 30001
```

Render a single URL (for agent use):

```python
from pixelrag_render import render_url

tiles = render_url("https://en.wikipedia.org/wiki/Python", "./tiles")
```

## Claude Code plugin

Give Claude eyes — no MCP server, no backend. The plugin teaches Claude to call
`pixelrag-render` directly via Bash and read the resulting tile images.

```bash
# One-time setup
./plugin/setup.sh

# Then copy-paste any of these:
claude --plugin-dir ./plugin -p "screenshot https://news.ycombinator.com and summarize the top stories"
claude --plugin-dir ./plugin -p "screenshot https://arxiv.org/abs/2404.12387 and explain the key findings"
claude --plugin-dir ./plugin -p "screenshot http://localhost:3000 and tell me if anything looks broken"
```

Or start an interactive session and use the slash command:

```bash
claude --plugin-dir ./plugin
# then type: /screenshot https://example.com
```

## Embed tools (standalone)

Each tool runs independently, without the orchestrator:

```bash
pixelrag-chunk --tiles-dir ./tiles
pixelrag-embed --shard-dir ./tiles --output-dir ./embeddings --gpu-ids 0,1
pixelrag-build-index --embeddings-dir ./embeddings --output-dir ./index
```

## Training

`pixelrag-train` LoRA fine-tunes `Qwen/Qwen3-VL-Embedding-2B` for webpage retrieval.
See [`train/README.md`](train/README.md) for the full recipe.

You don't need to retrain to use the model — the trained adapters are published at
[`Chrisyichuan/wiki-screenshot-embedding-lora`](https://huggingface.co/Chrisyichuan/wiki-screenshot-embedding-lora/tree/main/lora_vit/ckpt200).

We also release the full training set
([`Chrisyichuan/screenshot-training-natural-filtered-v2`](https://huggingface.co/datasets/Chrisyichuan/screenshot-training-natural-filtered-v2)),
so you can adapt other backbones yourself — a larger Qwen, or any other embedding model.

## License

Apache-2.0
