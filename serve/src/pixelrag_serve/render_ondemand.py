"""On-demand tile rendering for pixelrag-serve.

When a tile image is not on disk, render the source page from a kiwix ZIM (served
over HTTP by ``kiwix-serve``), slice it into the same 1024px chunks the index was
built on, and return the requested chunk. This removes the dependency on a
materialized multi-TB ``tiles/`` corpus -- only the pages that retrieval actually
returns get rendered, lazily, and then cached on disk.

Pixel-validated against the original pipeline's tiles (mean |diff| ~2/255 on every
referenced chunk), using the exact build config: viewport_width=875, tile_height=8192,
and the shared ``pixelrag_embed.chunk`` slicer (1024px chunks).
"""

import glob
import os
import shutil
import subprocess
import sys
import threading
from urllib.parse import quote

_render_lock = threading.Lock()  # one Chrome render at a time per process
# Hard timeout (seconds) for a single page render subprocess.
_RENDER_TIMEOUT = float(os.environ.get("PIXELRAG_RENDER_TIMEOUT", "120"))


class OnDemandTiles:
    def __init__(
        self,
        kiwix_url: str,
        book: str,
        cache_dir: str,
        viewport_width: int = 875,
        tile_height: int = 8192,
    ):
        self.kiwix_url = kiwix_url.rstrip("/")
        self.book = book
        self.cache_dir = cache_dir
        self.viewport_width = viewport_width
        self.tile_height = tile_height
        os.makedirs(cache_dir, exist_ok=True)

    def _article_dir(self, article_id: int) -> str:
        return os.path.join(self.cache_dir, f"{article_id}.png.tiles")

    def chunk_path(
        self, article_id: int, title: str, tile_index: int, chunk_index: int
    ):
        """Path to chunk_{ti}_{ci}.png, rendering+chunking the page on a cache miss."""
        chunk_name = f"chunk_{tile_index:04d}_{chunk_index:02d}.png"
        cpath = os.path.join(self._article_dir(article_id), chunk_name)
        if os.path.exists(cpath):
            return cpath
        if not title:
            return None
        with _render_lock:
            if os.path.exists(cpath):  # filled while we waited for the lock
                return cpath
            try:
                self._render_and_chunk(article_id, title)
            except Exception:
                return None
        return cpath if os.path.exists(cpath) else None

    def _render_and_chunk(self, article_id: int, title: str) -> None:
        from pixelrag_embed.chunk import chunk_article

        url = f"{self.kiwix_url}/content/{self.book}/{quote(title, safe='')}"
        staging = os.path.join(self.cache_dir, f".render_{article_id}")
        shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(staging, exist_ok=True)
        # Render in a SEPARATE PROCESS via the pixelshot CLI. render_url internally uses
        # asyncio.run() + multiprocessing.Pool (fork); calling it a second time in this
        # long-lived serve process deadlocks (fork-in-threaded-process). A fresh subprocess
        # per render is the reliable fix — verified: same-process 2nd render hangs, a fresh
        # process per render does not.
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pixelrag_render.render",
                    url,
                    "--output",
                    staging,
                    "--viewport-width",
                    str(self.viewport_width),
                    "--tile-height",
                    str(self.tile_height),
                    "--workers",
                    "1",
                ],
                check=True,
                timeout=_RENDER_TIMEOUT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            shutil.rmtree(staging, ignore_errors=True)
            return
        dirs = glob.glob(os.path.join(staging, "*.png.tiles"))
        if not dirs:
            shutil.rmtree(staging, ignore_errors=True)
            return
        rendered = dirs[0]  # <sanitized-url>.png.tiles/ (has tiles.json)
        chunk_article(rendered)  # writes chunk_XXXX_YY.png + chunks.json
        dest = self._article_dir(article_id)
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(rendered, dest)  # atomic; commit only after chunking succeeds
        shutil.rmtree(staging, ignore_errors=True)
