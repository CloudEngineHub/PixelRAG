"""Chrome binary management for pixelshot.

Downloads and manages a patched headless Chrome binary with rawFilePath
support. Similar to `playwright install chromium`.

Usage:
    pixelshot install-chrome     # download patched headless_shell
    pixelshot which-chrome       # print path to active binary

Programmatic:
    from pixelrag_render.chrome import find_chrome, install_chrome
    path = find_chrome()             # auto-detect best available
    path = install_chrome()          # download if needed
"""

import json
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

INSTALL_DIR = Path.home() / ".cache" / "pixelrag" / "chrome"
VERSION_FILE = "version.json"

# Update these when releasing a new build
CHROME_VERSION = "150.0.7844.0"
RELEASE_URL_TEMPLATE = (
    "https://github.com/andylizf/pixelrag/releases/download/"
    "chrome-{version}/headless_shell-linux-x64.tar.zst"
)

# Search order for find_chrome()
_SEARCH_PATHS = [
    lambda: os.environ.get("CHROME_PATH", ""),
    lambda: str(INSTALL_DIR / "headless_shell"),
    lambda: os.path.expanduser(
        "~/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome"
    ),
    lambda: "/usr/bin/google-chrome",
    lambda: "/usr/bin/google-chrome-stable",
    lambda: "/usr/bin/chromium-browser",
    lambda: "/usr/bin/chromium",
]


def find_chrome(auto_install: bool = True) -> str:
    """Find the best available Chrome binary. Auto-installs if none found.

    Search order:
    1. CHROME_PATH env var
    2. pixelrag-installed headless_shell (~/.cache/pixelrag/chrome/)
    3. Playwright's Chrome
    4. System Chrome/Chromium
    5. Auto-install patched headless_shell (if auto_install=True)

    Returns:
        Path to Chrome binary.

    Raises:
        FileNotFoundError: No Chrome binary found and auto_install=False.
    """
    for path_fn in _SEARCH_PATHS:
        path = path_fn()
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    if auto_install:
        print("No Chrome found. Installing headless_shell...", flush=True)
        return str(install_chrome())

    raise FileNotFoundError(
        "No Chrome binary found. Run 'pixelshot install-chrome' or set CHROME_PATH."
    )


def get_installed_version() -> str | None:
    """Return version string of installed headless_shell, or None."""
    version_path = INSTALL_DIR / VERSION_FILE
    if version_path.exists():
        try:
            data = json.loads(version_path.read_text())
            return data.get("version")
        except Exception:
            pass
    return None


def install_chrome(version: str | None = None, force: bool = False) -> Path:
    """Download and install the patched headless_shell binary.

    Args:
        version: Chrome version to install. Defaults to CHROME_VERSION.
        force: Re-download even if already installed.

    Returns:
        Path to the installed headless_shell binary.
    """
    version = version or CHROME_VERSION
    binary_path = INSTALL_DIR / "headless_shell"

    if binary_path.exists() and not force:
        installed = get_installed_version()
        if installed == version:
            print(f"Already installed: headless_shell {version}")
            return binary_path

    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError(
            f"Pre-built headless_shell only available for linux-x64, "
            f"got {platform.system()}-{platform.machine()}"
        )

    url = RELEASE_URL_TEMPLATE.format(version=version)
    print(f"Downloading headless_shell {version}...")
    print(f"  URL: {url}")

    INSTALL_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".tar.zst", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urllib.request.urlretrieve(url, tmp_path, _progress_hook)
        print()

        # Decompress: zstd → tar → extract
        print("Extracting...")
        # Try zstd decompression
        decomp_path = tmp_path + ".tar"
        try:
            subprocess.run(
                ["zstd", "-d", tmp_path, "-o", decomp_path],
                check=True,
                capture_output=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            # Fallback: try python zstandard
            try:
                import zstandard

                with open(tmp_path, "rb") as f_in, open(decomp_path, "wb") as f_out:
                    dctx = zstandard.ZstdDecompressor()
                    dctx.copy_stream(f_in, f_out)
            except ImportError:
                raise RuntimeError(
                    "zstd not found. Install with: apt install zstd (or pip install zstandard)"
                )

        with tarfile.open(decomp_path) as tar:
            tar.extractall(INSTALL_DIR)
        os.unlink(decomp_path)

        # Set executable permission
        binary_path.chmod(0o755)

        # Write version file
        version_data = {"version": version, "binary": str(binary_path)}
        (INSTALL_DIR / VERSION_FILE).write_text(json.dumps(version_data))

        print(
            f"Installed: {binary_path} ({binary_path.stat().st_size / 1024 / 1024:.0f}MB)"
        )
        return binary_path

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / 1024 / 1024
        total_mb = total_size / 1024 / 1024
        print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct}%)", end="", flush=True)


def main():
    """CLI entry point for chrome management."""
    import argparse

    parser = argparse.ArgumentParser(description="Manage Chrome for pixelshot")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("install", help="Download patched headless_shell")
    sub.add_parser("which", help="Print path to active Chrome binary")
    sub.add_parser("version", help="Print installed version")

    args = parser.parse_args()

    if args.command == "install":
        install_chrome()
    elif args.command == "which":
        try:
            print(find_chrome())
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
    elif args.command == "version":
        v = get_installed_version()
        if v:
            print(v)
        else:
            print("Not installed", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
