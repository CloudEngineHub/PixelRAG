"""Cross-platform Chrome resolution candidates (logic is testable on any OS)."""

import os
from pixelrag_render.chrome import _candidate_chrome_paths, find_chrome


def test_macos_candidates_include_chrome_app():
    paths = _candidate_chrome_paths("Darwin")
    assert any("Google Chrome.app/Contents/MacOS/Google Chrome" in p for p in paths)
    # No Linux system paths leak into the macOS candidate list.
    assert "/usr/bin/google-chrome" not in paths


def test_windows_candidates_include_chrome_exe():
    paths = _candidate_chrome_paths("Windows")
    assert any(p.endswith("chrome.exe") and "Google" in p for p in paths)


def test_linux_candidates_include_system_chrome():
    paths = _candidate_chrome_paths("Linux")
    assert "/usr/bin/google-chrome" in paths
    assert "/usr/bin/chromium" in paths


def test_chrome_path_env_is_first(monkeypatch):
    monkeypatch.setenv("CHROME_PATH", "/custom/chrome")
    assert _candidate_chrome_paths("Linux")[0] == "/custom/chrome"


def test_find_chrome_returns_executable():
    # On any supported dev box this resolves to an installed/usable binary.
    path = find_chrome()
    assert os.path.isfile(path) and os.access(path, os.X_OK)
