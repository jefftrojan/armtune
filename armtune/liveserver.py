"""Local HTTP server for `armtune sweep --serve`.

Serves a live-progress dashboard while the sweep runs -- polling /state.json,
which SweepState (livestate.py) is updated with as quantize.py and bench.py
report their own progress -- and then serves the actual generated artifacts
(report.html, results.csv, ...) once they exist, so the same page becomes
the final chartable report when the sweep finishes.

Standard library only (http.server), consistent with armtune's
dependency-free pyproject.toml. Binds to 127.0.0.1 by default: this is a
local dev convenience, not meant to be exposed on a network.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .livestate import SweepState

_STATIC_FILES = {
    "report.html", "report.md", "results.csv", "results_raw.json",
    "recommended_launch.sh", "concurrency_raw.json",
}

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".json": "application/json",
    ".sh": "text/plain; charset=utf-8",
}

_LIVE_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ArmTune sweep -- live</title>
<style>
  :root { --bg:#fff; --fg:#1a1a1a; --muted:#6b7280; --card:#f7f7f8; --border:#e5e7eb; --accent:#4C72B0; --err:#b42318; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14161a; --fg:#e6e6e6; --muted:#9aa1ab; --card:#1c1f24; --border:#2b2f36; }
  }
  * { box-sizing: border-box; }
  body { background: var(--bg); color: var(--fg); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin:0; padding:2rem 1.5rem 4rem; }
  main { max-width: 720px; margin: 0 auto; }
  h1 { font-size: 1.4rem; margin-bottom: 0.25rem; }
  .meta { color: var(--muted); font-size: 0.9rem; }
  .status-line { font-size: 1.05rem; margin: 1.25rem 0 0.5rem; }
  .dot { display:inline-block; width:10px; height:10px; border-radius:50%; background: var(--accent); margin-right:8px; animation: pulse 1.2s infinite; }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.25; } }
  .bar-track { background: var(--card); border:1px solid var(--border); border-radius: 8px; height: 10px; overflow: hidden; margin: 0.5rem 0 1.5rem; }
  .bar-fill { background: var(--accent); height: 100%; width: 4%; transition: width 0.4s ease; }
  .console { background: var(--card); border:1px solid var(--border); border-radius: 10px; padding: 0.75rem 1rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.78rem; height: 280px; overflow-y: auto; white-space: pre-wrap; color: var(--muted); }
  .console div:last-child { color: var(--fg); }
  .error { color: var(--err); font-weight: 600; }
</style>
</head>
<body>
<main>
  <h1>ArmTune sweep</h1>
  <p class="meta">Live progress -- this page updates automatically and will open the full report when done.</p>
  <div class="status-line"><span class="dot" id="dot"></span><span id="status-text">Starting...</span></div>
  <div class="bar-track"><div class="bar-fill" id="bar"></div></div>
  <div class="console" id="console"></div>
</main>
<script>
function poll() {
  fetch('/state.json').then(r => r.json()).then(s => {
    const dot = document.getElementById('dot');
    const statusText = document.getElementById('status-text');
    const bar = document.getElementById('bar');
    const consoleEl = document.getElementById('console');

    if (s.status === 'error') {
      dot.style.animation = 'none';
      dot.style.background = 'var(--err)';
      statusText.innerHTML = '<span class="error">Error: ' + (s.error || 'unknown') + '</span>';
      return;
    }
    if (s.status === 'done') {
      dot.style.animation = 'none';
      statusText.textContent = 'Done -- opening full report...';
      bar.style.width = '100%';
      window.location.href = '/report.html';
      return;
    }

    const suffix = s.total_steps ? ' (' + s.steps_done + '/' + s.total_steps + ')' : '';
    statusText.textContent = s.message + suffix;
    const pct = s.total_steps ? Math.max(4, Math.round((s.steps_done / s.total_steps) * 100)) : 4;
    bar.style.width = pct + '%';

    consoleEl.innerHTML = s.progress_lines.map(function (l) {
      return '<div>' + l.replace(/&/g, '&amp;').replace(/</g, '&lt;') + '</div>';
    }).join('');
    consoleEl.scrollTop = consoleEl.scrollHeight;

    setTimeout(poll, 1000);
  }).catch(function () { setTimeout(poll, 1500); });
}
poll();
</script>
</body>
</html>
"""


def _make_handler(state: SweepState, out_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # silence default request logging
            pass

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path in ("/", "/live", "/index.html"):
                self._send(_LIVE_PAGE.encode(), "text/html; charset=utf-8")
                return
            if self.path == "/state.json":
                self._send(json.dumps(state.to_dict()).encode(), "application/json")
                return

            name = self.path.lstrip("/")
            if name in _STATIC_FILES:
                path = out_dir / name
                if path.exists():
                    ctype = _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
                    self._send(path.read_bytes(), ctype)
                    return
                self.send_response(404)
                self.end_headers()
                return

            self.send_response(404)
            self.end_headers()

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


class LiveServer:
    def __init__(self, state: SweepState, out_dir: Path, host: str = "127.0.0.1", port: int = 8877) -> None:
        self.state = state
        self.out_dir = out_dir
        self.host = host
        self.port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> None:
        handler = _make_handler(self.state, self.out_dir)
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
