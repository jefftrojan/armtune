"""Renders a self-contained report.html next to report.md.

report.md is the right format for reading in a PR/terminal/job log, but it's
a poor format for actually *seeing* how throughput moves across the sweep --
that needs a chart, not a sorted table. This module renders one, as plain
SVG built with string formatting (no matplotlib/plotly, no CDN, no JS
framework) so it stays consistent with armtune's dependency-free
pyproject.toml and still opens correctly with zero network access.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape

from .report import ConfigResult, cost_per_1m_tokens

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#64B5CD"]


def _best_by_threads(results: list[ConfigResult], quant: str) -> list[tuple[int, float]]:
    by_threads: dict[int, float] = {}
    for r in results:
        if r.quant != quant:
            continue
        if r.threads not in by_threads or r.tg_tokens_per_s > by_threads[r.threads]:
            by_threads[r.threads] = r.tg_tokens_per_s
    return sorted(by_threads.items())


def _svg_line_chart(
    series: dict[str, list[tuple[int, float]]],
    *, title: str, x_label: str, y_label: str,
    width: int = 640, height: int = 360,
) -> str:
    margin = {"top": 40, "right": 150, "bottom": 50, "left": 60}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    all_x = sorted({x for pts in series.values() for x, _ in pts})
    all_y = [y for pts in series.values() for _, y in pts]
    if not all_x or not all_y:
        return "<p>No data.</p>"
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = 0.0, max(all_y) * 1.1

    def sx(x: float) -> float:
        if x_max == x_min:
            return margin["left"] + plot_w / 2
        return margin["left"] + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return margin["top"] + plot_h - (y - y_min) / (y_max - y_min) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{escape(title)}">']
    parts.append(f'<text x="{width / 2}" y="20" class="chart-title" text-anchor="middle">{escape(title)}</text>')

    for i in range(6):
        y_val = y_min + (y_max - y_min) * i / 5
        y_px = sy(y_val)
        parts.append(f'<line x1="{margin["left"]}" y1="{y_px:.1f}" x2="{width - margin["right"]}" y2="{y_px:.1f}" class="gridline" />')
        parts.append(f'<text x="{margin["left"] - 8}" y="{y_px + 4:.1f}" class="tick-label" text-anchor="end">{y_val:.0f}</text>')

    for x in all_x:
        x_px = sx(x)
        parts.append(f'<text x="{x_px:.1f}" y="{height - margin["bottom"] + 20}" class="tick-label" text-anchor="middle">{x}</text>')

    parts.append(f'<text x="{width / 2}" y="{height - 8}" class="axis-label" text-anchor="middle">{escape(x_label)}</text>')
    parts.append(f'<text x="16" y="{height / 2}" class="axis-label" text-anchor="middle" transform="rotate(-90 16 {height / 2})">{escape(y_label)}</text>')

    for i, (quant, pts) in enumerate(sorted(series.items())):
        color = _PALETTE[i % len(_PALETTE)]
        poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.5" />')
        for x, y in pts:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{color}" />')
        legend_y = margin["top"] + i * 22
        legend_x = width - margin["right"] + 16
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="12" height="12" fill="{color}" rx="2" />')
        parts.append(f'<text x="{legend_x + 18}" y="{legend_y}" class="legend-label">{escape(quant)}</text>')

    parts.append("</svg>")
    return "".join(parts)


def _svg_bar_chart(
    bars: list[tuple[str, float]],
    *, title: str, y_label: str, value_fmt: str = "{:.1f}",
    width: int = 640, height: int = 320,
) -> str:
    margin = {"top": 40, "right": 20, "bottom": 60, "left": 60}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    if not bars:
        return "<p>No data.</p>"
    max_val = max((v for _, v in bars), default=0.0) * 1.15 or 1.0
    n = len(bars)
    slot = plot_w / n
    bar_w = slot * 0.5

    def sy(v: float) -> float:
        return margin["top"] + plot_h - (v / max_val) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{escape(title)}">']
    parts.append(f'<text x="{width / 2}" y="20" class="chart-title" text-anchor="middle">{escape(title)}</text>')

    for i in range(5):
        v = max_val * i / 4
        y_px = sy(v)
        parts.append(f'<line x1="{margin["left"]}" y1="{y_px:.1f}" x2="{width - margin["right"]}" y2="{y_px:.1f}" class="gridline" />')
        parts.append(f'<text x="{margin["left"] - 8}" y="{y_px + 4:.1f}" class="tick-label" text-anchor="end">{v:.1f}</text>')

    for i, (label, val) in enumerate(bars):
        x0 = margin["left"] + i * slot + (slot - bar_w) / 2
        y0 = sy(val)
        h = margin["top"] + plot_h - y0
        color = _PALETTE[i % len(_PALETTE)]
        parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" rx="3" />')
        parts.append(f'<text x="{x0 + bar_w / 2:.1f}" y="{y0 - 6:.1f}" class="bar-value" text-anchor="middle">{value_fmt.format(val)}</text>')
        parts.append(f'<text x="{x0 + bar_w / 2:.1f}" y="{height - margin["bottom"] + 20}" class="tick-label" text-anchor="middle">{escape(label)}</text>')

    parts.append(f'<text x="16" y="{height / 2}" class="axis-label" text-anchor="middle" transform="rotate(-90 16 {height / 2})">{escape(y_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _data_table(results: list[ConfigResult]) -> str:
    cols = [
        ("quant", "Quant", "str"), ("threads", "Threads", "num"), ("batch", "Batch", "num"),
        ("model_size_mib", "Size (MiB)", "num"), ("ttft_ms", "TTFT (ms)", "num"),
        ("pp_tokens_per_s", "Prompt t/s", "num"), ("tg_tokens_per_s", "Gen t/s", "num"),
    ]
    head = "".join(f'<th data-key="{k}" data-type="{t}">{label} <span class="sort-arrow"></span></th>' for k, label, t in cols)
    rows = []
    for r in results:
        cells = "".join(f"<td>{getattr(r, k)}</td>" for k, _, _ in cols)
        rows.append(f"<tr>{cells}</tr>")
    return (
        f'<table id="sweep-table"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


_SORT_SCRIPT = """
document.querySelectorAll('#sweep-table th').forEach((th, idx) => {
  th.addEventListener('click', () => {
    const table = th.closest('table');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const type = th.dataset.type;
    const asc = th.dataset.asc !== 'true';
    table.querySelectorAll('th').forEach(h => { h.dataset.asc = ''; h.querySelector('.sort-arrow').textContent = ''; });
    th.dataset.asc = String(asc);
    th.querySelector('.sort-arrow').textContent = asc ? ' \\u25B2' : ' \\u25BC';
    rows.sort((a, b) => {
      const av = a.children[idx].textContent, bv = b.children[idx].textContent;
      const cmp = type === 'num' ? (parseFloat(av) - parseFloat(bv)) : av.localeCompare(bv);
      return asc ? cmp : -cmp;
    });
    rows.forEach(r => tbody.appendChild(r));
  });
});
""".strip()

_CSS = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #6b7280; --card: #f7f7f8;
  --border: #e5e7eb; --grid: #e5e7eb; --accent: #4C72B0;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #14161a; --fg: #e6e6e6; --muted: #9aa1ab; --card: #1c1f24; --border: #2b2f36; --grid: #333844; }
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--fg); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 0; padding: 2rem 1.5rem 4rem; line-height: 1.5; }
main { max-width: 960px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
h2 { font-size: 1.15rem; margin-top: 2.5rem; border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; }
.meta { color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }
.mock-banner { background: #7a4b00; color: #fff; padding: 0.6rem 1rem; border-radius: 6px; font-size: 0.9rem; margin-bottom: 1.5rem; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.75rem; margin: 1rem 0; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 0.9rem 1rem; }
.card .label { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.03em; }
.card .value { font-size: 1.3rem; font-weight: 600; margin-top: 0.15rem; }
.card .sub { color: var(--muted); font-size: 0.82rem; margin-top: 0.15rem; }
.chart-wrap { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 0.75rem; overflow-x: auto; }
.chart { width: 100%; height: auto; min-width: 480px; }
.chart-title { fill: var(--fg); font-size: 13px; font-weight: 600; }
.axis-label { fill: var(--muted); font-size: 11px; }
.tick-label { fill: var(--muted); font-size: 10px; }
.legend-label { fill: var(--fg); font-size: 11px; }
.bar-value { fill: var(--fg); font-size: 11px; font-weight: 600; }
.gridline { stroke: var(--grid); stroke-width: 1; }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th, td { padding: 0.4rem 0.6rem; text-align: right; border-bottom: 1px solid var(--border); }
th:first-child, td:first-child { text-align: left; }
th { cursor: pointer; user-select: none; color: var(--muted); font-weight: 600; white-space: nowrap; }
th:hover { color: var(--fg); }
.sort-arrow { font-size: 0.7em; }
footer { color: var(--muted); font-size: 0.8rem; margin-top: 3rem; }
"""


def render_html_report(
    results: list[ConfigResult],
    winners: dict[str, ConfigResult],
    model_label: str,
    cpu_label: str,
    cost_per_hour: float | None = None,
    concurrency_results: list[dict] | None = None,
) -> str:
    is_mock = any(r.mock for r in results)
    w = winners["fastest_throughput"]
    v = winners["best_value"]
    l = winners["lowest_ttft"]
    b = winners["baseline"]

    quants = sorted({r.quant for r in results})
    series = {q: _best_by_threads(results, q) for q in quants}
    throughput_chart = _svg_line_chart(
        series, title="Generation throughput vs. threads", x_label="threads", y_label="tok/s",
    )

    baseline_bars = [("Baseline", b.tg_tokens_per_s), ("Tuned (fastest)", w.tg_tokens_per_s)]
    baseline_chart = _svg_bar_chart(baseline_bars, title="Baseline vs. tuned throughput", y_label="tok/s")

    cost_html = ""
    if cost_per_hour:
        b_cost = cost_per_1m_tokens(b.tg_tokens_per_s, cost_per_hour)
        w_cost = cost_per_1m_tokens(w.tg_tokens_per_s, cost_per_hour)
        cost_bars = [("Baseline", b_cost), ("Tuned (fastest)", w_cost)]
        cost_chart = _svg_bar_chart(cost_bars, title="$ per 1M generated tokens", y_label="$ / 1M tok", value_fmt="${:.4f}")
        cost_html = f'<div class="chart-wrap">{cost_chart}</div>'

    concurrency_html = ""
    if concurrency_results:
        clean = [r for r in concurrency_results if "error" not in r]
        if clean:
            conc_bars = [(str(r["concurrency"]), r["aggregate_tok_s"]) for r in clean]
            conc_chart = _svg_bar_chart(conc_bars, title="Aggregate serving throughput vs. concurrency", y_label="tok/s")
            concurrency_html = f"""
            <h2>Concurrent serving throughput</h2>
            <p class="meta">Winning config only, via <code>llama-server</code> under N simultaneous requests.</p>
            <div class="chart-wrap">{conc_chart}</div>
            """

    mock_banner = (
        '<div class="mock-banner">SYNTHETIC DEMO DATA (--mock) — not a real benchmark.</div>' if is_mock else ""
    )

    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ArmTune sweep report</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<h1>ArmTune sweep report</h1>
<p class="meta">Model: <code>{escape(model_label)}</code> &middot; Host CPU: <code>{escape(cpu_label)}</code> &middot; Generated: {generated}</p>
{mock_banner}

<div class="cards">
  <div class="card"><div class="label">Fastest throughput</div>
    <div class="value">{w.tg_tokens_per_s} tok/s</div>
    <div class="sub">{escape(w.quant)}, {w.threads} threads, batch {w.batch}</div></div>
  <div class="card"><div class="label">Best value</div>
    <div class="value">{v.model_size_mib} MiB</div>
    <div class="sub">{escape(v.quant)}, {v.threads} threads, batch {v.batch} &middot; {v.tg_tokens_per_s} tok/s</div></div>
  <div class="card"><div class="label">Lowest TTFT</div>
    <div class="value">{l.ttft_ms} ms</div>
    <div class="sub">{escape(l.quant)}, {l.threads} threads, batch {l.batch}</div></div>
</div>

<h2>Throughput across the sweep</h2>
<div class="chart-wrap">{throughput_chart}</div>

<h2>Baseline vs. tuned</h2>
<p class="meta">Baseline = untuned deploy default: least-compressed quant (<code>{escape(b.quant)}</code>), all {b.threads} threads, batch {b.batch}.</p>
<div class="chart-wrap">{baseline_chart}</div>
{cost_html}

{concurrency_html}

<h2>Full sweep</h2>
<p class="meta">Click a column header to sort.</p>
{_data_table(results)}

<footer>Generated by <code>armtune sweep</code>.</footer>
</main>
<script>{_SORT_SCRIPT}</script>
</body>
</html>
"""
