"""Measures concurrent-request serving throughput for the winning config.

llama-bench (bench.py) only ever runs one request stream at a time. Real
inference *serving* is almost always concurrent -- multiple requests hitting
llama-server at once -- and that's where batch size actually earns its
keep, in a way single-stream numbers can't show. This module launches
llama-server for one config and fires N simultaneous completion requests at
it, using only the standard library (no new pip dependency), to measure
aggregate tokens/sec as concurrency increases.

This is deliberately scoped to the single winning config from the main
sweep, not the full quant x thread x batch grid: a llama-server start/stop
cycle per concurrency level is much more expensive than a llama-bench
invocation, and the question here is narrower -- "how does *the* recommended
config hold up under concurrent load" -- not a second full sweep.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_PROMPT = "Explain the theory of relativity in one paragraph."


class ServeError(RuntimeError):
    pass


def _wait_for_health(base_url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception as e:  # noqa: BLE001 - server may not be listening yet
            last_err = e
        time.sleep(1)
    raise ServeError(f"llama-server did not become healthy within {timeout}s: {last_err}")


def _one_completion(base_url: str, prompt: str, n_predict: int) -> tuple[int, float]:
    payload = json.dumps({"prompt": prompt, "n_predict": n_predict}).encode()
    req = urllib.request.Request(
        f"{base_url}/completion", data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as r:
        body = json.loads(r.read())
    elapsed = time.perf_counter() - t0
    n_tokens = int(body.get("tokens_predicted", n_predict))
    return n_tokens, elapsed


def bench_concurrency(
    server_bin: Path,
    model_path: str,
    threads: int,
    batch: int,
    concurrency_levels: list[int],
    n_predict: int = 64,
    prompt: str = DEFAULT_PROMPT,
    port: int = 8811,
) -> list[dict]:
    """Launch llama-server once for the given config, then measure
    aggregate tokens/sec at each requested concurrency level.

    Each level fires `n` simultaneous /completion requests and measures
    wall-clock time for the whole batch to finish, so aggregate_tok_s
    reflects real concurrent throughput, not n independent single-stream
    measurements.
    """
    base_url = f"http://127.0.0.1:{port}"
    max_parallel = max(concurrency_levels)
    cmd = [
        str(server_bin), "-m", model_path, "-t", str(threads), "-b", str(batch),
        "--port", str(port), "-c", str(max(4096, n_predict * max_parallel + 1024)),
        "--parallel", str(max_parallel),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results: list[dict] = []
    try:
        _wait_for_health(base_url)
        for n in concurrency_levels:
            results.append(_run_one_level(base_url, n, prompt, n_predict))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    return results


def _run_one_level(base_url: str, n: int, prompt: str, n_predict: int) -> dict:
    token_counts = [0] * n
    errors: list[str] = []

    def worker(i: int) -> None:
        try:
            tok, _ = _one_completion(base_url, prompt, n_predict)
            token_counts[i] = tok
        except Exception as e:  # noqa: BLE001 - report and move on
            errors.append(str(e))

    workers = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    t0 = time.perf_counter()
    for th in workers:
        th.start()
    for th in workers:
        th.join()
    wall_s = time.perf_counter() - t0

    if errors:
        return {"concurrency": n, "error": errors[0]}

    total_tokens = sum(token_counts)
    aggregate_tok_s = total_tokens / wall_s if wall_s > 0 else 0.0
    return {
        "concurrency": n,
        "wall_s": round(wall_s, 3),
        "total_tokens": total_tokens,
        "aggregate_tok_s": round(aggregate_tok_s, 2),
        "per_request_tok_s": round(aggregate_tok_s / n, 2),
    }


# --------------------------------------------------------------------------
# Mock: synthetic concurrency curve for local pipeline testing, matching the
# same "used by --mock, never a substitute for real numbers" pattern as
# bench.py's generate_mock_results.
# --------------------------------------------------------------------------

def generate_mock_concurrency(concurrency_levels: list[int], single_stream_tok_s: float) -> list[dict]:
    """Synthetic aggregate-throughput curve: scales with concurrency but
    with diminishing returns past ~4 concurrent requests, like a real CPU
    server contending for the same cores/memory bandwidth."""
    results = []
    for n in concurrency_levels:
        eff = min(n, 4) + max(0, n - 4) * 0.4
        aggregate_tok_s = round(single_stream_tok_s * (eff ** 0.7), 2)
        results.append({
            "concurrency": n,
            "wall_s": None,
            "total_tokens": None,
            "aggregate_tok_s": aggregate_tok_s,
            "per_request_tok_s": round(aggregate_tok_s / n, 2),
            "armtune_mock": True,
        })
    return results
