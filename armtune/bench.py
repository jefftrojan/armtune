"""Runs llama.cpp's llama-bench across a sweep of models/threads/batch sizes.

llama-bench natively accepts comma-separated values for -m/-t/-b/-p/-n and
benchmarks the full cartesian product in a single process invocation, so
ArmTune's job is mostly: build the right command line, capture -o json,
and hand the parsed rows to the report module.
"""

from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable


class BenchError(RuntimeError):
    pass


def run_llama_bench(
    bench_bin: Path,
    model_paths: dict[str, Path],
    threads: list[int],
    batch_sizes: list[int],
    n_prompt: int,
    n_gen: int,
    repetitions: int = 3,
    on_quant_start: Callable[[str], None] | None = None,
    on_quant_done: Callable[[str], None] | None = None,
    on_progress_line: Callable[[str], None] | None = None,
) -> list[dict]:
    """Run llama-bench across every quantized model and every thread/batch combo.

    Runs one llama-bench invocation per model so results stay clearly
    attributed to a quant type even if llama-bench's own model-type string
    is ambiguous.

    llama-bench prints its own --progress lines to stderr (stdout stays pure
    JSON for -o json) as each test point completes -- we stream the process
    instead of blocking on subprocess.run so those lines are visible in real
    time, both on our own stderr and via on_progress_line for callers like
    the --serve live dashboard.
    """
    all_rows: list[dict] = []

    for quant, model_path in model_paths.items():
        if on_quant_start:
            on_quant_start(quant)

        cmd = [
            str(bench_bin),
            "-m", str(model_path),
            "-t", ",".join(str(t) for t in threads),
            "-b", ",".join(str(b) for b in batch_sizes),
            "-p", str(n_prompt),
            "-n", str(n_gen),
            "-r", str(repetitions),
            "-o", "json",
            "--progress",
        ]
        print(f"[armtune] benchmarking {quant}: {' '.join(cmd)}")

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

        def _drain_stderr(p: subprocess.Popen = proc) -> None:
            for line in p.stderr:
                line = line.rstrip("\n")
                if not line:
                    continue
                print(line, file=sys.stderr)
                if on_progress_line:
                    on_progress_line(line)
            p.stderr.close()

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()
        stdout_data = proc.stdout.read()
        proc.stdout.close()
        proc.wait()
        stderr_thread.join(timeout=5)

        if proc.returncode != 0:
            raise BenchError(f"llama-bench failed for {quant} (exit {proc.returncode})")
        try:
            rows = json.loads(stdout_data)
        except json.JSONDecodeError as e:
            raise BenchError(
                f"Could not parse llama-bench JSON output for {quant}: {e}\n"
                f"stdout was:\n{stdout_data}"
            ) from e

        for row in rows:
            row["armtune_quant"] = quant
        all_rows.extend(rows)
        if on_quant_done:
            on_quant_done(quant)

    return all_rows


# --------------------------------------------------------------------------
# Mock runner: used for local pipeline testing (e.g. dev machines without
# Arm hardware or a built llama.cpp) via `armtune sweep --mock`. Generates
# synthetic but internally-consistent data matching llama-bench's JSON
# schema, purely so the quantize -> bench -> report -> winner-selection
# pipeline can be exercised end to end before running on real hardware.
# It is NEVER a substitute for real on-device numbers.
# --------------------------------------------------------------------------

# Relative speed multipliers, roughly reflecting real-world llama.cpp
# behavior on Arm (smaller quant = smaller file + faster, up to a point).
_QUANT_PROFILE = {
    "Q4_0":   {"bits": 4.5, "tg_base": 34.0, "pp_base": 210.0},
    "Q4_K_M": {"bits": 4.8, "tg_base": 31.0, "pp_base": 195.0},
    "Q5_K_M": {"bits": 5.6, "tg_base": 26.0, "pp_base": 175.0},
    "Q8_0":   {"bits": 8.5, "tg_base": 19.0, "pp_base": 150.0},
}


def generate_mock_results(
    quant_types: list[str],
    threads: list[int],
    batch_sizes: list[int],
    n_prompt: int,
    n_gen: int,
    n_params: int = 3_212_749_824,
    seed: int = 7,
) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    cpu_info = f"{platform.machine()} (SYNTHETIC MOCK DATA - not a real benchmark)"

    for quant in quant_types:
        profile = _QUANT_PROFILE.get(quant, {"bits": 5.0, "tg_base": 25.0, "pp_base": 180.0})
        model_size = int(n_params * profile["bits"] / 8)

        for threads_n in threads:
            # Diminishing returns / contention past ~8 threads, like real CPUs.
            thread_eff = min(threads_n, 8) + max(0, threads_n - 8) * 0.35
            for batch in batch_sizes:
                batch_eff = 1.0 + (min(batch, 2048) / 2048) * 0.15

                tg_ts = profile["tg_base"] * (thread_eff / 4.0) ** 0.55 * rng.uniform(0.95, 1.05)
                pp_ts = profile["pp_base"] * (thread_eff / 4.0) ** 0.75 * batch_eff * rng.uniform(0.95, 1.05)

                pp_ns = int((n_prompt / max(pp_ts, 0.01)) * 1e9)
                tg_ns = int((n_gen / max(tg_ts, 0.01)) * 1e9)

                common = {
                    "build_commit": "mockbuild",
                    "build_number": 0,
                    "cpu_info": cpu_info,
                    "gpu_info": "",
                    "backends": "CPU",
                    "model_filename": f"mock/{quant}.gguf",
                    "model_type": f"mock-model {quant}",
                    "model_size": model_size,
                    "model_n_params": n_params,
                    "n_batch": batch,
                    "n_threads": threads_n,
                    "n_gpu_layers": 0,
                    "armtune_quant": quant,
                    "armtune_mock": True,
                }
                rows.append({**common, "n_prompt": n_prompt, "n_gen": 0,
                             "avg_ns": pp_ns, "stddev_ns": int(pp_ns * 0.02),
                             "avg_ts": pp_ts, "stddev_ts": pp_ts * 0.02})
                rows.append({**common, "n_prompt": 0, "n_gen": n_gen,
                             "avg_ns": tg_ns, "stddev_ns": int(tg_ns * 0.02),
                             "avg_ts": tg_ts, "stddev_ts": tg_ts * 0.02})
    return rows
