"""Wraps llama.cpp's llama-quantize to produce the GGUF variants under test."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable


class QuantizeError(RuntimeError):
    pass


def find_binary(bin_dir: Path, name: str) -> Path:
    """Locate a llama.cpp binary, checking bin_dir first, then PATH."""
    candidate = bin_dir / name
    if candidate.exists():
        return candidate
    on_path = shutil.which(name)
    if on_path:
        return Path(on_path)
    raise QuantizeError(
        f"Could not find '{name}'. Pass --llama-bin-dir pointing at your "
        f"llama.cpp build/bin directory (see scripts/setup_graviton.sh)."
    )


def ensure_quantized(
    base_model: Path,
    quant_types: list[str],
    cache_dir: Path,
    quantize_bin: Path,
    on_quant_start: Callable[[str], None] | None = None,
    on_quant_done: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Produce (or reuse cached) GGUF files for each requested quant type.

    Returns a dict mapping quant type -> path to the quantized .gguf file.
    on_quant_start/on_quant_done let callers (e.g. the --serve live
    dashboard) track progress without this module knowing anything about it.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    for quant in quant_types:
        if on_quant_start:
            on_quant_start(quant)

        stem = base_model.stem
        target = cache_dir / f"{stem}-{quant}.gguf"

        if target.exists() and target.stat().st_size > 0:
            out[quant] = target
            if on_quant_done:
                on_quant_done(quant)
            continue

        cmd = [str(quantize_bin), str(base_model), str(target), quant]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not target.exists():
            raise QuantizeError(
                f"llama-quantize failed for {quant}:\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
            )
        out[quant] = target
        if on_quant_done:
            on_quant_done(quant)

    return out
