"""Fetches a base model from Hugging Face Hub for `armtune sweep --hf-repo`.

Two paths, both by shelling out to the `hf` CLI (huggingface_hub's official
CLI) rather than importing huggingface_hub as a Python dependency -- armtune
itself is deliberately dependency-free (see pyproject.toml).

  - --hf-file given: the repo already publishes a ready-to-use GGUF file
    (e.g. an "-GGUF" repo); download that file directly.
  - --hf-file omitted: the repo is a plain (safetensors) model repo;
    download the whole snapshot and convert it to an f16 GGUF with
    llama.cpp's own convert_hf_to_gguf.py.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


class HFModelError(RuntimeError):
    pass


def _find_hf_cli() -> str:
    hf = shutil.which("hf")
    if hf:
        return hf
    raise HFModelError(
        "Could not find the 'hf' CLI on PATH. Install it with "
        "`pip install -U huggingface_hub` (the older `huggingface-cli` "
        "is deprecated and no longer works)."
    )


def download_gguf(repo: str, filename: str, cache_dir: Path) -> Path:
    """Download a single, already-quantized GGUF file from a HF repo."""
    hf = _find_hf_cli()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cmd = [hf, "download", repo, filename, "--local-dir", str(cache_dir)]
    print(f"[armtune] downloading {repo}/{filename}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HFModelError(f"hf download failed:\ncmd: {' '.join(cmd)}\nstderr: {proc.stderr}")
    target = cache_dir / filename
    if not target.exists():
        raise HFModelError(f"hf download reported success but {target} is missing")
    return target


def download_and_convert(repo: str, cache_dir: Path, llama_src_dir: Path) -> Path:
    """Download a full HF model repo and convert it to an f16 GGUF.

    Requires a llama.cpp *source* checkout (not just build/bin) containing
    convert_hf_to_gguf.py, plus that script's own Python requirements
    (llama.cpp/requirements.txt: torch, transformers, gguf, etc.) installed
    separately -- those are conversion-time deps, not armtune's.
    """
    hf = _find_hf_cli()
    convert_script = llama_src_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise HFModelError(
            f"convert_hf_to_gguf.py not found at {convert_script}. Pass "
            "--llama-src-dir pointing at a llama.cpp source checkout "
            "(scripts/setup_graviton.sh clones one to ./llama.cpp), or "
            "pass --hf-file if this repo already publishes a GGUF."
        )

    snapshot_dir = cache_dir / repo.replace("/", "__")
    cmd = [hf, "download", repo, "--local-dir", str(snapshot_dir)]
    print(f"[armtune] downloading {repo} for conversion")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HFModelError(f"hf download failed:\ncmd: {' '.join(cmd)}\nstderr: {proc.stderr}")

    out_file = cache_dir / f"{repo.split('/')[-1]}-f16.gguf"
    convert_cmd = [
        sys.executable, str(convert_script), str(snapshot_dir),
        "--outfile", str(out_file), "--outtype", "f16",
    ]
    print(f"[armtune] converting {repo} -> f16 GGUF (this can take a while for larger models)")
    proc = subprocess.run(convert_cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_file.exists():
        raise HFModelError(
            f"convert_hf_to_gguf.py failed:\ncmd: {' '.join(convert_cmd)}\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}\n\n"
            "Make sure llama.cpp's conversion requirements are installed: "
            f"pip install -r {llama_src_dir}/requirements.txt"
        )
    return out_file
