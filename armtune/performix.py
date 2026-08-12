"""Best-effort Arm Performix (`apx`) integration.

Arm Performix is normally driven from a desktop host (GUI or `apx` CLI) that
connects to an Arm Linux target over SSH to collect hardware performance
counters (top-down methodology, cache/branch stats, etc.) - it is a separate
tool from llama-bench's own self-reported tokens/sec.

ArmTune's sweep loop is designed to run *on* the Arm target itself (where
llama.cpp is built), so this module is a thin, best-effort convenience: if
an `apx` binary is already on PATH there (e.g. installed per Arm's install
guide), we shell out to capture a version check and, optionally, wrap the
winning config's benchmark command so its profiling log sits next to the
rest of the sweep's artifacts. It never blocks or fails the sweep - if `apx`
isn't set up, ArmTune just says so and moves on.

For full hardware-counter analysis, the documented path is to install Arm
Performix on your desktop and point it at the Graviton/Neoverse target over
SSH per https://learn.arm.com/install-guides/performix/ - that GUI/SSH
workflow is out of scope for this automated wrapper.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def is_available() -> bool:
    return shutil.which("apx") is not None


def try_profile_winner(cmd: list[str], out_dir: Path) -> str:
    """Best-effort: run `apx run -- <cmd>` and save its output.

    Returns a short status string for the report; never raises.
    """
    if not is_available():
        return (
            "apx not found on PATH - skipped. Install Arm Performix and see "
            "https://learn.arm.com/install-guides/performix/ to profile this "
            "config with hardware performance counters."
        )

    log_path = out_dir / "performix_winner.log"
    try:
        proc = subprocess.run(
            ["apx", "run", "--"] + cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        log_path.write_text(
            f"$ apx run -- {' '.join(cmd)}\n\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n"
        )
        if proc.returncode == 0:
            return f"Performix profile captured -> {log_path.name}"
        return (
            f"apx exited with code {proc.returncode} (see {log_path.name}). "
            "Recipe/target flags may need adjusting for your apx version - "
            "see the Arm Performix User Guide."
        )
    except Exception as e:  # noqa: BLE001 - best-effort, never fail the sweep
        log_path.write_text(f"apx invocation raised: {e}\n")
        return f"apx invocation failed ({e}); see {log_path.name}."
