"""ArmTune CLI entrypoint.

`armtune sweep` automatically quantizes a base GGUF model to multiple
precisions, benchmarks every (quant x thread-count x batch-size) combination
on the current host with llama.cpp's llama-bench, and writes out a ranked
report plus a recommended launch command for the fastest config actually
measured on *this* machine.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import webbrowser
from pathlib import Path

from . import __version__, bench, hfmodel, performix, quantize, report, serve
from .livestate import SweepState
from .liveserver import LiveServer

DEFAULT_QUANTS = ["Q4_0", "Q4_K_M", "Q8_0"]


def _default_threads() -> list[int]:
    n = os.cpu_count() or 4
    candidates = sorted({max(1, n // 4), max(1, n // 2), n})
    return candidates


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="armtune",
        description="Auto-tune LLM inference serving configs for Arm64 CPUs.",
    )
    p.add_argument("--version", action="version", version=f"armtune {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sweep = sub.add_parser("sweep", help="Quantize, benchmark, and rank configs.")
    sweep.add_argument(
        "--base-model", type=Path, default=None,
        help="Path to an unquantized (f16/f32) base .gguf file. "
             "Required unless --mock or --hf-repo is given.",
    )
    sweep.add_argument(
        "--hf-repo", type=str, default=None,
        help="Hugging Face repo to pull the base model from, instead of --base-model.",
    )
    sweep.add_argument(
        "--hf-file", type=str, default=None,
        help="Specific GGUF filename within --hf-repo, if that repo already publishes "
             "one (e.g. an '-GGUF' repo). Omit to download+convert a plain "
             "(safetensors) HF model repo instead -- requires --llama-src-dir.",
    )
    sweep.add_argument(
        "--llama-src-dir", type=Path, default=Path("llama.cpp"),
        help="llama.cpp source checkout containing convert_hf_to_gguf.py, used only "
             "when --hf-repo is given without --hf-file (default: llama.cpp).",
    )
    sweep.add_argument(
        "--quants", type=str, default=",".join(DEFAULT_QUANTS),
        help=f"Comma-separated quant types to test (default: {','.join(DEFAULT_QUANTS)}). "
             "Arm has published optimized kernels for Q4_0-family and Q8_0 in particular.",
    )
    sweep.add_argument(
        "--threads", type=str, default=None,
        help="Comma-separated thread counts to test (default: derived from nproc).",
    )
    sweep.add_argument(
        "--batch", type=str, default="512,2048",
        help="Comma-separated batch sizes to test (default: 512,2048).",
    )
    sweep.add_argument("--n-prompt", type=int, default=512, help="Prompt tokens per pp test (default: 512).")
    sweep.add_argument("--n-gen", type=int, default=128, help="Tokens to generate per tg test (default: 128).")
    sweep.add_argument("--repetitions", type=int, default=3, help="Repeats per test point, averaged (default: 3).")
    sweep.add_argument(
        "--llama-bin-dir", type=Path, default=Path("llama.cpp/build/bin"),
        help="Directory containing llama-quantize / llama-bench (default: llama.cpp/build/bin).",
    )
    sweep.add_argument("--model-cache", type=Path, default=Path("models"), help="Where quantized .gguf files are cached.")
    sweep.add_argument("--out-dir", type=Path, default=Path("results"), help="Where reports are written.")
    sweep.add_argument("--performix", action="store_true", help="Best-effort: profile the winning config with `apx` if installed.")
    sweep.add_argument(
        "--instance-cost-per-hour", type=float, default=None,
        help="On-demand $/hr for this instance type (e.g. current Graviton/Cobalt/Axion "
             "pricing). If given, the report includes $/1M generated tokens for the "
             "baseline and tuned configs.",
    )
    sweep.add_argument(
        "--concurrency", type=str, default=None,
        help="Comma-separated concurrent-request counts to test against the winning "
             "config via llama-server (e.g. 1,4,8). llama-bench above only measures "
             "one request stream at a time; this shows how the winner holds up under "
             "concurrent load. Off by default (adds a llama-server start/stop cycle).",
    )
    sweep.add_argument(
        "--mock", action="store_true",
        help="Generate synthetic benchmark data instead of running llama.cpp. "
             "For testing the armtune pipeline itself on machines without a "
             "built llama.cpp / Arm hardware. NOT valid as real performance results.",
    )
    sweep.add_argument(
        "--serve", action="store_true",
        help="Start a local live-progress dashboard (http://127.0.0.1:<port>) while "
             "the sweep runs, which becomes the full chartable report when it finishes. "
             "Stays running after completion until Ctrl-C.",
    )
    sweep.add_argument("--port", type=int, default=8877, help="Port for --serve's dashboard (default: 8877).")
    sweep.add_argument("--no-open", action="store_true", help="With --serve, don't auto-open the dashboard in a browser.")

    sub.add_parser("version", help="Print the armtune version.")
    return p


def cmd_sweep(args: argparse.Namespace) -> int:
    quant_types = [q.strip() for q in args.quants.split(",") if q.strip()]
    threads = (
        [int(t) for t in args.threads.split(",")] if args.threads else _default_threads()
    )
    batch_sizes = [int(b) for b in args.batch.split(",")]

    cpu_label = platform.processor() or platform.machine()
    is_arm = platform.machine().lower() in ("aarch64", "arm64")

    state: SweepState | None = None
    live_server: LiveServer | None = None
    if args.serve:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        state = SweepState()
        live_server = LiveServer(state, out_dir=args.out_dir, port=args.port)
        live_server.start()
        print(f"[armtune] live dashboard: {live_server.url}")
        if not args.no_open:
            try:
                webbrowser.open(live_server.url)
            except Exception:
                pass

    if not is_arm and not args.mock:
        print(
            f"[armtune] warning: host arch is '{platform.machine()}', not aarch64/arm64. "
            "ArmTune is designed to sweep on the Arm target itself. Continuing anyway, "
            "but these numbers won't reflect Arm-specific (KleidiAI/SVE/MATMUL_INT8) kernels.",
            file=sys.stderr,
        )

    if args.mock:
        print("[armtune] --mock set: generating synthetic demo data, not running llama.cpp.")
        if state:
            state.set_total_steps(1)
            state.set_status("generating_mock", "Generating synthetic demo data...")
        rows = bench.generate_mock_results(
            quant_types=quant_types, threads=threads, batch_sizes=batch_sizes,
            n_prompt=args.n_prompt, n_gen=args.n_gen,
        )
        if state:
            state.advance("Synthetic data generated.")
        model_label = "mock-model (synthetic, --mock)"
    else:
        base_model = args.base_model
        if base_model is None and args.hf_repo:
            try:
                if state:
                    state.set_status("downloading", f"Downloading {args.hf_repo}...")
                if args.hf_file:
                    base_model = hfmodel.download_gguf(args.hf_repo, args.hf_file, args.model_cache)
                else:
                    base_model = hfmodel.download_and_convert(
                        args.hf_repo, args.model_cache, args.llama_src_dir
                    )
            except hfmodel.HFModelError as e:
                if state:
                    state.set_error(str(e))
                print(f"error: {e}", file=sys.stderr)
                return 2

        if base_model is None:
            msg = "--base-model or --hf-repo is required unless --mock is set"
            if state:
                state.set_error(msg)
            print(f"error: {msg}", file=sys.stderr)
            return 2
        if not base_model.exists():
            msg = f"base model not found: {base_model}"
            if state:
                state.set_error(msg)
            print(f"error: {msg}", file=sys.stderr)
            return 2

        quantize_bin = quantize.find_binary(args.llama_bin_dir, "llama-quantize")
        bench_bin = quantize.find_binary(args.llama_bin_dir, "llama-bench")

        if state:
            state.set_total_steps(len(quant_types) * 2)  # quantize + bench, per quant
            state.set_status("quantizing", f"Quantizing to {', '.join(quant_types)}...")

        print(f"[armtune] quantizing {base_model} -> {quant_types}")
        try:
            quantized = quantize.ensure_quantized(
                base_model, quant_types, args.model_cache, quantize_bin,
                on_quant_start=(lambda q: state.set_status("quantizing", f"Quantizing {q}...")) if state else None,
                on_quant_done=(lambda q: state.advance()) if state else None,
            )

            print(f"[armtune] sweeping threads={threads} batch={batch_sizes} on {len(quantized)} quant variants")
            rows = bench.run_llama_bench(
                bench_bin=bench_bin, model_paths=quantized, threads=threads,
                batch_sizes=batch_sizes, n_prompt=args.n_prompt, n_gen=args.n_gen,
                repetitions=args.repetitions,
                on_quant_start=(lambda q: state.set_status("benchmarking", f"Benchmarking {q}...")) if state else None,
                on_quant_done=(lambda q: state.advance()) if state else None,
                on_progress_line=state.add_progress_line if state else None,
            )
        except (quantize.QuantizeError, bench.BenchError) as e:
            if state:
                state.set_error(str(e))
            print(f"error: {e}", file=sys.stderr)
            return 1
        model_label = str(base_model)

    results = report.merge_rows(rows)
    if not results:
        msg = "no complete pp+tg result pairs were produced"
        if state:
            state.set_error(msg)
        print(f"error: {msg}", file=sys.stderr)
        return 1
    winners = report.pick_winners(results)
    if state:
        state.set_status("writing_report", "Writing report...")
    paths = report.write_artifacts(
        rows, results, winners, args.out_dir, model_label, cpu_label,
        cost_per_hour=args.instance_cost_per_hour,
    )

    if args.performix:
        if args.mock:
            print("[armtune] performix: skipped (no real binary/hardware under --mock)")
        else:
            winner = winners["fastest_throughput"]
            profile_cmd = [
                str(bench_bin), "-m", winner.model_path,
                "-t", str(winner.threads), "-b", str(winner.batch),
                "-p", str(args.n_prompt), "-n", str(args.n_gen), "-r", "1",
            ]
            status = performix.try_profile_winner(cmd=profile_cmd, out_dir=args.out_dir)
            print(f"[armtune] performix: {status}")

    if args.concurrency:
        levels = [int(c) for c in args.concurrency.split(",") if c.strip()]
        winner = winners["fastest_throughput"]
        print(f"[armtune] measuring concurrent serving throughput at {levels} for the winning config")
        conc_results = None
        if args.mock:
            conc_results = serve.generate_mock_concurrency(levels, winner.tg_tokens_per_s)
        else:
            try:
                server_bin = quantize.find_binary(args.llama_bin_dir, "llama-server")
                conc_results = serve.bench_concurrency(
                    server_bin=server_bin, model_path=winner.model_path,
                    threads=winner.threads, batch=winner.batch,
                    concurrency_levels=levels, n_predict=args.n_gen,
                )
            except (quantize.QuantizeError, serve.ServeError) as e:
                print(f"[armtune] concurrency benchmark skipped: {e}", file=sys.stderr)
        if conc_results:
            report.append_concurrency(args.out_dir, conc_results)
            report.write_html_report(
                args.out_dir, results, winners, model_label, cpu_label,
                cost_per_hour=args.instance_cost_per_hour, concurrency_results=conc_results,
            )
            print(f"[armtune] concurrency results appended to {paths['markdown']} and {paths['html']}")

    w = winners["fastest_throughput"]
    b = winners["baseline"]
    print("\n=== ArmTune sweep complete ===")
    print(f"Fastest config: {w.quant}, {w.threads} threads, batch {w.batch} "
          f"-> {w.tg_tokens_per_s} tok/s gen, {w.ttft_ms} ms TTFT, {w.model_size_mib} MiB")
    if b.tg_tokens_per_s and (b.quant, b.threads, b.batch) != (w.quant, w.threads, w.batch):
        speedup_pct = (w.tg_tokens_per_s - b.tg_tokens_per_s) / b.tg_tokens_per_s * 100
        print(f"vs. untuned baseline ({b.quant}, {b.threads} threads, batch {b.batch}): "
              f"{speedup_pct:.1f}% faster generation throughput")
    print(f"Report:  {paths['markdown']}")
    print(f"HTML:    {paths['html']}")
    print(f"CSV:     {paths['csv']}")
    print(f"Launch:  {paths['launch_script']}")

    if state:
        state.set_status("done", "Sweep complete.")
    if live_server:
        print(f"\n[armtune] live dashboard still running at {live_server.url} -- press Ctrl-C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[armtune] stopping live server.")
        finally:
            live_server.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(f"armtune {__version__}")
        return 0
    if args.command == "sweep":
        return cmd_sweep(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
