# ArmTune

**An auto-tuning agent that finds the fastest LLM inference serving config for your specific Arm64 CPU.**

## Project overview

Getting good LLM inference performance out of an Arm64 cloud CPU (AWS Graviton, Azure Cobalt, Google Axion) isn't just "does it run" — the right quantization level, thread count, and batch size for *your* instance type and *your* model interact in ways that aren't obvious ahead of time, and the search space is annoying to sweep by hand.

ArmTune automates that search. Point it at a base model and your llama.cpp build, and it:

1. Quantizes the model to several precisions (Q4_0, Q4_K_M, Q8_0 by default — the formats Arm has published optimized Neon/SVE/MATMUL_INT8 kernels for).
2. Benchmarks every (quant × thread-count × batch-size) combination on the host using llama.cpp's own `llama-bench`.
3. Ranks every config by measured generation throughput, time-to-first-token, and on-disk size, and picks winners for "fastest," "lowest latency," and "best value."
4. Writes a report and a ready-to-run `llama-server` launch command for the winning config — so the output isn't just a benchmark, it's a decision.

Most ad hoc optimization write-ups show *one* before/after number for *one* config someone happened to pick. ArmTune turns "which config is fastest on this specific Arm chip" into a repeatable, automated, one-command answer — and the tool itself is reusable by anyone deploying an LLM on Arm64.

The same sweep-and-recommend approach generalizes to on-device Arm CPUs (mobile/edge) too, since llama.cpp's Arm kernels and the thread/batch/quant trade-off space are conceptually the same problem there — that's a natural direction for future work.

## Functionality / output

Running `armtune sweep` on an Arm64 target produces, in `results/`:

- `report.md` — human-readable comparison table of every config tested, recommended winners (fastest throughput, lowest TTFT, best size/speed trade-off, and best config per quant level), and a baseline-vs-tuned comparison against the config an untuned deploy would likely land on (full thread count, least-compressed quant, largest batch tested).
- `report.html` — the same data as a self-contained, chartable page: throughput vs. thread count per quant, baseline-vs-tuned bars, a sortable full-results table, and (if run) the concurrency and cost charts. No dependencies, no CDN — open it straight in a browser.
- `results.csv` — the same data, flat, for spreadsheets or further analysis.
- `results_raw.json` — raw `llama-bench` output for every test point.
- `recommended_launch.sh` — a ready-to-run `llama-server` command using the winning config.

Example (synthetic, generated with `--mock` for illustration — see [`examples/example_report.md`](examples/example_report.md) for a full sample; **real numbers must come from a run on Arm hardware**):

| quant | threads | batch | size (MiB) | TTFT (ms) | gen t/s |
|---|---:|---:|---:|---:|---:|
| Q4_0 | 8 | 2048 | 1723.5 | 1314.4 | 47.6 |
| Q4_K_M | 8 | 512 | 1838.4 | 1576.3 | 47.6 |
| Q8_0 | 8 | 512 | 3255.4 | 2045.7 | 28.0 |

Optional flags:

- `--performix` wraps the winning config's `llama-bench` run with Arm Performix's `apx` CLI (if installed) for hardware-performance-counter profiling on top of llama-bench's own timing. For full top-down analysis, running Arm Performix's GUI or `apx` from a separate host against the target over SSH (per [Arm's install guide](https://learn.arm.com/install-guides/performix/)) is the more complete path — ArmTune's built-in hook is a convenience, not a replacement for it.
- `--instance-cost-per-hour 0.0672` adds a $/1M generated tokens column to the baseline-vs-tuned comparison, using your instance's on-demand hourly price (generation-only; ignores prompt-processing time).
- `--concurrency 1,4,8` measures aggregate serving throughput for the winning config under concurrent load: launches `llama-server` once and fires N simultaneous `/completion` requests at each level, appending a "Concurrent serving throughput" section to `report.md` and a `concurrency_raw.json`. `llama-bench` (the main sweep) only ever tests one request stream at a time, which understates how batch size matters for a real multi-request serving workload — this fills that gap for the single config that won the sweep, rather than re-running the full grid.
- `--serve` starts a local live-progress dashboard (`http://127.0.0.1:8877`, auto-opened in your browser) while the sweep runs — showing which quant is being quantized/benchmarked, `llama-bench`'s own progress lines as they happen, and a running step counter — then becomes the full chartable `report.html` the moment the sweep finishes. Stays up afterward so you can browse and export (`--port`/`--no-open` to change the port or skip auto-opening). Standard library only — no new dependency.

## Setup instructions

### Run it on an Arm64 instance (Graviton / Cobalt / Axion)

Requires an Arm64 Linux host (tested against AWS Graviton-class instances, Ubuntu 22.04/24.04, 4+ cores, 8GB+ RAM for a ~3-8B model).

```bash
git clone <this-repo-url> armtune
cd armtune

# Installs build deps, clones + builds llama.cpp natively (-mcpu=native
# picks up this host's Neon/SVE/MATMUL_INT8 support automatically), and
# installs the armtune CLI into a local venv.
./scripts/setup_graviton.sh

source .venv/bin/activate

# Run the full sweep against any Hugging Face model
armtune sweep \
  --hf-repo <org>/<model>-GGUF --hf-file <model>-f16.gguf \
  --llama-bin-dir llama.cpp/build/bin \
  --quants Q4_0,Q4_K_M,Q8_0 \
  --batch 512,2048

# See the results
cat results/report.md
```

### Using your own model

`armtune sweep` accepts a base model three ways:

- `--base-model models/<model>-f16.gguf` — a GGUF file you already have locally.
- `--hf-repo <org>/<model>-GGUF --hf-file <model>-f16.gguf` — downloads a specific GGUF file from a Hugging Face repo that already publishes one.
- `--hf-repo <org>/<model>` (no `--hf-file`) — downloads a plain (safetensors) Hugging Face repo and converts it to an f16 GGUF with llama.cpp's own `convert_hf_to_gguf.py`. This needs a llama.cpp *source* checkout (`--llama-src-dir`, default `llama.cpp` — `setup_graviton.sh` clones one) and that script's own Python requirements installed: `pip install -r llama.cpp/requirements.txt`.

Any architecture llama.cpp itself supports works here — ArmTune doesn't hardcode a model list.

### Try the pipeline without an Arm host first

`armtune sweep --mock` runs the full quantize→bench→report→winner-selection pipeline with synthetic data (clearly labeled as such in every output artifact), so you can sanity-check the tool on any machine before running it for real on Arm hardware:

```bash
pip install -e .
armtune sweep --mock --quants Q4_0,Q4_K_M,Q8_0 --threads 2,4,8 --batch 512,2048
```

## Key optimizations

- **Arm-specific optimization**: builds llama.cpp with `-mcpu=native` to engage Arm's contributed Neon/SVE/MATMUL_INT8 GEMV/GEMM kernels for Q4_0-family and Q8_0 quantization on Graviton2/3/4-class CPUs.
- **Model size**: reports on-disk size per quantization level so the size/speed trade-off is explicit, not guessed.
- **Model speed / inference server speed**: measures and ranks generation throughput (tok/s) and time-to-first-token across the full config sweep.
- **Developer experience**: turns a manual, error-prone benchmarking chore into a single command with a decision-ready report and a ready-to-run launch script for the winning config.

## Repository layout

```
armtune/
├── armtune/                # CLI package (quantize, bench, report, htmlreport, hfmodel, serve, liveserver, performix wrapper)
├── scripts/
│   └── setup_graviton.sh   # Arm64 host bootstrap (deps, llama.cpp build, venv)
├── examples/
│   └── example_report.md   # Sample report output
├── tests/
│   └── test_sweep_logic.py
├── LICENSE                 # MIT
└── pyproject.toml
```

## License

MIT — see [LICENSE](LICENSE).
