> This is a real sample report from an actual run of the [GitHub Actions
> workflow](../.github/workflows/sweep.yml) (`armtune sweep`, no `--mock`)
> against `Qwen/Qwen2.5-0.5B-Instruct-GGUF` on a real arm64-hosted GitHub
> Actions runner (Azure Cobalt 100, Neoverse-based, `aarch64`). Numbers will
> vary by model, instance type, and thread range swept — see the setup
> instructions in the main README to reproduce on your own target.

# ArmTune sweep report

- Model: `models/qwen2.5-0.5b-instruct-fp16.gguf`
- Host CPU: `aarch64`
- Generated: 2026-08-12T23:30:57+00:00

## Recommended configs

- **Fastest throughput:** `Q4_0`, 4 threads, batch 512 → **116.98 tok/s** generation, 598.3 ms TTFT, 403.2 MiB
- **Best value (smallest within 10% of fastest):** `Q4_0`, 4 threads, batch 512 → 116.98 tok/s, **403.2 MiB**
- **Lowest time-to-first-token:** `Q4_0`, 4 threads, batch 512 → **598.3 ms TTFT**

## Full sweep (sorted by generation throughput)

| quant | threads | batch | size (MiB) | TTFT (ms) | prompt t/s | gen t/s |
|---|---:|---:|---:|---:|---:|---:|
| Q4_0 | 4 | 512 | 403.2 | 598.3 | 427.88 | 116.98 |
| Q4_0 | 4 | 1024 | 403.2 | 599.88 | 426.75 | 115.57 |
| Q8_0 | 4 | 1024 | 638.7 | 674.38 | 379.74 | 113.56 |
| Q8_0 | 4 | 512 | 638.7 | 664.85 | 385.05 | 103.06 |
| Q4_K_M | 4 | 1024 | 463.0 | 1827.5 | 140.08 | 78.35 |
| Q4_K_M | 4 | 512 | 463.0 | 1829.09 | 139.96 | 77.96 |
| Q4_0 | 2 | 512 | 403.2 | 1048.97 | 244.05 | 67.82 |
| Q8_0 | 2 | 1024 | 638.7 | 1176.55 | 217.59 | 65.97 |
| Q4_0 | 2 | 1024 | 403.2 | 1050.93 | 243.59 | 65.93 |
| Q8_0 | 2 | 512 | 638.7 | 1190.99 | 214.95 | 57.3 |
| Q4_K_M | 2 | 512 | 463.0 | 3496.78 | 73.21 | 43.75 |
| Q4_K_M | 2 | 1024 | 463.0 | 3501.48 | 73.11 | 42.59 |
| Q4_0 | 1 | 512 | 403.2 | 2061.25 | 124.2 | 38.78 |
| Q4_0 | 1 | 1024 | 403.2 | 2060.18 | 124.26 | 38.21 |
| Q8_0 | 1 | 1024 | 638.7 | 2341.17 | 109.35 | 35.79 |
| Q8_0 | 1 | 512 | 638.7 | 2319.72 | 110.36 | 34.74 |
| Q4_K_M | 1 | 1024 | 463.0 | 6956.41 | 36.8 | 25.01 |
| Q4_K_M | 1 | 512 | 463.0 | 6970.12 | 36.73 | 24.63 |

## Best config per quantization level

| quant | threads | batch | size (MiB) | TTFT (ms) | gen t/s |
|---|---:|---:|---:|---:|---:|
| Q4_0 | 4 | 512 | 403.2 | 598.3 | 116.98 |
| Q8_0 | 4 | 1024 | 638.7 | 674.38 | 113.56 |
| Q4_K_M | 4 | 1024 | 463.0 | 1827.5 | 78.35 |

Note: this run only swept `--threads 1,2,4`, so 4 threads (the runner's full
core count) is both the winner and the top of the tested range — the sweep
hadn't found a throughput ceiling yet, just the edge of what was tested. The
CI workflow now sweeps a wider thread range for exactly this reason.

Also notable: `Q4_K_M` is consistently 2-3x slower than `Q4_0` and `Q8_0` at
every thread count (e.g. 24.6 vs 38.8/34.7 tok/s at 1 thread), well outside
what quant size alone would predict — a sign it isn't hitting the same
optimized Arm kernel path as the other two on this CPU.
