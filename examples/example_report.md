> This is a sample report generated with `armtune sweep --mock` to show the
> report format. It is synthetic demo data, not a real benchmark — real
> numbers come from running `armtune sweep` on an actual Arm64
> (Graviton) host per the setup instructions in the main README.

# ArmTune sweep report

- Model: `mock-model (synthetic, --mock)`
- Host CPU: `x86_64`
- Generated: 2026-08-12T22:20:20+00:00

> **SYNTHETIC DEMO DATA.** This report was generated with `--mock` for pipeline testing and does not reflect real hardware performance. Re-run without `--mock` on an Arm64 target for real numbers.


## Recommended configs

- **Fastest throughput:** `Q4_0`, 8 threads, batch 2048 → **47.64 tok/s** generation, 1314.41 ms TTFT, 1723.5 MiB
- **Best value (smallest within 10% of fastest):** `Q4_0`, 8 threads, batch 2048 → 47.64 tok/s, **1723.5 MiB**
- **Lowest time-to-first-token:** `Q4_0`, 8 threads, batch 2048 → **1314.41 ms TTFT**

## Full sweep (sorted by generation throughput)

| quant | threads | batch | size (MiB) | TTFT (ms) | prompt t/s | gen t/s |
|---|---:|---:|---:|---:|---:|---:|
| Q4_0 | 8 | 2048 | 1723.5 | 1314.41 | 389.53 | 47.64 |
| Q4_K_M | 8 | 512 | 1838.4 | 1576.26 | 324.82 | 47.55 |
| Q4_0 | 8 | 512 | 1723.5 | 1406.63 | 363.99 | 47.48 |
| Q4_K_M | 8 | 2048 | 1838.4 | 1386.75 | 369.21 | 47.01 |
| Q4_0 | 4 | 512 | 1723.5 | 2381.96 | 214.95 | 34.12 |
| Q4_0 | 4 | 2048 | 1723.5 | 2118.51 | 241.68 | 32.5 |
| Q4_K_M | 4 | 512 | 1838.4 | 2422.29 | 211.37 | 31.4 |
| Q4_K_M | 4 | 2048 | 1838.4 | 2307.0 | 221.93 | 31.24 |
| Q8_0 | 8 | 512 | 3255.4 | 2045.66 | 250.29 | 27.95 |
| Q8_0 | 8 | 2048 | 3255.4 | 1818.32 | 281.58 | 26.59 |
| Q4_0 | 2 | 2048 | 1723.5 | 3724.8 | 137.46 | 23.57 |
| Q4_0 | 2 | 512 | 1723.5 | 4095.15 | 125.03 | 22.81 |
| Q4_K_M | 2 | 512 | 1838.4 | 4121.47 | 124.23 | 21.01 |
| Q4_K_M | 2 | 2048 | 1838.4 | 3949.11 | 129.65 | 20.38 |
| Q8_0 | 4 | 2048 | 3255.4 | 3006.48 | 170.3 | 19.26 |
| Q8_0 | 4 | 512 | 3255.4 | 3263.33 | 156.89 | 18.39 |
| Q8_0 | 2 | 2048 | 3255.4 | 4838.79 | 105.81 | 12.73 |
| Q8_0 | 2 | 512 | 3255.4 | 5752.91 | 89.0 | 12.52 |

## Best config per quantization level

| quant | threads | batch | size (MiB) | TTFT (ms) | gen t/s |
|---|---:|---:|---:|---:|---:|
| Q4_0 | 8 | 2048 | 1723.5 | 1314.41 | 47.64 |
| Q4_K_M | 8 | 512 | 1838.4 | 1576.26 | 47.55 |
| Q8_0 | 8 | 512 | 3255.4 | 2045.66 | 27.95 |
