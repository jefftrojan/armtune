#!/usr/bin/env bash
# Sets up llama.cpp + ArmTune on an Arm64 (AWS Graviton / any Neoverse) Ubuntu
# instance. Based on Arm's own llama.cpp-on-Graviton learning path:
# https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama-cpu/
#
# Usage: ./scripts/setup_graviton.sh
# Run from the root of the armtune repo, on the Arm64 target itself.

set -euo pipefail

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "warning: this script is intended for an aarch64 (Arm64) host." >&2
  echo "         detected: $(uname -m). Continuing, but you won't get" >&2
  echo "         KleidiAI/SVE/MATMUL_INT8 kernels on non-Arm hardware." >&2
fi

echo "==> Installing build dependencies"
sudo apt update
sudo apt install -y make cmake gcc g++ build-essential python-is-python3 python3-pip python3-venv git

echo "==> Cloning llama.cpp"
if [[ ! -d llama.cpp ]]; then
  git clone https://github.com/ggml-org/llama.cpp
fi

echo "==> Building llama.cpp with -mcpu=native"
# No special "Arm mode" flag needed: llama.cpp auto-detects and uses Arm's
# contributed GEMV/GEMM kernels (Neon / SVE / MATMUL_INT8, aka the KleidiAI
# optimizations) for supported quant types (Q4_0-family, Q8_0) when built
# natively on Arm. -mcpu=native lets the compiler target this exact CPU
# (e.g. Neoverse V2 on Graviton4).
cd llama.cpp
mkdir -p build && cd build
cmake .. -DCMAKE_CXX_FLAGS="-mcpu=native" -DCMAKE_C_FLAGS="-mcpu=native"
cmake --build . -v --config Release -j "$(nproc)"
cd ../..

echo "==> Verifying the build reports Arm kernel support"
./llama.cpp/build/bin/llama-cli --version || true
echo "    (Look for NEON/SVE/MATMUL_INT8=1 in llama-cli's startup banner when you run a model.)"

echo "==> Setting up Python venv + installing armtune"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -e . -q
pip install huggingface_hub -q

echo ""
echo "==> Done. Next steps:"
echo "  1. source .venv/bin/activate"
echo "  2. Download a base (unquantized f16) GGUF model, e.g.:"
echo "       hf download <org>/<model>-GGUF <model>-f16.gguf --local-dir models"
echo "  3. Run the sweep:"
echo "       armtune sweep --base-model models/<model>-f16.gguf --llama-bin-dir llama.cpp/build/bin"
echo "  4. Check results/report.md for the winning config."
