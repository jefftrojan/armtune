"""ArmTune: an auto-tuning agent for LLM inference serving configs on Arm64 CPUs.

ArmTune wraps llama.cpp's llama-quantize and llama-bench to automatically sweep
quantization level, thread count, and batch size for a given model on a given
Arm64 (Neoverse / Graviton) host, then recommends the winning serving config
based on measured throughput, time-to-first-token, and on-disk model size.
"""

__version__ = "0.1.0"
