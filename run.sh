#!/usr/bin/env bash
set -euo pipefail

echo "RUN_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "COMPUTE_BACKEND=kubernetes"
echo "GPU_MODEL=NVIDIA RTX PRO 6000 Blackwell"
echo "ALLOCATED_GPU_COUNT=4"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -m pip install --quiet --disable-pip-version-check \
  "transformers==4.53.2" \
  "huggingface_hub==0.33.2" \
  "sentencepiece==0.2.0"
python -u -m reproduce.run_campaign
echo "RUN_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
