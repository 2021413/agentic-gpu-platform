#!/usr/bin/env bash
# Deploy the GPU worker and run the real end-to-end test.
#
# EU-FR-1 had no capacity at all when this was written — neither secure nor
# community, across six GPU types — and the weights volume is locked to that
# region, so there was no pod to test against. Availability fluctuates; this
# script is the whole attempt in one command so retrying costs nothing but the
# time it takes to answer.
#
#   ./scripts/retry-gpu-test.sh
#
# It creates nothing if no GPU is free: the deployer fails before a pod exists,
# so an unsuccessful run bills nothing.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE/gpu-worker"

IMAGE=${IMAGE:-ghcr.io/2021413/agentic-gpu-platform/agentic-gpu-worker:1.0.0}
VOLUME=${VOLUME:-6pbjmoov04}
REGION=${REGION:-EU-FR-1}

set -a; . ./.env; set +a

# Deliberately contradictory on two points, because both reconciliations have
# only ever been tested against a double:
#   - SERVED_MODEL_NAME renames the model, so the agent must adopt the served
#     name or the control plane sends one vLLM answers 404 to.
#   - MAX_MODEL_LEN is far below the model's native context, so the agent must
#     advertise what is served rather than what it was configured with.
exec .venv/bin/python -m tools.runpod_deployer.cli deploy \
  --name agentic-real-test \
  --image "$IMAGE" \
  --gpu-type "NVIDIA H100 80GB HBM3" \
  --gpu-type "NVIDIA H100 PCIe" \
  --gpu-type "NVIDIA H100 NVL" \
  --gpu-type "NVIDIA H200" \
  --gpu-type "NVIDIA L40S" \
  --data-center "$REGION" \
  --network-volume "$VOLUME" \
  --expose tcp \
  --max-model-len 32768 \
  --served-model-name qwen3-coder \
  "$@"
