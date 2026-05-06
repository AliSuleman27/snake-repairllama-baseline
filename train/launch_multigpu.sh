#!/bin/bash
# launch_multigpu.sh
# ------------------
# Launch DDP training across all available GPUs in this pod.
#
# Usage:
#   bash train/launch_multigpu.sh                # 1 epoch from scratch
#   RESUME=1 bash train/launch_multigpu.sh       # resume from latest checkpoint
#   NUM_GPUS=2 bash train/launch_multigpu.sh     # force a specific GPU count
#   HF_USERNAME=myuser bash train/launch_multigpu.sh   # override HF username
#
# Run this from the repo root (snake-repairllama-baseline/).

set -e

# Auto-detect GPU count if not overridden
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
fi

echo "=============================================="
echo "  Multi-GPU LoRA training launcher"
echo "=============================================="
echo "  GPUs detected:  $NUM_GPUS"
echo "  Resume:         ${RESUME:-(no, fresh start)}"
echo "  HF username:    ${HF_USERNAME:-alisuleman525 (default)}"
echo "  Output dir:     ${LOCAL_OUTPUT_DIR:-./train/output/snake-repairllama-7b-fim-r16}"
echo "=============================================="
echo ""

if [ "$NUM_GPUS" -lt 2 ]; then
    echo "Only $NUM_GPUS GPU detected — falling back to single-GPU launch."
    python3 train/train_multigpu.py
else
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        train/train_multigpu.py
fi

echo ""
echo "=============================================="
echo "  Training run complete"
echo "=============================================="
