#!/bin/bash
# train_ann.sh
#
# Wrapper executed by HTCondor for each ANN training job.
# Usage: train_ann.sh <task_idx>
#
# Adjust CONDA_ENV and SNN_DIR to match your installation.

set -euo pipefail

TASK_IDX="${1:?ERROR: task_idx argument required}"
SNN_DIR="/lhome/ext/uovi123/uovi123j/SNN-Cal/SNN"
CONDA_ENV="snn_hgcal"   # <-- change to your conda environment name

# ── Activate conda ─────────────────────────────────────────────────────────────
CONDA_BASE="/lhome/ext/uovi123/uovi123j/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ── Run training ───────────────────────────────────────────────────────────────
cd "${SNN_DIR}"
echo "Starting job: task_idx=${TASK_IDX}  host=$(hostname)  $(date)"
python train_ann_job.py --task_idx "${TASK_IDX}"
echo "Finished job: task_idx=${TASK_IDX}  $(date)"
