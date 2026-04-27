#!/bin/bash
#
# submit_tune_mini.sh
#
# Mini-test version of submit_tune.sh -- 8 trials instead of 100.
# Used to verify (a) SLURM submission works, (b) the constant_liar fix
# produces diverse hyperparameters, and (c) per-trial timing under
# SLURM's resource isolation.
#
# Usage:
#     sbatch submit_tune_mini.sh

# ============================================================================
# SLURM directives
# ============================================================================

#SBATCH --job-name=ae_tune_mini
#SBATCH --partition=LocalQ
#SBATCH --nodelist=mantis-04
#SBATCH --time=02:00:00                    # 2-hour cap (mini-test)
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/tune_mini_%j.out
#SBATCH --error=slurm_logs/tune_mini_%j.err

# Email notifications
#SBATCH --mail-user=jrathi4302@sdsu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Environment setup
# ============================================================================

set -eu

mkdir -p slurm_logs

# ROCm paths
export ROCM_PATH=/opt/rocm-6.3.3
export HIP_PATH=/opt/rocm-6.3.3
export LD_LIBRARY_PATH=/opt/rocm-6.3.3/lib:${LD_LIBRARY_PATH:-}
export PATH=/opt/rocm-6.3.3/bin:$PATH

# Activate venv
source /mnt/beegfs/mantis/jrathi/AE_Model_Thesis/ae_env/bin/activate

cd /mnt/beegfs/mantis/jrathi/AE_Model_Thesis

# ============================================================================
# Pre-flight diagnostics
# ============================================================================

echo "================================================================"
echo "SLURM mini-test: $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "Date     : $(date)"
echo "Python   : $(which python)"
echo "----------------------------------------------------------------"
echo "GPU status at start:"
rocm-smi --showtemp --showuse 2>/dev/null | head -20
echo "================================================================"

# ============================================================================
# Run a small study to verify everything works under SLURM
# ============================================================================

# Use a fresh study name so this doesn't conflict with the real run later
python tune.py \
    --study-name   ae_topology_minitest_v1 \
    --study-db     mini_test_outputs/study.db \
    --n-trials     8 \
    --n-workers    4 \
    --output-dir   mini_test_outputs

echo "----------------------------------------------------------------"
echo "Mini-test finished at $(date)"
echo "================================================================"
