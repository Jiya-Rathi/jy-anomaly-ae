#!/bin/bash
#
# submit_tune.sh
#
# Launch the 100-trial Optuna hyperparameter study on Node E (mantis-04).
# Usage:
#     sbatch submit_tune.sh
#
# Notes:
#   - The study is resumable: if this job is interrupted, re-submitting
#     will pick up where it left off by reading the same study.db.
#   - Email notifications are sent at BEGIN / END / FAIL / 80%-time-used.
#   - Logs go to slurm_logs/tune_<jobid>.{out,err}.
#   - tune.py itself handles the 4 parallel workers (one per GPU) internally.

# ============================================================================
# SLURM directives
# ============================================================================

#SBATCH --job-name=ae_tune
#SBATCH --partition=LocalQ
#SBATCH --nodelist=mantis-04               # Node E -- the one with 4 MI210s
#SBATCH --time=24:00:00                    # 24h cap (study finishes well before)
#SBATCH --cpus-per-task=16                 # plenty of CPU for 4 parallel workers
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/tune_%j.out
#SBATCH --error=slurm_logs/tune_%j.err

# Email notifications
#SBATCH --mail-user=jrathi4302@sdsu.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

# ============================================================================
# Environment setup
# ============================================================================

set -eu                                    # stop on any error or unset var

# Make slurm_logs/ ahead of time (SLURM won't create it)
mkdir -p slurm_logs

# ROCm paths (also in .bashrc, re-exported here to be safe in SLURM's shell)
export ROCM_PATH=/opt/rocm-6.3.3
export HIP_PATH=/opt/rocm-6.3.3
export LD_LIBRARY_PATH=/opt/rocm-6.3.3/lib:${LD_LIBRARY_PATH:-}
export PATH=/opt/rocm-6.3.3/bin:$PATH

# Activate the venv
source /mnt/beegfs/mantis/jrathi/AE_Model_Thesis/ae_env/bin/activate

# Go to project root
cd /mnt/beegfs/mantis/jrathi/AE_Model_Thesis

# ============================================================================
# Pre-flight diagnostics (appended to the SLURM log)
# ============================================================================

echo "================================================================"
echo "SLURM job: $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "Date     : $(date)"
echo "Python   : $(which python)"
echo "----------------------------------------------------------------"
echo "GPU status at start:"
rocm-smi --showtemp --showuse 2>/dev/null | head -20
echo "================================================================"

# ============================================================================
# Run the study
# ============================================================================

python tune.py \
    --study-name   ae_topology_v1 \
    --study-db     study_outputs/study.db \
    --n-trials     100 \
    --n-workers    4 \
    --output-dir   study_outputs

echo "----------------------------------------------------------------"
echo "Study finished at $(date)"
echo "================================================================"
