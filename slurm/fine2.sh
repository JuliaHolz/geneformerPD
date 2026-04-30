#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --job-name=fi2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --output=/orcd/compute/edsun/001/jholz/finetuning/slurm/0fine-slurm-%j.out
#SBATCH -e /orcd/compute/edsun/001/jholz/finetuning/slurm/0fine-slurm-%j.err
#SBATCH -G a100:1
# Re-queue this job automatically when preempted so it picks up from its
# latest checkpoint.  SLURM sends SIGTERM before killing; the trap below
# resubmits the same script before the process is terminated.
#SBATCH --requeue

# ── Requeue on preemption ────────────────────────────────────────────────────
# When SLURM preempts this job it sends SIGTERM.  We catch it and immediately
# resubmit the script; the Python code will find the latest checkpoint and
# resume from there automatically.
requeue_job() {
    echo "SIGTERM received — requeuing job $SLURM_JOB_ID at $(date)"
    scontrol requeue "$SLURM_JOB_ID"
}
trap requeue_job SIGTERM

# ── Environment ──────────────────────────────────────────────────────────────
source /home/jholz/.bashrc
conda activate geneformer

# W&B credentials — set your key here (or export it in .bashrc instead)
#export WANDB_API_KEY=
# To disable W&B entirely for a run, uncomment the next line:
export WANDB_MODE=disabled

# ── GPU sanity check ─────────────────────────────────────────────────────────
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device count:  ', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU:           ', torch.cuda.get_device_name(0))
else:
    print('WARNING: No GPU detected — training on CPU')
"

# ── Run (add --fold N to target a specific fold) ─────────────────────────────
python /orcd/compute/edsun/001/jholz/finetuning/02_finetune_tracking.py --fold 2

# The & + wait pattern lets the trap fire while Python is running.
# (Remove the & and wait if your cluster discourages background subshells.)
# python /orcd/compute/edsun/001/jholz/finetuning/02_finetune_with_val.py --fold 0 &
# wait $!