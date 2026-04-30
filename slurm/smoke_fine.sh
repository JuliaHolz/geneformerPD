#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --job-name=smoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                    # memory per node
#SBATCH --time=12:00:00              # max walltime (HH:MM:SS)
#SBATCH --output=/orcd/compute/edsun/001/jholz/finetuning/slurm/fine-slurm-%j.out   # output file (%j = job ID) to capture logs for debugging
#SBATCH -e /orcd/compute/edsun/001/jholz/finetuning/slurm/smoke-slurm-%j.err  
#SBATCH --nodes=1
#SBATCH -G a100:1             # 1 L40S GPU (44GB VRAM); use a100:1 if available
#SBATCH -t 12:00:00           # preemptable allows longer runtimes than mit_normal_gpu

source /home/jholz/.bashrc
conda activate geneformer
cd /orcd/compute/edsun/001/jholz/finetuning
python /orcd/compute/edsun/001/jholz/finetuning/02_smoke.py