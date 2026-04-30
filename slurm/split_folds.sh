#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --job-name=split
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G                    # memory per node
#SBATCH --time=12:00:00              # max walltime (HH:MM:SS)
#SBATCH --output=/orcd/compute/edsun/001/jholz/finetuning/slurm/split-slurm-%j.out   # output file (%j = job ID) to capture logs for debugging
#SBATCH -e /orcd/compute/edsun/001/jholz/finetuning/slurm/split-slurm-%j.err  
#SBATCH --nodes=1

source /home/jholz/.bashrc
conda activate geneformer
cd /orcd/compute/edsun/001/jholz/finetuning
python /orcd/compute/edsun/001/jholz/finetuning/01b_split_folds_with_val.py