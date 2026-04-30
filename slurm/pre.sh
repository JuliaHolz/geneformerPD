#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --job-name=pre
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1000G                    # memory per node
#SBATCH --time=24:00:00              # max walltime (HH:MM:SS)
#SBATCH --output=/orcd/compute/edsun/001/jholz/finetuning/slurm/0pre-slurm-%j.out        # output file (%j = job ID) to capture logs for debugging
source /home/jholz/.bashrc
conda activate geneformer
cd /orcd/compute/edsun/001/jholz/finetuning
python /orcd/compute/edsun/001/jholz/finetuning/01_preprocess_and_tokenize.py