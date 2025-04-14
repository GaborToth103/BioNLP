#!/bin/sh
#SBATCH --job-name=BioNLP
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=128G
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --output=data/output/out.txt
#SBATCH --error=data/output/error.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tothg@inf.u-szeged.hu

SIF_FILE=data/komondor.sif
DATA_PATH=/scratch/c_qa_gen

module load singularity
singularity exec --nv -B $DATA_PATH $SIF_FILE python src/true_main.py
