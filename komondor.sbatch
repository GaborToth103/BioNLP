#!/bin/sh
#SBATCH --job-name=gemma-print
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=data/output/out.txt
#SBATCH --error=data/output/error.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tothg@inf.u-szeged.hu

SIF_FILE=data/komondor.sif
DATA_PATH=/scratch/c_qa_gen

module load singularity
singularity exec --nv -B $DATA_PATH $SIF_FILE python src/true_main.py --model_name google/gemma-3-27b-it