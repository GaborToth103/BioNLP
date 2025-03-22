#!/bin/sh
#SBATCH --job-name=llama_testing
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=128G
#SBATCH --gres=gpu:4
#SBATCH --time=00:05:00
#SBATCH --output=data/output/out.txt
#SBATCH --error=data/output/error.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tothg@inf.u-szeged.hu

SIF_FILE=data/komondor.sif
DATA_PATH=/scratch/c_qa_gen

# Betöltjük a singularity-t, ha eddig nem tettük meg (így nem kell használni mindig amikor bejelentkezünk)
module load singularity

# A közös tárhelyet a konténerben felcsatoljuk, ezáltal a dolgok a konténeren kívül is elérhetővé válnak. Az --nv pedig az Nvidia GPU támogatást teszi lehetővé.
singularity exec --nv -B $DATA_PATH $SIF_FILE python test/komondor.py
