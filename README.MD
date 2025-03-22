# BioNLP Komondor Repository

## Steps
* Pull down the repository and open a terminal inside the repository. You can try the following commands:
``` python
module load singularity # loads the Singularity module into the Linux work session. 
singularity build --fakeroot --fix-perms data/komondor.sif src/komondor.def # building the komondor.sif kontainer file.
sbatch komondor.sbatch # sends signal to the Slurm manager.
squeue # you can get your task details that are inside the queue.
scancel # you can cancel your task if needed.
squota # you can get the space quota for both the whole project and your personal space.
sbalance # you can get the project resource balance for both the cpu and gpu. 
```

Keep in mind that you might need a `HF_TOKEN` environmental variable to access restricted models. It is recommended to create an untracked `.env` file and store your keys there.

## Console Commands

