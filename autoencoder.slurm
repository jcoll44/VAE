#!/bin/bash
#SBATCH --account=engs-a2i
#SBATCH --nodes=1
#SBATCH --job-name=autoencoder_test
#SBATCH --time=0-00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
 
cd /home/engs2402/Autoencoder
singularity exec --nv -B .:/home/ autoencoder.sif jupyter nbconvert --to script /home/Autoencoder_Notebook.ipynb --stdout