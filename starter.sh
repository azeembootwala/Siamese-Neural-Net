#!/bin/bash
#SBATCH --job-name=0.3_contrastive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH -o /home/%u/output-%j.out
#SBATCH -e /home/%u/code/error-%j.err
#SBATCH --mail-type=ALL

# Small Python packages can be installed in own home directory. Not recommended$
# cluster_requirements.txt is a text file listing the required pip packages (on$
#pip3 install --user -r cluster_requirements.txty

python3 VGG_gen_preproc.py
