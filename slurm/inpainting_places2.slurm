#!/bin/bash
#SBATCH -o experiments/slurm.log
#SBATCH -J base
#SBATCH -p dell
#SBATCH --gres=gpu:4
#SBATCH -c 16
python run.py -c config/inpainting_places2.json -gpu 0,1,2,3 -b 8 
