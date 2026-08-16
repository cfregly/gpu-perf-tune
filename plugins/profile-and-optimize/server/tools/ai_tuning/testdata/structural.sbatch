#!/usr/bin/env bash
#SBATCH --job-name=ai-tuning-structural
#SBATCH --nodes=8
#SBATCH --partition=example
#SBATCH --gres=gpu:8
env -i bash run.sh
