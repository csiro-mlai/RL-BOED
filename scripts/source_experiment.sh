#!/bin/bash
#SBATCH --time=3-0:00:00
#SBATCH --mem=20gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

source start.sh
python ~/boed/Experiments/Adaptive_Source_SAC.py --n-contr-samples=100000 --n-rl-itr=20001 --log-dir=$FLUSHDIR/source  --bound-type=lower --id=1 --budget=30 --discount=1  --alpha=0
