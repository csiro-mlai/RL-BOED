#!/bin/bash
#SBATCH --time=3-0:00:00
#SBATCH --mem=20gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

source /home/bla363/start.sh

python /home/bla363/bla363/boed/Experiments/Adaptive_Source_SAC.py --n-contr-samples=100000 --n-rl-itr=20001 --log-dir=/flush5/bla363/source  --bound-type=lower --id=1 --budget=30 --discount=1 --src-filepath="/flush5/bla363/source_25/itr_10000.pkl" --alpha=0
