#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH --mem=20gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

module load python/3.7.11
source $FLUSHDIR/boed/bin/activate
cd ~/boed
python -m Experiments.Adaptive_Prey_SAC --n-contr-samples=10000 --n-rl-itr=40001 --log-dir=$FLUSHDIR/boed_results/prey  --bound-type=lower --id=$1 --budget=10 --discount=0.9 --pi-lr=$2
