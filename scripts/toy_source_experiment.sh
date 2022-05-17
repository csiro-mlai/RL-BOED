#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --mem=20gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

module load python/3.7.11
source $FLUSHDIR/boed/bin/activate
python -m Experiments.Adaptive_Source_SAC --n-contr-samples=100000 --n-rl-itr=20001 --log-dir=$FLUSHDIR/boed_results/toy_source  --bound-type=lower --id=1 --budget=2 --discount=1 --buffer-capacity=10000000 --k=1 --d=1 --tau=0.001 --pi-lr=0.0001 --qf-lr=0.0003
