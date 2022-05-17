#!/bin/bash
#SBATCH --time=3-0:00:00
#SBATCH --mem=20gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

module load python/3.7.11
source $FLUSHDIR/boed/bin/activate
cd ~/boed
python -m Experiments.Adaptive_CES_SAC --n-contr-samples=100000 --n-rl-itr=20001 --log-dir=$FLUSHDIR/boed_results/ces  --bound-type=lower --id=1 --budget=10 --discount=0.9 --buffer-capacity=1000000 --ens-size=2
