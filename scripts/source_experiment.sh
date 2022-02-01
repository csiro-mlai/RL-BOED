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
python -m Experiments.Adaptive_Source_SAC --n-contr-samples=100000 --n-rl-itr=60001 --log-dir=$FLUSHDIR/boed_results/source  --bound-type=terminal --id=1 --budget=30 --discount=1 --buffer-capacity=10000000 --tau=0.001 --pi-lr=0.0001 --qf-lr=0.0003   #--src-filepath=$FLUSH5DIR/boed_results/source_65/itr_2500.pkl --alpha=0
