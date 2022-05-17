#!/bin/bash
#SBATCH --time=1:59:00
#SBATCH --mem=20gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

module load python/3.7.11
source $FLUSHDIR/boed/bin/activate
cd ~/boed
python select_policy.py --src=/flush5/bla363/boed_results/ces_"$1"/itr_20000.pkl --seq_length=10 --n_contrastive_samples=10000000 --dest=/flush5/bla363/boed_results/ces_"$1"/evaluation_20000.log --n_parallel=10 --n_samples=2000

