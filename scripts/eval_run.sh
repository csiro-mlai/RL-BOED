#!/bin/bash
#SBATCH --time=1:59:00
#SBATCH --mem=20gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

module load python/3.7.11
source $FLUSHDIR/boed/bin/activate

folder=$1
for val in $(seq 5000 500 20000)
do
    src="$folder"itr_"$val".pkl
    dest="$folder"evaluation_"$val".log
    python -m scripts.select_policy --src="$src" --dest="$dest" --edit_type="w" --seq_length=30 --bound_type=lower --n_contrastive_samples=1000000 --n_parallel=100 --n_samples=2000
done
