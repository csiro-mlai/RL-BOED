#Overview

This folder holds scripts that were used to run various experiments and
evaluations. This work was done in the context of a HPC cluster running a Slurm
job scheduling system. Hence, most files are shell scripts designed to be called
with the `sbatch` command.

Outside the context of the Slurm system, most scripts contain a python command
that can be run independently to execute an experiment. Make sure that you
activate an appropriate virtual env before doing this.

## select_policy.py

This python script evaluates a saved agent. Its arguments are:

- src: a filepath to a `.pkl` file containing a `Garage` algorithm and env.
- dest: a filepath at which to save the results of the evaluation.
- seq_length: the budget of experiments.
- bound_type: `lower` for sPCE and `upper` for sNMC.
- n_contrastive_samples: number of contrastive samples to use in calculating the
bound.
- n_parallel: number of sequences to evaluate in parallel.
- n_samples: number of sequences to evaluate in total.

example:
    
    python -m scripts.select_policy --src=source_20/itr_10000.pkl
    --dest=source_20/evaluation.log --edit_type="w" --seq_length=30
    --bound_type=lower --n_contrastive_samples=10000 --n_parallel=1000
    --n_samples=2000 --seed=1

## eval_run.sh

This script file is used to run policy evaluation in a slurm job scheduling
system. its arguments are

1) a filepath to the folder where the `.pkl` file is stored. Assumes that
filenames are in the format `itr_<x>.pkl` where `<x>` is the iteration
number.
2) the iteration number `<x>`.

example:

    sbatch eval_run.sh boed_results/prey_17/ 10000

## hyperparameter search

Scripts of the form `hyperparameter_search_<x>.sh` execute a linear search on
various hyperpameters. They rely on corresponding scripts of the form
`<x>_experiment_<y>` where `<y>` is a hyperparameter, which are a convenience
for passing an experiment id and a single hyperparameter value as arguments. For
example, the command line

    sbatch source_experiment_buff.sh 1 1e6

will run a source location experiment with id `1` and a maximum buffer size of
`1e6`.

## replicate experiment

Scripts of the form `replicate_<x>.sh` are for re-running an experiment with
different random seeds. Each will run the corresponding `<x>_experiment_id.sh`
file with the `10` different random seeds specified in the corresponding python
file in `Experiments`, e.g. `Experiments/Adaptive_Source_SAC.py` for the source
location experiment.