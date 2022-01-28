# RL for SED

This repository holds the implementation code for the paper **Optimizing Sequential Experimental Design with Deep Reinforcement Learning**, submitted to ICML 2022.

# Installation 

This code has been tested in Python 3.7.11. We offer no guarantee that it will be supported in other versions. Due to dependence on the `rpy2` package, you will need a local copy of `R` installed (see the [rpy2 installation instructions](https://rpy2.github.io/doc/latest/html/overview.html#install-installation)). If you do not want to install `rpy2` and are not interested inr unning the SMC experiment, simply remove this package from the `requirements.txt` file.

Installation is done using `venv`:

````
python -m venv boed
source boed/bin/activate
pip install -r requirements.txt
````

# Running Experiments

To run the experiments in the paper we must use different utilities for the different algorithms, since they come from different sources.

## RL Experiments

The RL experiments can be run by executing the python files in the `Experiments` folder. These expect certain arguments, although some have default values. To reproduce the exact settings in the paper:

For Source location

````
python -m Experiments.Adaptive_Source_SAC --n-contr-samples=100000 --n-rl-itr=20001 --log-dir=<log_dir>/boed_results/source  --bound-type=lower --id=1 --budget=30 --discount=0.9 --buffer-capacity=1000000 --tau=0.001 --pi-lr=0.0001 --qf-lr=0.0003 --ens-size=2
````

For CES
````
ython -m Experiments.Adaptive_CES_SAC --n-contr-samples=100000 --n-rl-itr=20001 --log-dir=<log_dir>/boed_results/ces  --bound-type=lower --id=1 --budget=10 --discount=0.9 --buffer-capacity=1000000 
````

For Prey population
````
python -m Experiments.Adaptive_Prey_SAC --n-contr-samples=10000 --n-rl-itr=40001 --log-dir=<log_dir>/boed_results/prey  --bound-type=lower --id=1 --budget=10 --tau=0.01 --pi-lr=0.0001 --qf-lr=0.001 --discount=0.95 --buffer-capacity=1000000 --temp=1 --target-entropy=2.85189124 --ens-size=10

````

where `<log_dir>` needs to be replaced with a path to a logging directory of your choice.

## PCE Experiments

The PCE experiments are run by executing files in the root folder of the repository that are based on the ones provided by [Foster et al](https://github.com/ae-foster/pyro/tree/sgboed-reproduce).


For Source location

````
python source.py --num-steps=30 --num-parallel=100  --name=pce --typs=pce --num-gradient-steps=2500 
````

For CES
````
python ces.py --num-steps=10 --num-parallel=100  --name=pce --typs=pce --num-acquisition=1 --num-gradient-steps=2500 
````

For CES with PCE-BO
````
python ces.py --num-steps=10 --num-parallel=100  --name=bo --typs=bo --num-gradient-steps=2500 
````

For Prey population
````
python prey.py --num-steps=10 --num-parallel=100  --name=pce --typs=pce
````
## Random Experiments

Random experiments can be executed just like the PCE experiments, but replace the `--types=pce` flag to `--typs=rand` and change the `--name` flag appropriately.

## SMC Experiments

The SMC experiment can be run by executing the file `SMC_prey.py` in the root folder.

## DAD Experiments

DAD experiments are not included in this repository, and must be run using code from [the original paper's repository](https://github.com/ae-foster/dad).