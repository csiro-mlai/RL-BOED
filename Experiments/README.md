The experiments are accessed through the Python files in this folder with the
naming scheme `Adaptive_X_Y.py` where `X` is the problem type and `Y` is the RL
algorithm used. These files accept the following arguments:

--id : this is an identifier indicating which replicate of the experiment to
run. Experiments with the same id number will use common random numbers and 
should reproduce the exact same result.

--n-parallel : the number of trajectories to generate in parallel.

--budget : the length of a the sequence of experiments in each trajectory.

--n-rl-itr : number of rl iterations to run.

--n-contr-samples : number of contrastive samples to use for estimating the EIG.

--log-dir : path to the folder where results will be saved

--snapshot-mode : defines how to take snapshots of the agent. `gap` will save 
periodically. `all` will save every iteration. `last` will save every iteration
but keep only the most recent.

--snapshot-gap : if `snapshot-mode` is `gap`, snapshots will be saved every
`snapshot-gap` iterations of rl.

--bound-type : which type of bound to use in estimating EIG. `lower` and `upper`
 correspond to the bounds from *Deep Adaptive Design* by Foster et al.
`terminal` means that non-terminal rewards are 0 and the terminal reward is the
sPCE integrand of the entire trajectory. Note that if `bound-type` is `lower` or
`terminal` then the EIG estimate is upper bounded by `log(n-contr-samples) + 1`.