#!/bin/bash
declare -a TauArray=("1e-3" "5e-3" "1e-2")
declare -a PilrArray=("1e-5" "1e-4" "3e-4" "1e-3")
declare -a QflrArray=("1e-5" "1e-4" "3e-4" "1e-3")
declare -a BuffArray=("1e5" "1e6" "1e7")
declare -a GammaArray=("0.99" "0.95" "0.9")
for id in 1 2 3
do
  for tau in ${TauArray[@]}
  do
    sbatch prey_experiment_tau.sh $id $tau
  done
  for pilr in ${PilrArray[@]}
  do
    sbatch prey_experiment_pilr.sh $id $pilr
  done
  for qflr in ${QflrArray[@]}
  do
    sbatch prey_experiment_qflr.sh $id $qflr
  done
  for buff in ${BuffArray[@]}
  do
    sbatch prey_experiment_buff.sh $id $buff
  done
 for gamma in ${GammaArray[@]}
 do
   sbatch prey_experiment_gamma.sh $id $gamma
 done
done
