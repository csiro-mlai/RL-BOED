#!/bin/bash
for d in 2 3 4 5
do
  for k in 2 3 4 5
  do
    sh source_experiment_kd.sh $d $k
  done
done
