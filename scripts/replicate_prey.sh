#!/bin/bash
for id in {1..10}
do
   sbatch prey_experiment_id.sh $id
done
