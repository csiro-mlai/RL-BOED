#!/bin/bash
for id in {2..10}
do
   sbatch ces_experiment_id.sh $id
done
