#!/bin/bash
for id in {1..10}
do
   sbatch source_experiment_id.sh $id
done
