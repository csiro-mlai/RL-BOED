#!/bin/bash

for folder in $FLUSHDIR/boed_results/*/
do
    sbatch eval_run.sh $folder
done
