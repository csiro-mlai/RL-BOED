#!/bin/bash

module load python/3.7.11
python -m venv $FLUSHDIR/boed
source $FLUSHDIR/boed/bin/activate
pip install -r ../requirements.txt
