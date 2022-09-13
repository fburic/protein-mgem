#!/bin/bash

ml load CUDA/11.3.1

CMD=$*

date
hostname
echo "Command:" $CMD

time singularity exec --nv protein_mgem.sif bash -c "pip install -e . && python ${CMD}"
