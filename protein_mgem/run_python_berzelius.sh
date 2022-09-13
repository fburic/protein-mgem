#!/bin/bash

CMD=$*

date
hostname
echo "Command:" $CMD

time singularity exec --bind $(pwd) --nv protein_mgem_berzelius.sif bash -c "pip install -e . && python ${CMD}"

