#!/bin/bash

date
hostname

time singularity exec --nv protein_mgem.sif python $*

