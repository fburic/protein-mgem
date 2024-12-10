#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -G 1

DIR_DATA=/proj/berzelius-2024-205/sandra/DeepTranslation/data/PaxDb/Paxdb-sequences_processed_Human_cell_line #PaxDb/Paxdb-sequences


module load Mambaforge
mamba activate jupyter

# create an array with all the filer/dir inside
arr=("${DIR_DATA}"/*)
echo SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID
file=${arr[$SLURM_ARRAY_TASK_ID]}

organism_name=$(echo $file | rev | cut -f 1 -d "/"  | cut -f 2 -d "." | rev)
echo working on organism: $organism_name
echo using file: $file
python train_PaxDB_Models.py -i ${file} 
