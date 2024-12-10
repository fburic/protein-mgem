#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -G 1

DIR_DATA=/proj/berzelius-2024-205/sandra/DeepTranslation/data/Ho2018/fasta #PaxDb/Paxdb-sequences
DIR_OUTPUT=/proj/berzelius-2024-205/sandra/DeepTranslation/data/Ho2018/embedding #PaxDb/Paxdb-embedding/

module load Mambaforge
mamba activate jupyter

# create an array with all the filer/dir inside
arr=("${DIR_DATA}"/*)
echo SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID
file=${arr[$SLURM_ARRAY_TASK_ID]}

organism_name=$(echo $file | rev | cut -f 1 -d "/"  | cut -f 2 -d "." | rev)
echo working on organism: $organism_name
echo using file: $file
python make_embedings.py -i ${file} -o ${DIR_OUTPUT}/${organism_name}.pkl --model ESM

