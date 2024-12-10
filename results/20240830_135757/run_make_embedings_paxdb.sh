#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -G 1

DIR_DATA=/proj/berzelius-2024-205/sandra/DeepTranslation/data/PaxDb/data_sets/fasta/
DIR_OUTPUT=/proj/berzelius-2024-205/sandra/DeepTranslation/data/PaxDb/data_sets/embedding/

module load Mambaforge
mamba activate jupyter

for file in $DIR_DATA/*;
do
organism_name=$(echo $file | rev | cut -f 1 -d "/" | rev | cut -f 1 -d ".") 
python make_embedings.py -i ${file} -o ${DIR_OUTPUT}/${organism_name}.pkl
done