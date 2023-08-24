# Enhanced dataset with counter-examples for `BERT : aa_seq -> prot_abundance`

Used quantities from all Ho et al. (2018) experiments, not just medians,
by having protein sequences repeated.
For each sequence duplicate, have a shuffled version with abundance zero.
Dataset prepared with the notebook `prep_data.ipynb`


## Figure Source Guide

*(Unless otherwise stated, notebooks use the same protein sequence `data/seq/scerevisiae_aminoacid_uniprot_20200120_seqlen_100_to_1000.fasta`
and abundances `data/ho2018/prot_abundance_molecules_per_cell_no_gfp.csv` files as input)*

### Figure 1

- **A**: inspect_best_model.ipynb
  * inputs `model/bert`
- **B**: cost_attention_corr.ipynb
  * inputs: `data/protein_features/uncorr_aaindex_profiles.h5`, `results/20211223_182228/attention_patterns/`
- **C**: aaindex_attention_corr.ipynb
  * inputs: `data/aa_costs/aa_costs_barton.csv`, `results/20211223_182228/informed_mutation/guided_mutation_results.csv.gz`
- **D**: attention_and_secondary_structure.ipynb
  * input: `data/pdb_yeast/pdb_files_alphafold` (precomputed`results/20211223_182228/secondary_structures.csv.gz` available as alternative) 
- **E**: attention_and_interpro.ipynb
  * input: `data/interpro/yeast_protein2ipr.csv.gz`, `data/interpro/entry.list`, `results/20211223_182228/attention_patterns/`, `data/seq/yeast_gene_entries_2020_02_26.csv`, `data/s288c_genes_ncbi_20210305.tsv`

### Figure 2

- **C**: inspect_embedded_ordering.ipynb
  * inputs: `seq_orderings.h5` (precomputed ordering values, see notebook for recreating them)
- **D**: guided_mutation_results.ipynb
  * inputs: `guided_mutation_results.csv.gz`, `data/aa_costs/aa_costs_barton.csv`
- **E**: ordering_overview_over_sequences.ipynb
  * inputs: `seq_orderings.h5` 
- **F**: guided_mutation_results.ipynb
- **G**: guided_mutation_results.ipynb

### Figure 3

- **F?**: cost_attention_corr.ipynb

### Supplemental Figures: TODO

- **S1**: data_description.Rmd


## History


### Trained model with the best parameters from normal dataset

Used the best hypermodel params from `20201223_171730/best_config.json`
(copied here)

* Started: 2022-01-13
* System: Berzelius
* Commit: `9c91f44b`
* Log: `slurm-1320101.out`
* Time taken: 31.5 h

```bash
sbatch -t 3-00:00:00 --gpus 2 run_python_berzelius.sh \
    scripts/model/bert.py --config results/20211223_182228/best_config.json \
    --res_dir results/20211223_182228
```


### Hyperparameter search 

Done in previous experiment iteration (20201223_171730), on the unenhanced 
version of the same data.

The hypersearch consisted of 10 trials

* Started: 2021-01-07
* System: Ox
* Commit: `68b67c0d`
* Log: `logs/hyperbert_ox.log`
* Time taken: 28 days

```shell
python results/20201223_171730/hyperbert.py > results/20201223_171730/logs/hyperbert_ox_2.log 2>&1 &
```

Best trial from Ray:

| TrialID  | StartTime           | MeanLoss         | 
| d4de20a4 | 2021-02-02 06:51:10 | 3.28383748523e-1 |

Best params:  

```python
{'learning_rate': 2.5681310045153124e-07, 
 'num_train_epochs': 500, 
 'batch_size': 32,
 'num_hidden_layers': 8, 
 'num_attention_heads': 4, 
 'hidden_size': 1024,
 'intermediate_size': 3072, 
 'hidden_dropout_prob': 0.0,
 'attention_probs_dropout_prob': 0.0, 
 'hidden_act': 'relu'}
```


### Analysis

#### Extracted attention patterns

* Started: 2022-02-02
* System: Ox
* Commit: `30ae17b2`
* Log: `extract_attention_patterns.log`
* Time taken: 1.5 h

```bash
nohup python scripts/extract_attention_patterns.py \
  -c results/20211223_182228/experiment_config.yaml \
  > results/20211223_182228/logs/extract_attention_patterns.log &
```

#### Correlated attention profiles with AAindex profiles

* Started: 2022-02-02
* System: Ox
* Commit: `30ae17b2`
* Time taken: 2 min

`aaindex_attention_corr.ipynb`

#### Assessed domain coverage by attention profiles

* Started: 2022-02-02
* System: Ox
* Commit: `30ae17b2`
* Time taken: 2 min

`attention_and_interpro.ipynb`


### MGEM

#### Constructed embedded ordering and save values for all files

* Started: 2022-02-04
* System: Ox
* Commit: `b2275d98`
* Time taken: 1 h

```bash
python scripts/construct_embedding_order.py -c results/20211223_182228/experiment_config.yaml
```

Best UMAP hyperparam n_neighbors_fract = 0.16
Wrote `umap_embedder_pos`


* Started: 2022-02-04
* System: Ox
* Commit: `81e94784`
* Time taken: 4 min


```bash
python scripts/get_embedded_ordering.py --config results/20211223_182228/experiment_config.yaml \
  --fasta data/seq/scerevisiae_aminoacid_uniprot_20200120_seqlen_100_to_1000.fasta \
  --output results/20211223_182228/seq_orderings.h5
```

#### Embedded ordering profile overview

`ordering_overview_over_sequences.ipynb`


#### Guided Mutation

The script was cleaned up. Now the `inc_ordering_dist` is the only approach used.
The amount with which residue orderings are increased is taken as the length of the
O interval with 99% of values. In this case `8.61`

* Started: 2022-02-08
* System: Alvis
* Commit: `05a90bf4`
* Time taken: 30 min

```bash
sbatch -A SNIC2021-7-44 -t 1:00:00 --gpus-per-node=T4:1 run_python_alvis_new.sh \
  scripts/informed_mutation.py --config results/20211223_182228/experiment_config.yaml \
  --num-mutated-residues 2

sbatch -A SNIC2021-7-44 -t 1:00:00 --gpus-per-node=T4:1 run_python_alvis_new.sh \
  scripts/informed_mutation.py --config results/20211223_182228/experiment_config.yaml \
  --num-mutated-residues 5
  
sbatch -A SNIC2021-7-44 -t 1:00:00 --gpus-per-node=T4:1 run_python_alvis_new.sh \
  scripts/informed_mutation.py --config results/20211223_182228/experiment_config.yaml \
  --num-mutated-residues 10
  
sbatch -A SNIC2021-7-44 -t 1:00:00 --gpus-per-node=T4:1 run_python_alvis_new.sh \
  scripts/informed_mutation.py --config results/20211223_182228/experiment_config.yaml \
  --num-mutated-residues 20
  
sbatch -A SNIC2021-7-44 -t 1:00:00 --gpus-per-node=T4:1 run_python_alvis_new.sh \
  scripts/informed_mutation.py --config results/20211223_182228/experiment_config.yaml \
  --num-mutated-residues 10%
  
sbatch -A SNIC2021-7-44 -t 1:00:00 --gpus-per-node=T4:1 run_python_alvis_new.sh \
  scripts/informed_mutation.py --config results/20211223_182228/experiment_config.yaml \
  --num-mutated-residues 20%
  
sbatch -A SNIC2021-7-44 -t 1:00:00 --gpus-per-node=T4:1 run_python_alvis_new.sh \
  scripts/informed_mutation.py --config results/20211223_182228/experiment_config.yaml \
  --num-mutated-residues 30% 
```

```csv
fixed_02,informed_mutation_ca6f70f288dc11ecb991e1ef75376b7a.csv,slurm-195037.out
fixed_05,informed_mutation_b7b798f388db11ec9bff3704140e7db9.csv,slurm-195029.out
fixed_10,informed_mutation_f06bc50288db11ecbe6b390230713c4f.csv,slurm-195032.out
fixed_20,informed_mutation_1d6ee48f88dc11ec9a51d1a48702a2d2.csv,slurm-195033.out
percent_10,informed_mutation_49d6e7ba88dc11ec8bb0370c9d5bfe2e.csv,slurm-195034.out
percent_20,informed_mutation_68dcca6188dc11eca91d755d65979837.csv,slurm-195035.out
percent_30,informed_mutation_805c90f588dc11ecbcc6f1f79ca0643f.csv,slurm-195036.out
```

