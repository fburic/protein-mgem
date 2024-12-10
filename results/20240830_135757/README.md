# ESM protein seq -> abundance models from PaxDb and Ho2018 data

## PaxDb data

Protein sequence and abundance values were downloaded from https://pax-db.org/download
Expected relative locations:
- `../../data/PaxDb`
- `../../data//Ho2018`

Data processing, including matching sequences with abundance values was performed in
`Constructing_dataset.ipynb`

Median abundance for each gene across experiments was computed with
`extract_median_abundance.py`


## Generate ESM embeddings

- HPC (slurm) wrappers:
  - PaxDb sequences : `run_make_embedings_paxdb.sh`
  - Ho 2018 sequence: `run_make_embedings_ho2018.sh`
- script `make_embedings.py`


## Training for all PaxDb organisms

- HPC (slurm) wrapper `train_paxdb_all.sh`
- script `train_PaxDB_Models.py`


## Notebooks

- `Train_models_PaxDB_all.ipynb`
- `train_models.ipynb`
