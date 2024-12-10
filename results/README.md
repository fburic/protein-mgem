
## Figure Source Guide

*(Unless otherwise stated, notebooks use the same protein sequence `data/seq/scerevisiae_aminoacid_uniprot_20200120_seqlen_100_to_1000.fasta`
and abundances `data/ho2018/prot_abundance_molecules_per_cell_no_gfp.csv` files as input)*

### Figure 1

- **A** and **B**: [20240830_135757/Train_models_PaxDB_all.ipynb](20240830_135757/Train_models_PaxDB_all.ipynb)
  * inputs: `data/PaxDb` (must be downloaded separately, see [details](20240830_135757/README.md)), `data/ho2018`
- **C**: [20211223_182228/inspect_best_model.ipynb](20211223_182228/inspect_best_model.ipynb)
  * inputs `model/bert`
- **E**: [20211223_182228/cost_attention_corr.ipynb](20211223_182228/cost_attention_corr.ipynb) 
          and [20240528_151534/cost_attention_corr_shuffled.ipynb](20240528_151534/cost_attention_corr_shuffled.ipynb) (with controls)
  * inputs: `data/protein_features/uncorr_aaindex_profiles.h5`, `20211223_182228/attention_patterns`, `20240528_151534/attention_patterns` (for shuffled sequences)
- **F**: [20211223_182228/aaindex_attention_corr.ipynb](20211223_182228/aaindex_attention_corr.ipynb) 
          and [20240528_151534/aaindex_attention_corr_shuffled.ipynb](20240528_151534/aaindex_attention_corr_shuffled.ipynb) (with controls)
  * inputs: `data/aa_costs/aa_costs_barton.csv`, `20211223_182228/informed_mutation/guided_mutation_results.csv.gz`
- **G**: [20211223_182228/attention_and_secondary_structure.ipynb](20211223_182228/attention_and_secondary_structure.ipynb)
  * inputs: `data/pdb_yeast/pdb_files_alphafold` (precomputed`20211223_182228/secondary_structures.csv.gz` available as alternative) 
- **H**: [20211223_182228/attention_and_interpro.ipynb](20211223_182228/attention_and_interpro.ipynb)
  * inputs: `data/interpro/yeast_protein2ipr.csv.gz`, `data/interpro/entry.list`, `20211223_182228/attention_patterns/`, `data/seq/yeast_gene_entries_2020_02_26.csv`, `data/s288c_genes_ncbi_20210305.tsv`

### Figure 2

- **C**: [20211223_182228/inspect_embedded_ordering.ipynb](20211223_182228/inspect_embedded_ordering.ipynb)
  * inputs: `seq_orderings.h5` (precomputed ordering values, see notebook for recreating them)
- **D**: [20211223_182228/guided_mutation_results.ipynb](20211223_182228/guided_mutation_results.ipynb)
  * inputs: `guided_mutation_results.csv.gz`, `data/aa_costs/aa_costs_barton.csv`
- **E**: [20211223_182228/ordering_overview_over_sequences.ipynb](20211223_182228/ordering_overview_over_sequences.ipynb)
  * inputs: `seq_orderings.h5` 
- **F**: [20211223_182228/guided_mutation_results.ipynb](20211223_182228/guided_mutation_results.ipynb)
- **G**: [20211223_182228/guided_mutation_results.ipynb](20211223_182228/guided_mutation_results.ipynb)


### Supplemental Figures:
- **S1**: [20240115_135009/ribosome_halflife_abundance.ipynb](20240115_135009/ribosome_halflife_abundance.ipynb)
  * inputs: `data/ho2018/prot_abundance_molecules_per_cell_no_gfp.csv`, `data/seq/scerevisiae_aminoacid_uniprot_20200120.fasta`,
   `data/protein_stability/christiano_2014_table_s1.xlsx`, `data/ribosome_profiling/weinberg2016_GSE75897_RPF_RPKMs.txt.gz`, `data/ribosome_profiling/weinberg2016_GSE75897_RiboZero_RPKMs.txt.gz`,
   `data/riba2019/pnas.1817299116.sd02.tsv`
- **S2**, **S3**, and **S4** [20240830_135757/Train_models_PaxDB_all.ipynb](20240830_135757/Train_models_PaxDB_all.ipynb)
  * inputs: `data/PaxDb` (must be downloaded separately, see [details](20240830_135757/README.md)), `data/ho2018`
- **S5**: [20211223_182228/inspect_best_model.ipynb](20211223_182228/inspect_best_model.ipynb)
  * inputs `model/bert`
- **S6**: [20211223_182228/data_description.Rmd](20211223_182228/data_description.Rmd) and [20240528_151534/aa_freq_abundance.ipynb](20240528_151534/aa_freq_abundance.ipynb)
  * inputs: `data/ho2018/prot_abundance_molecules_per_cell_no_gfp.csv`
- **S8**: [20211223_182228/attention_and_interpro.ipynb](20211223_182228/attention_and_interpro.ipynb)
  * inputs: `data/interpro/yeast_protein2ipr.csv.gz`, `data/interpro/entry.list`, `20211223_182228/attention_patterns/`, `data/seq/yeast_gene_entries_2020_02_26.csv`, `data/s288c_genes_ncbi_20210305.tsv`
- **S9**: [20211223_182228/aindex_attention_corr.ipynb](20211223_182228/aindex_attention_corr.ipynb)
  * inputs: `data/aa_costs/aa_costs_barton.csv`, `results/20211223_182228/informed_mutation/guided_mutation_results.csv.gz`
- **S12** [20240528_151534/OGT_predictions.ipynb](20240528_151534/OGT_predictions.ipynb)
  * inputs: `data/temperature/*.fasta`
- **S15**: [20240528_151534/partial_corr_mf_tai.ipynb](20240528_151534/partial_corr_mf_tai.ipynb)
  * inputs: `data/seq/codons/tai_yeast_ecoli.csv`, `data/seq/S288C_reference_genome_R64-2-1_20150113/orf_coding_all_R64-2-1_20150113.fasta`,
    `data/mrna_folding/sce_Score.tab.gz`, `data/ho2018/prot_abundance_molecules_per_cell_no_gfp.csv`
