# Review Results

Computations and models for the article review.


## Random control for attention profile correlation with AAindex and costs

Extract attention profiles from shuffled versions of the same sequences 
and correlate them with AAindex and cost profiles.

Attention extraction from shuffled sequences:

```bash
python scripts/extract_attention_patterns_shuffled.py \
  -c results/20240528_151534/experiment_config.yaml
```

Notebooks:

* `cost_attention_corr_shuffled`
* `aaindex_attention_corr_shuffled.ipynb`


## AA frequency - abundance dependence plot

Look for a dependence between AA frequency and 

Notebook: `aa_freq_abundance.ipynb`


## Partial correlations with mRNA folding strength and tAI

Notebook: `partial_corr_mf_tai.ipynb`


## Thermostability predictions

Notebooks: 

* `OET_predictions.ipynb`
* `OGT_predictions.ipynb`

