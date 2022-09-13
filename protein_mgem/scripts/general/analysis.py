from collections import Counter
from typing import Iterable, Union

from goatools.obo_parser import GODag
from goatools.mapslim import mapslim
import numpy as np
import pandas as pd
from scipy import special, stats
from scipy.spatial.distance import jensenshannon
from sklearn import metrics
import torch
from tape import TAPETokenizer

from scripts.general import util


def mase_median(actual: np.ndarray, pred: np.ndarray) -> float:
    """
    MASE using median(actual) as a naive predictor
    """
    mae = np.mean(np.abs(pred - actual + 1e-7))
    mae_naive_predictor = np.mean(np.abs(np.median(actual) - actual) + 1e-7)
    return mae / mae_naive_predictor


def model_performance_table(target_vals: np.array, predicted_vals: np.array) -> pd.DataFrame:
    return pd.DataFrame.from_records([
        ('MSE', metrics.mean_squared_error(target_vals, predicted_vals)),
        ('MAE', metrics.mean_absolute_error(target_vals, predicted_vals)),
        ('MASE', mase_median(target_vals, predicted_vals)),
        ('JSD', jensenshannon(target_vals, predicted_vals) ** 2),
        ('Pearson', stats.pearsonr(target_vals, predicted_vals)[0]),
        ('Spearman', stats.spearmanr(target_vals, predicted_vals)[0]),
        ('R^2', metrics.r2_score(target_vals, predicted_vals))],
        columns=['measure', 'value']
    )


def get_goslims_for_ids(go_ids: list, goslim_dag: GODag, go_dag: GODag) -> pd.DataFrame:
    """
    Map given GO terms to their respective GO slim terms in the provided graph (DAG)
    """
    aspect_acronym = {'molecular_function': 'MF', 'biological_process': 'BP', 'cellular_component': 'CC'}
    go_slim_direct_ancestors = [
        mapslim(term, go_dag, goslim_dag)[0] for term in go_ids
    ]
    go_slim_terms = list(set.union(*go_slim_direct_ancestors))
    go_slim_terms = pd.DataFrame.from_records(
        [(go_id,
          aspect_acronym[goslim_dag[go_id].namespace],
          goslim_dag[go_id].name)
         for go_id in go_slim_terms],
        columns=['go_id', 'aspect', 'go_term']
    )
    return go_slim_terms


class ModelEvaluator(object):
    """Convenience encapsulation of BERT model"""
    def __init__(self, config: dict):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            print('WARN: CUDA device not found. Using CPU.')
            self.device = 'cpu'
        self.model = util.load_model(config).to(self.device)
        self.most_important_head = get_most_important_head(self.model)
        self.tokenizer = TAPETokenizer(vocab='iupac')
        self.boxcox_lambda = config['boxcox_lambda']

    def predict(self, seq, return_attention=False, inverse_boxcox=True):
        """
        Evaluate model on sequence and inverse Box-Cox transforms the output.
        """
        seq_encoded = torch.tensor([self.tokenizer.encode(seq)])
        seq_encoded = seq_encoded.to(self.device)
        pred, attention = self.model(seq_encoded)
        pred = pred.detach().cpu().numpy()[0][0]
        if inverse_boxcox:
            pred = special.inv_boxcox(pred, self.boxcox_lambda)
        if not return_attention:
            return pred
        else:
            return pred, attention

    def get_embedded_vector_repr(self, seq, include_start_token=False):
        """
        Get the embedded values (BERT encoder output) for the sequence
        (without the flanking special tokens).
        """
        model_input = torch.tensor([self.tokenizer.encode(seq)]).to(self.device)
        embedding_output = self.model.bert.embeddings(model_input)
        encoder_outputs = self.model.bert.encoder(embedding_output,
                                                  self.get_mask(model_input),
                                                  chunks=None)
        seq_vector_repr = encoder_outputs[0][0].detach().cpu().numpy()
        if include_start_token:
            seq_vector_repr = seq_vector_repr[:-1]
        else:
            seq_vector_repr = seq_vector_repr[1:-1]
        return seq_vector_repr

    def get_mask(self, model_input):
        """
        Extracted from TAPE code base, used in getting encoder embedded values.
        """
        extended_attention_mask = torch.ones_like(model_input).unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.model.bert.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


def embedded_ordering(seq: str, model_eval: ModelEvaluator, umap_embedder) -> np.array:
    embedded_values = model_eval.get_embedded_vector_repr(seq)
    return umap_embedder.transform(embedded_values).flatten().astype(np.float32)


def get_most_important_head(model) -> int:
    """
    Return index of attention in the top attention layer
    that has highest absolute total weight.
    """
    logger = util.get_logger('get_most_important_head')

    top_layer = model.bert.encoder.layer[-1]

    out_weights = top_layer.attention.output.dense.weight.detach().cpu().numpy()
    out_bias = top_layer.attention.output.dense.bias.detach().cpu().numpy()

    n_heads = top_layer.attention.self.num_attention_heads
    head_size = top_layer.attention.self.attention_head_size

    out_weights_per_head = np.zeros(n_heads)
    out_bias_sum_per_head = np.zeros(n_heads)

    for i in range(top_layer.attention.self.num_attention_heads):
        out_weights_per_head[i] = out_weights.sum(axis=0)[i * head_size: (i + 1) * head_size].sum()

    for i in range(top_layer.attention.self.num_attention_heads):
        out_bias_sum_per_head[i] = out_bias[i * head_size: (i + 1) * head_size].sum()

    most_important_head = np.argmax(np.abs(out_weights_per_head + out_bias_sum_per_head))
    logger.info(f'Most important top attention head in terms of weights: {most_important_head}')
    return most_important_head


def get_significant_aa_count(seq: str, zscores: np.ndarray, zscore_cutoff: float = 1) -> pd.DataFrame:
    """
    Return the count of each amino acid type
    in positions that have an abs(zscore) >= the `zscore_cutoff`
    """
    residues = [seq[i] for i in np.nonzero(np.abs(zscores) >= zscore_cutoff)[0]]
    aa_counts = pd.DataFrame.from_records(
        list(Counter(residues).items()),
        columns = ['aa', 'count']
    )
    return aa_counts


def test_get_significant_aa_count():
    seq = 'MUPPETSMUPPETS'
    zscores = np.array([2, -2, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2])
    expected_counts = pd.DataFrame.from_records(
        [('M', 1),
         ('U', 1),
         ('P', 2),
         ('T', 1),
         ('S', 1)],
        columns=['aa', 'count']
    ).sort_values('aa', ignore_index=True)

    counts = get_significant_aa_count(seq, zscores, zscore_cutoff=2)
    counts = counts.sort_values('aa', ignore_index=True)  # make comparable

    assert expected_counts.compare(counts).empty


def hypergeom_test_aa_counts(aa_counts: pd.DataFrame) -> pd.DataFrame:
    """
    Performs two one-sided hypergeometric tests
    to assess under- and overrepresentaiton.

    Expects columns: aa, count, background_count
    Returns columns: ['aa', 'pval_under', 'pval_over']
    """
    total_count = aa_counts['count'].sum()
    total_background_count = aa_counts['background_count'].sum()
    aa_pval = []
    for _, row in aa_counts.iterrows():
        distrib = stats.hypergeom(M=total_background_count,
                                  n=row['background_count'],
                                  N=total_count)
        pval_under = distrib.cdf(row['count'])
        pval_over = distrib.sf(row['count'] - 1)
        aa_pval.append((row['aa'], pval_under, pval_over))
    return pd.DataFrame.from_records(aa_pval,
                                     columns=['aa', 'pval_under', 'pval_over'])


def test_hypergeom_test_aa_counts():
    aa_counts = pd.DataFrame.from_records(
        [('A', 80, 2000),
         ('B', 10, 2000),
         ('C', 10, 2000)],
        columns=['aa', 'count', 'background_count']
    )
    expected_pvals = pd.DataFrame.from_records(
        [('A', 1.000000e+00, 5.449704e-22),
         ('B', 4.300177e-08, 1.000000e+00),
         ('C', 4.300177e-08, 1.000000e+00)],
        columns=['aa', 'pval_under', 'pval_over']
    )
    sample_ratio_signif_aa = hypergeom_test_aa_counts(aa_counts)
    assert np.allclose(expected_pvals[['pval_under', 'pval_over']].values,
                       sample_ratio_signif_aa[['pval_under', 'pval_over']].values)


def jaccard_similarity(a: Iterable, b: Iterable) -> float:
    a = set(list(a))
    b = set(list(b))
    return len(a.intersection(b)) / len(a.union(b))

