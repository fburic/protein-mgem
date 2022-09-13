from multiprocessing import cpu_count
import os
import random

import numexpr
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from umap.parametric_umap import ParametricUMAP
import tensorflow as tf
from tqdm import tqdm

from scripts.general import analysis, data, util

numexpr.set_num_threads(cpu_count() - 2)
logger = util.get_logger()


def main():
    args = util.get_args()

    sequences = data.load_seq_and_abundance_data(args.config)['seq'].values
    n_seq = sequences.shape[0]
    model_eval = analysis.ModelEvaluator(args.config)
    logger.info('Model predictions for all sequences')
    predictions = [model_eval.predict(seq, inverse_boxcox=False)
                   for seq in tqdm(sequences, desc='sequences')]
    predictions = np.array(predictions, dtype='float32')

    logger.info('Embedded orderings for all sequences')
    embedded_start_tokens = []
    embedded_sequences = []
    for i, seq in tqdm(enumerate(sequences), total=n_seq, desc='sequences'):
        embedded_seq = model_eval.get_embedded_vector_repr(seq, include_start_token=True)
        embedded_start_tokens.append(embedded_seq[0])
        embedded_sequences.append(embedded_seq[1:])
    embedded_start_tokens = np.stack(embedded_start_tokens)

    (start_tokens_train, _,
     _, embedded_sequences_test,
     _, predictions_test) = train_test_split(embedded_start_tokens, embedded_sequences,
                                             predictions, test_size=0.1,
                                             shuffle=True, random_state=123)

    n_neighbors_fract = np.arange(start=0.01, stop=0.26, step=0.01)
    correlations = []
    umap_embedders = []
    logger.info('Fitting UMAP over n_neighbors')
    for p in tqdm(n_neighbors_fract):
        umap_embedder = fit_umap(start_tokens_train, p)
        corr = corr_projection_target(umap_embedder,
                                      embedded_sequences_test, predictions_test)
        correlations.append(corr)
        umap_embedders.append(umap_embedder)
        logger.info(f'Neighbors% : {p:.2f} - Spearman corr = {corr:.2f}')
    correlations = np.array(correlations, dtype='float32')

    signs = np.sign(correlations)
    best_idx = np.abs(correlations).argmax()
    logger.info(f'Best corr: {correlations[best_idx]:.2f}')

    corr_direction = {-1: 'neg', 0: 'uncorr', 1: 'pos'}
    embedder_fname = f'umap_embedder_{corr_direction[signs[best_idx]]}'
    umap_embedders[best_idx].save(str(args.exp_dir / embedder_fname))
    logger.info(str(args.exp_dir / embedder_fname))


def fit_umap(embedded_start_tokens, n_neighbors_fract):
    RND_SEED = 42
    np.random.seed(RND_SEED)
    random.seed(RND_SEED)
    tf.random.set_seed(RND_SEED)
    os.environ['PYTHONHASHSEED'] = '0'

    n_neighbors = int(n_neighbors_fract * len(embedded_start_tokens))
    umap_embedder = ParametricUMAP(n_components=1,
                                   n_neighbors=max(2, n_neighbors),
                                   min_dist=0.0)
    umap_embedder = umap_embedder.fit(embedded_start_tokens)
    return umap_embedder


def corr_projection_target(umap_embedder, embedded_sequences, predictions):
    """
    Return Spearman corr between the centroid of UMAP-projected point clouds and
    the predictions of the corresponding sequences.
    """
    n_seq = len(embedded_sequences)
    centroids_of_projections = np.zeros(n_seq, dtype='float32')
    for i, seq_e in enumerate(embedded_sequences):
        embedded_ordering = umap_embedder.transform(seq_e).flatten()
        centroids_of_projections[i] = embedded_ordering.mean()
    return stats.spearmanr(centroids_of_projections.flatten(), predictions)[0]


if __name__ == '__main__':
    main()
