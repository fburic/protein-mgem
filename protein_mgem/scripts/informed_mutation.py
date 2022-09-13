import argparse
import hashlib
import uuid
from multiprocessing import cpu_count
import os
from pathlib import Path

import h5py
import numpy as np
import numexpr
from numpy.random import default_rng
from scipy.spatial.distance import cdist
import pandas as pd
from tqdm import tqdm
import yaml

from scripts.general import analysis, attribution, data, util

numexpr.set_num_threads(cpu_count())
logger = util.get_logger()


def main():
    args = get_args()
    output_fname = get_output_filename(args)
    logger.info(f'Writing results to {output_fname}')

    sequences = data.load_sequences_that_have_abundance(args.config)
    higher_abundance_sequences = data.load_top_n_sequences(args.config)
    seq_ordering = {}
    with h5py.File(args.config['files']['saved_seq_orderings'], 'r') as store:
        for prot_id in store.keys():
            seq_ordering[prot_id] = np.array(store[prot_id], dtype=np.float32)
    guide_seq_orderings = {prot_id: seq_ordering[prot_id]
                           for prot_id in higher_abundance_sequences.keys()}
    logger.info('Loaded sequence data.')

    model_eval = analysis.ModelEvaluator(args.config)

    logger.info('Mutating proteome')
    ordering_increase = get_width_of_majority_ordering_range(seq_ordering)
    logger.info(f'Residue orderings increased with {ordering_increase: .2f}')
    mutation_results = []
    if args.subsample:
        rng = default_rng(seed=42)
        sequence_entries = rng.choice(list(sequences.items()), size=100)
    else:
        sequence_entries = list(sequences.items())

    for id_wt, seq_wt in tqdm(sequence_entries):
        n_res_mut = parse_num_res_spec(args.num_mutated_residues, seq_wt)
        pred_wt = model_eval.predict(seq_wt, inverse_boxcox=True)
        seq_mut, pred_mut = get_max_mutation_using_guides(model_eval = model_eval,
                                                          sequences = sequences,
                                                          embedded_ordering_wt = seq_ordering,
                                                          id_wt = id_wt,
                                                          n_res_mut = n_res_mut,
                                                          ordering_increase = ordering_increase,
                                                          guide_seq_orderings=guide_seq_orderings)

        seq_mut_control = mutate_seq_random_n_residues(seq_wt, n=n_res_mut)
        pred_control = model_eval.predict(seq_mut_control, inverse_boxcox=True)

        mutation_results.append(
            (id_wt, seq_wt, seq_mut, seq_mut_control, pred_wt, pred_mut, pred_control)
        )
    mutation_results_df = pd.DataFrame(
        mutation_results,
        columns=['swissprot_ac', 'seq_wt', 'seq_mut', 'seq_mut_control',
                 'pred_wt', 'pred_mut', 'pred_control']
    )
    mutation_results_df.to_csv(output_fname, index=False)
    logger.info('Done.')


def get_max_mutation_using_guides(model_eval: analysis.ModelEvaluator, sequences: dict,
                                  embedded_ordering_wt: dict, id_wt: str,
                                  n_res_mut: int, ordering_increase: float,
                                  guide_seq_orderings: dict) -> tuple:
    """
    Produce a mutated sequence using each target sequence,
    by taking target residues according to embedded space ordering.

    Keep only mutant with the highest prediction increase.
    """
    mut_positions = np.argsort(embedded_ordering_wt[id_wt])[:n_res_mut]

    max_set_mut = ''
    max_pred_mut = -np.inf
    for id_guide, ordering_guide in guide_seq_orderings.items():
        seq_mut = get_mutated_sequence(seq_wt = sequences[id_wt],
                                       embedded_ordering_wt = embedded_ordering_wt[id_wt],
                                       ordering_increase = ordering_increase,
                                       mut_positions = mut_positions,
                                       seq_guide = sequences[id_guide],
                                       embedded_ordering_guide = ordering_guide.reshape(-1, 1))
        pred_mut = model_eval.predict(seq_mut, inverse_boxcox=True)
        if pred_mut > max_pred_mut:
            max_pred_mut = pred_mut
            max_set_mut = seq_mut
    return max_set_mut, max_pred_mut


def get_mutated_sequence(seq_wt: str, embedded_ordering_wt: np.array,
                         ordering_increase: float, mut_positions: np.array,
                         seq_guide: str, embedded_ordering_guide: np.array) -> str:
    seq_mut = list(seq_wt)
    for mut_pos in mut_positions:
        # Avoid mutating leading M
        if mut_pos == 0:
            mut_pos += 1
        dist_to_guide = cdist(
            embedded_ordering_wt[mut_pos].reshape(-1, 1) + ordering_increase,
            embedded_ordering_guide
        )
        closest_idx = np.argmin(dist_to_guide)
        seq_mut[mut_pos] = list(seq_guide)[closest_idx]
    seq_mut = ''.join(seq_mut)
    return seq_mut


def mutate_seq_random_n_residues(seq_wt: str, n: int) -> str:
    """
    Mutate n residues to random values, excluding first residue (M normally).
    The mutation is reproducible and specific to sequence.
    """
    # https://stackoverflow.com/a/42089311
    seed_for_seq = int(hashlib.sha256(seq_wt.encode('utf-8')).hexdigest(), 16) % 10**8
    rng = default_rng(seed=seed_for_seq)
    seq_rand = list(seq_wt)
    rand_positions = rng.integers(1, len(seq_wt), size=n)
    for pos in rand_positions:
        replacement_set = set(list(attribution.VOCAB_AMINO_ACIDS)) - set([seq_rand[pos]])
        replacement_set = list(sorted(replacement_set))
        seq_rand[pos] = rng.choice(replacement_set)
    seq_rand = ''.join(seq_rand)
    return seq_rand


def get_width_of_majority_ordering_range(projection_values: dict) -> float:
    """
    Return width of the interval with 99% of UMAP-projected embedded ordering values.
    :param projection_values: {prot_id: str -> projection_vector: np.array(len(seq))}
    """
    all_projection_values = list(map(lambda a: a.tolist(),
                                     projection_values.values()))
    all_projection_values = np.concatenate(all_projection_values)
    majority_range = (np.percentile(all_projection_values, q=1),
                      np.percentile(all_projection_values, q=99))
    len_majority_range = abs(abs(majority_range[0]) - abs(majority_range[1]))
    return len_majority_range


def parse_num_res_spec(amount_spec, seq) -> int:
    try:
        if '%' in amount_spec:
            return int(np.round(len(seq) * int(amount_spec[:-1]) / 100))
        else:
            return int(amount_spec)
    except ValueError as e:
        logger.error('Invalid format for n. residues: should be e.g. 5 or 5%')
        raise e


def get_output_filename(args):
    run_id = uuid.uuid1().hex
    if not (args.exp_dir / 'informed_mutation').exists():
        os.makedirs(str(args.exp_dir / 'informed_mutation'))
    output_fname = str(args.exp_dir / 'informed_mutation' / f'informed_mutation_{run_id}.csv')
    return output_fname


def test_guied_mutation():
    seq_wt = 'MTEST'
    o_wt = np.array([5, 3, 1, 2, 4])
    seq_guide = 'GUIDE'
    o_guide = np.array([8, 9, 10, 11, 12]).reshape(-1, 1)
    n_res_mut = 2
    o_inc = 10
    expected_seq_mut = 'MTDET'

    mut_positions = np.argsort(o_wt)[:n_res_mut]
    seq_mut = get_mutated_sequence(
        seq_wt=seq_wt, embedded_ordering_wt=o_wt, ordering_increase=o_inc,
        mut_positions=mut_positions,
        seq_guide=seq_guide, embedded_ordering_guide=o_guide
    )
    assert seq_mut == expected_seq_mut


def get_args():
    parser = argparse.ArgumentParser(description="Performed informed mutation on entire proteome")
    parser.add_argument('--config',
                        required=True,
                        type=str,
                        help='Experiment YAML config file')
    parser.add_argument('--num-mutated-residues',
                        required=True,
                        type=str,
                        help='N. residues to mutate, fixed num x or x%')
    parser.add_argument('--subsample',
                        required=False,
                        action='store_true',
                        help='Run on a random subset of 100 sequences')
    args = parser.parse_args()
    _ = parse_num_res_spec(args.num_mutated_residues, 'X' * 10)  # Early input test
    args.exp_dir = Path(args.config).parent
    with open(args.config, 'r') as config_file:
        args.config = yaml.load(config_file, Loader=yaml.FullLoader)
    for path in args.config['files']:
        args.config['files'][path] = str(args.exp_dir / args.config['files'][path])
    return args


if __name__ == '__main__':
    main()
