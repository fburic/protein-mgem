import argparse
import json
import random
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import h5py
import numexpr
import numpy as np
import pandas as pd
import yaml
from scipy import stats
import torch
from tape import ProteinBertForValuePrediction, TAPETokenizer
from tqdm import tqdm

from scripts.general import attribution, preprocess, util

numexpr.set_num_threads(cpu_count())


def main():
    logger = util.get_logger()
    args = util.get_args()
    logger.info(f"Redundancy corr threshold for attention weights {args.config['attention_weights_redundancy_corr_threshold']}")

    res_dir = args.exp_dir / 'attention_patterns'
    if not res_dir.exists():
        res_dir.mkdir()
    logger.info(f'Profile and patterns will be written to {res_dir}')

    logger.info('Loading sequences...')
    sequences = load_sequences(args)

    logger.info('Computing attention values...')
    calc_attention_values(args, sequences, res_dir)

    logger.info('Extracting attention value patterns...')
    for seq_id in tqdm(sequences.keys()):
        try:
            extract_attention_patterns(seq_id, sequences=sequences, res_dir=res_dir)
        except Exception as e:
            logger.warning(str(e))
    logger.info('Done.')


def calc_attention_values(args, sequences: dict, results_dir: str):
    """
    Evaluate model and keep non-redundant attention matrices.
    Output a file for each protein.

    Note: Done as sequential batch since the model evaluation is a heavy process.
    """
    RND_SEED = 42
    torch.manual_seed(RND_SEED)
    np.random.seed(RND_SEED)
    random.seed(RND_SEED)
    torch.backends.cudnn.benchmark = False
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(RND_SEED)

    model = load_model(args)
    for seq_id, seq in tqdm(sequences.items()):
        attention, grad_attention = attribution.get_attention_values_for_sequence(
            seq, model['model'], model['arch']
        )

        (grad_attention_tensor,
         grad_attention_meta) = attribution.remove_redundant_attention_gradients(
            grad_attention,
            low_corr_threshold = float(args.config['attention_weights_redundancy_corr_threshold'])
        )

        # Only take those layers and heads kept in the grad_attention_tensor
        attention_tensor = []
        for layer, head in grad_attention_meta[['layer', 'head']].values:
            attention_tensor.append(attention[layer][head])
        attention_tensor = np.array(attention_tensor, dtype=np.float32)

        res_fname = results_dir / f'attention_{seq_id}.h5'
        with h5py.File(res_fname, 'w') as fres:
            group = fres.create_group(seq_id)
            group.create_dataset('attention', data=attention_tensor)
            group.create_dataset('grad_attention', data=grad_attention_tensor)
        grad_attention_meta.to_hdf(res_fname, key=f'{seq_id}/meta', mode='a')


def extract_attention_patterns(seq_id: str, sequences: dict, res_dir: str):
    """
    Extract all attention and attention gradient profiles and patterns
    for single sequence and write these to the protein file.
    """
    res_fname = res_dir / f'attention_{seq_id}.h5'
    with h5py.File(res_fname, 'r') as fres:
        attention = np.array(fres[f'{seq_id}/attention'])
        grad_attention = np.array(fres[f'{seq_id}/grad_attention'])

    attending_profiles = get_avg_profiles(attention, axis=1)
    attended_profiles = get_avg_profiles(attention, axis=0)
    attending_patterns = np.array(
        extract_all_patterns_for_sequence((sequences[seq_id], attending_profiles)),
        dtype='S'
    )
    attended_patterns = np.array(
        extract_all_patterns_for_sequence((sequences[seq_id], attended_profiles)),
        dtype='S'
    )

    grad_attending_profiles = get_avg_profiles(grad_attention, axis=1)
    grad_attended_profiles = get_avg_profiles(grad_attention, axis=0)
    grad_attending_patterns = np.array(
        extract_all_patterns_for_sequence((sequences[seq_id], grad_attending_profiles)),
        dtype='S'
    )
    grad_attended_patterns = np.array(
        extract_all_patterns_for_sequence((sequences[seq_id], grad_attended_profiles)),
        dtype='S'
    )

    with h5py.File(res_fname, 'a') as fres:
        fres.create_dataset(f'{seq_id}/attending_profiles', data=attending_profiles)
        fres.create_dataset(f'{seq_id}/attended_profiles', data=attended_profiles)
        fres.create_dataset(f'{seq_id}/attending_patterns', data=attending_patterns)
        fres.create_dataset(f'{seq_id}/attended_patterns', data=attended_patterns)

        fres.create_dataset(f'{seq_id}/grad_attending_profiles', data=grad_attending_profiles)
        fres.create_dataset(f'{seq_id}/grad_attended_profiles', data=grad_attended_profiles)
        fres.create_dataset(f'{seq_id}/grad_attending_patterns', data=grad_attending_patterns)
        fres.create_dataset(f'{seq_id}/grad_attended_patterns', data=grad_attended_patterns)


#TODO: Refactor
def model_output_for_sequences(config_filename: str,
                               sequences: dict,
                               low_corr_threshold: float,
                               silent: bool = False) -> dict:
    """Convenience function for explorative analysis."""
    RND_SEED = 42
    torch.manual_seed(RND_SEED)
    np.random.seed(RND_SEED)
    random.seed(RND_SEED)
    torch.backends.cudnn.benchmark = False
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(RND_SEED)

    args = argparse.Namespace()
    args.exp_dir = Path(config_filename).parent.parent
    with open(config_filename, 'r') as config_file:
        args.config = yaml.load(config_file, Loader=yaml.FullLoader)
    for path in args.config['files']:
        args.config['files'][path] = str(args.exp_dir / args.config['files'][path])
    model = load_model(args)

    predictions = []
    attention_profiles = []
    focus_profiles = []
    seq_attention_patterns = []
    seq_focus_patterns = []

    for seq_id, seq in tqdm(sequences.items(), disable=silent):
        tokenizer = TAPETokenizer(vocab='iupac')
        input_ids = torch.tensor([tokenizer.encode(seq)])

        pred, attention = model['model'](input_ids)
        pred.backward(retain_graph=True)
        grad_attention = torch.autograd.grad(pred, attention)

        attention_tensor = np.zeros((model['arch']['num_hidden_layers'],
                                     model['arch']['num_attention_heads'],
                                     len(seq), len(seq)),
                                    dtype=np.float32)
        grad_attention_tensor = np.zeros((model['arch']['num_hidden_layers'],
                                          model['arch']['num_attention_heads'],
                                          len(seq), len(seq)),
                                         dtype=np.float32)

        for layer_num in range(model['arch']['num_hidden_layers']):
            for head_num in range(model['arch']['num_attention_heads']):
                grad_attention_matrix = grad_attention[layer_num][0][head_num].detach().numpy()
                grad_attention_tensor[layer_num][head_num] = grad_attention_matrix[1:-1, 1:-1]
                attention_matrix = attention[layer_num][0][head_num].detach().numpy()
                attention_tensor[layer_num][head_num] = attention_matrix[1:-1, 1:-1]

        attention_unique, _ = attribution.remove_redundant_attention_gradients(attention_tensor,
                                                                               low_corr_threshold=low_corr_threshold)
        grad_attention_unique, _ = attribution.remove_redundant_attention_gradients(grad_attention_tensor,
                                                                                    low_corr_threshold=low_corr_threshold)

        attended_profiles = get_avg_profiles(attention_unique, axis=0)
        attended_patterns = extract_all_patterns_for_sequence((sequences[seq_id], attended_profiles))
        attended_patterns = np.array(attended_patterns, dtype='S')

        attended_focus_profiles = get_avg_profiles(grad_attention_unique, axis=0)
        attended_focus_patterns = extract_all_patterns_for_sequence((sequences[seq_id], attended_focus_profiles))
        attended_focus_patterns = np.array(attended_focus_patterns, dtype='S')

        predictions.append(pred.detach().numpy()[0][0])

        attention_profiles.append(attended_profiles)
        focus_profiles.append(attended_focus_profiles)

        seq_attention_patterns.append(attended_patterns)
        seq_focus_patterns.append(attended_focus_patterns)

    return {
        'predictions': predictions,
        'attention_profiles': attention_profiles,
        'focus_profiles': focus_profiles,
        'seq_attention_patterns': attended_patterns,
        'seq_focus_patterns': seq_focus_patterns
    }


def get_avg_profiles(grad_attention_tensor, axis) -> list:
    """
    Return all non-zero attention profiles for grad_attention_tensor matrices.
    """
    profiles = []
    for grad_attention in grad_attention_tensor:
        profile = grad_attention.mean(axis=axis)
        for value in profile:
            if ~np.isclose(value, 0.0):
                profiles.append(profile)
                break
    return profiles


def extract_all_patterns_for_sequence(seq_profiles_pair: tuple) -> list:
    """
    Return sequence letters at positions where the profile has significant values
    (i.e. outside of k standard deviations , k = 1 by default)
    """
    sequence, profiles = seq_profiles_pair
    profiles = map(attribution.keep_only_significant_values, profiles)
    patterns = map(partial(attribution.extract_sequence_pattern, sequence=sequence),
                   profiles)
    return list(patterns)


def load_sequences(args: argparse.Namespace) -> dict:
    """Return dict of swissprot_ac -> sequence"""
    all_sequences = preprocess.fasta_to_seq_df(args.config['files']['protein_sequences'])
    abundances = pd.read_csv(args.config['files']['protein_abundances'])
    seq_and_abundances = pd.merge(all_sequences, abundances, on='swissprot_ac')[
        ['swissprot_ac', 'seq', 'Median_molecules_per_cell']
    ]
    sequences = dict(seq_and_abundances[['swissprot_ac', 'seq']].values)
    return sequences


def load_model(args: argparse.Namespace) -> dict:
    """Return dict with model, arch, and tokenizer"""
    with Path(args.config['files']['arch']).open('r') as arch_file:
        arch = json.load(arch_file)
    model = ProteinBertForValuePrediction.from_pretrained(str(args.config['files']['model_checkpoint']),
                                                          output_attentions=True)
    model.eval()
    return {'arch': arch, 'model': model, 'tokenizer': TAPETokenizer(vocab='iupac')}


def test_extract_sequence_patterns():
    sequence = 'PEPTIDE'
    profiles = [np.array([0.0, 1.0, 0.5, 0.0, 0.0, -0.9, 0.0]),
                np.array([1e-4, 1.0, 1e-10, 0.0, 0.0, -0.9, 0.0])]
    patterns = list(map(partial(attribution.extract_sequence_pattern, sequence=sequence),
                        profiles))
    assert patterns[0] == '-EP--D-'
    assert patterns[1] == 'PE---D-'


def test_pattern_retrieval():
    RND_SEED = 42
    torch.manual_seed(RND_SEED)
    np.random.seed(RND_SEED)
    random.seed(RND_SEED)
    torch.backends.cudnn.benchmark = False
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(RND_SEED)

    args = argparse.Namespace()
    args.config = {
        'files': {
            'model_checkpoint': 'results/20201109_135748/learn_abundance_transformer_20-11-10-11-04-23_490645',
            'arch': 'results/20201109_135748/arch_2.json'
        }
    }

    model = load_model(args)
    sequence = 'GCTVEDRCLIGMGAILLNGCVIGSGSLVAAGALITQ'
    _, grad_attention = attribution.get_attention_values_for_sequence(sequence,
                                                                      model['model'],
                                                                      model['arch'])
    TOL = 1e-2
    assert np.isclose(grad_attention.mean(), 0.000112, atol=TOL)
    assert np.isclose(grad_attention.std(), 0.000895164, atol=TOL)
    assert np.isclose(grad_attention.sum(), 11.616196, atol=TOL)

    (grad_attention,
     grad_attention_meta) = attribution.remove_redundant_attention_gradients(grad_attention,
                                                                             low_corr_threshold=0.01)
    assert np.isclose(grad_attention.mean(), 6.966689e-05, atol=TOL)
    assert np.isclose(grad_attention.std(), 0.0001823742, atol=TOL)
    assert np.isclose(grad_attention.sum(), 0.63201797, atol=TOL)
    assert set(grad_attention_meta['idx']) == {64, 65, 66, 67, 68, 69, 70}

    # Test last attention gradient pattern
    attends_to_all_profile = stats.zscore(grad_attention[-1].mean(axis=1))
    attended_by_all_profile = stats.zscore(grad_attention[-1].mean(axis=0))

    attends_to_all_profile = attribution.keep_only_significant_values(attends_to_all_profile,
                                                                      num_sd=1)
    attended_by_all_profile = attribution.keep_only_significant_values(attended_by_all_profile,
                                                                       num_sd=1)
    attending_pattern = attribution.extract_sequence_pattern(attends_to_all_profile, sequence)
    attended_pattern = attribution.extract_sequence_pattern(attended_by_all_profile, sequence)

    assert attended_pattern == '----EDR-------I--N---I-SG------ALI-Q'
    assert attending_pattern == '-C----R--------------------------I--'


def test_complete_pattern_extraction_procedure():
    res_dir = Path('/tmp')
    sequences = {
        'A': 'GCTVEDRCLIGMGAILLNGCVIGSGSLVAAGALITQ',
        'B': 'PEPTIDEPEPTIDEPEPTIDEPEPTIDE'
    }

    args = argparse.Namespace()
    args.config = {
        'files': {
            'model_checkpoint': 'results/20201109_135748/learn_abundance_transformer_20-11-10-11-04-23_490645',
            'arch': 'results/20201109_135748/arch_2.json'
        },
        'attention_weights_redundancy_corr_threshold': 0.01
    }
    calc_attention_values(args, sequences, res_dir)

    with Pool(processes=2) as pool:
        pool.map(partial(extract_attention_patterns, sequences=sequences, res_dir=res_dir),
                 sequences.keys())

    with h5py.File(res_dir / 'attention_A.h5', 'r') as fres:
        attended_pattern = np.array(fres['A/grad_attended_patterns'])
        attending_pattern = np.array(fres['A/grad_attending_patterns'])

    assert attended_pattern[-1].decode('ascii') == '----EDR-------I--N---I-SG------ALI-Q'
    assert attending_pattern[-1].decode('ascii') == '-C----R--------------------------I--'


if __name__ == '__main__':
    main()
