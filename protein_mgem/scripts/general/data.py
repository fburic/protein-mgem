"""
Module with data management functionality
"""
import os
from pathlib import Path
import sys
from typing import Iterable, Union

import yaml
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.general import preprocess, util


def seq_df_to_fasta(seq: pd.DataFrame, id_col: str, seq_col: str, output_fname: str):
    fasta_records = []
    for _, entry in seq.iterrows():
        rec = SeqRecord(id=entry[id_col], seq=Seq(entry[seq_col]), description=entry[id_col])
        fasta_records.append(rec)

    with open(output_fname, 'w') as fout:
        SeqIO.write(fasta_records, fout, 'fasta')


def load_sequences_that_have_abundance(experiment_config: dict) -> dict:
    """
    Return dict of swissprot_ac -> sequence, only for proteins that have abundance values.
    Filenames are taken from the experiment config file.
    """
    seq_and_abundances = load_seq_and_abundance_data(experiment_config)
    sequences = dict(seq_and_abundances[['swissprot_ac', 'seq']].values)
    logger = util.get_logger('load_sequences_that_have_abundance')
    logger.info('Loaded sequences that have abundance')
    return sequences


def load_top_n_sequences(experiment_config: dict, n=10) -> dict:
    seq_and_abundances = load_seq_and_abundance_data(experiment_config)
    seq_and_abundances = seq_and_abundances.sort_values('Median_molecules_per_cell').tail(n)
    sequences = dict(seq_and_abundances[['swissprot_ac', 'seq']].values)
    return sequences


def load_seq_and_abundance_data(experiment_config: dict) -> pd.DataFrame:
    """
    Load protein sequences and abundance values from files specified in the
    experiment config file.
    Return merged data frame on 'swissprot_ac'

    TODO: Refactor with func loading from data config yaml used by TAPE
    """
    logger = util.get_logger('load_seq_and_abundance_data')
    try:
        all_sequences = preprocess.fasta_to_seq_df(
            experiment_config['files']['protein_sequences']
        )
        abundances = pd.read_csv(experiment_config['files']['protein_abundances'])

    except Exception as e:
        logger.error('Could not load data! Check the experiment config YAML')
        raise e

    seq_and_abundances = pd.merge(all_sequences, abundances, on='swissprot_ac')
    return seq_and_abundances


def load_input_data_as_df(data_config_filename: Union[str, Path]) -> pd.DataFrame:
    """
    Load protein sequences and abundance values from files specified in the
    data config file.
    Note: No target transformation done at this point.
    """
    with Path(data_config_filename).open('r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    data_root = os.path.expanduser(config['data_root'])
    fasta_fname = Path(data_root) / config['protein_sequence']
    abundances_fname = Path(data_root) / config['protein_abundance']
    seq_and_abundances = pd.merge(preprocess.fasta_to_seq_df(fasta_fname),
                                  pd.read_csv(abundances_fname),
                                  on=config.get('merge_col', 'swissprot_ac'))
    seq_and_abundances = seq_and_abundances[
        [config.get('input', 'seq'),
         config.get('target', 'Median_molecules_per_cell')]
    ]
    return seq_and_abundances


def get_dssp_annotation(pdb_file: str, seq_len: int, log_exceptions=False) -> str:
    """
    Return amino acid sequence annotation with secondary structure, using DSSP.
    Use only first model only in the PDB file.

    Coil is denoted as '-' and missing residues are denoted as '_'
    """
    try:
        dssp_dict = dssp_dict_from_pdb_file(str(pdb_file))[0]
        annotation = ['_'] * seq_len
        for res_id, res_features in dssp_dict.items():
            res_idx = res_id[1][1]
            struct = res_features[1]
            annotation[res_idx - 1] = struct
        return ''.join(annotation)

    except IndexError as e:
        if log_exceptions:
            print('Residue indexing error for ' + str(pdb_file), file=sys.stderr)
        return None

    except Exception as e:
        if log_exceptions:
            print('Error running DSSP on ' + str(pdb_file), file=sys.stderr)
        return None


def split_on_unique_inputs(inputs: Union[Iterable, np.array],
                           targets: Union[Iterable, np.array],
                           test_size: float,
                           shuffle=True,
                           random_state=42) -> tuple:
    """
    Wrapper around from sklearn.model_selection import train_test_split()
    to ensure the splitting keeps input repeats in the same partitions.
    """
    dataset = pd.DataFrame(zip(inputs, targets), columns=['x', 'y'])

    try:
        target_dtype = targets.dtype
    except Exception:
        target_dtype = None

    dataset = dataset.groupby('x')['y'].apply(list).reset_index()
    data_train, data_test = train_test_split(dataset,
                                             test_size=test_size,
                                             shuffle=shuffle,
                                             random_state=random_state)
    data_train = data_train.explode('y')
    data_test = data_test.explode('y')

    if target_dtype is not None:
        data_train['y'] = data_train['y'].astype(target_dtype)
        data_test['y'] = data_test['y'].astype(target_dtype)

    return data_train['x'].values, data_test['x'].values, \
           data_train['y'].values, data_test['y'].values


def test_split_on_unique_inputs_repeats():
    Y_DTYPE = 'uint8'
    X_train, X_test, y_train, y_test = split_on_unique_inputs(
        ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E'],
        np.array([1, 2, 10, 20, 3, 4, 30, 40, 8, 9], dtype=Y_DTYPE),
        test_size=0.2, shuffle=True, random_state=42
    )
    assert np.array_equal(
        X_train, np.array(['E', 'E', 'C', 'C', 'A', 'A', 'D', 'D'], dtype=object)
    )
    assert np.array_equal(
        X_test, np.array(['B', 'B'], dtype=object)
    )
    assert np.array_equal(
        y_train, np.array([8, 9, 3, 4, 1, 2, 30, 40], dtype=Y_DTYPE)
    )
    assert np.array_equal(
        y_test, np.array([10, 20], dtype=Y_DTYPE)
    )
    assert y_train.dtype == Y_DTYPE


def test_split_on_unique_inputs_no_repeats():
    Y_DTYPE = 'uint8'
    X_train, X_test, y_train, y_test = split_on_unique_inputs(
        ['A', 'B', 'C', 'D', 'E', 'F'],
        np.array([1, 2, 3, 4, 5, 6], dtype=Y_DTYPE),
        test_size=0.2, shuffle=True, random_state=42
    )
    assert np.array_equal(
        X_train, np.array(['F', 'C', 'E', 'D'], dtype=object)
    )
    assert np.array_equal(
        X_test, np.array(['A', 'B'], dtype=object)
    )
    assert np.array_equal(
        y_train, np.array([6, 3, 5, 4], dtype=Y_DTYPE)
    )
    assert np.array_equal(
        y_test, np.array([1, 2], dtype=Y_DTYPE)
    )
    assert y_train.dtype == Y_DTYPE
