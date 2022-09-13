"""
Loads model and saved UMAP transformation of embedded space
and saves the embedded ordering for the given fasta file
"""
import argparse
from multiprocessing import cpu_count
from pathlib import Path

import h5py
import numexpr
from tqdm import tqdm
from umap.parametric_umap import load_ParametricUMAP
import tensorflow as tf
import yaml

from scripts.general import preprocess, util
from scripts.general.analysis import ModelEvaluator, embedded_ordering

numexpr.set_num_threads(cpu_count())

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass


def main():
    logger = util.get_logger()
    args = get_args()
    if Path(args.output).exists():
        logger.error('Output file exists. Stopping execution.')
        raise FileExistsError

    logger.info('Loading models and sequences...')
    sequences = preprocess.fasta_to_seq_df(args.fasta)
    model_eval = ModelEvaluator(args.config)
    umap_embedder = load_ParametricUMAP(args.config['files']['umap_encoder_embedding_dir'])
    logger.info('Loading done.')

    tqdm.pandas(desc='Obtaining embedded orderings')
    sequences = sequences.assign(
        ordering = sequences.progress_apply(
            lambda row: embedded_ordering(row['seq'], model_eval, umap_embedder),
            axis='columns'
        )
    )

    with h5py.File(args.output, 'w') as store:
        sequences.apply(
            lambda row: store_values(store, row),
            axis='columns'
        )
    logger.info('DONE: Wrote embedded orderings to ' + args.output)


def store_values(store, row):
    """Workaround wrapper to avoid weird h5py error thrown when using lambda"""
    store.create_dataset(row['swissprot_ac'], data=row['ordering'])


def get_args():
    """
    Standard parsing of command line argument specifying experiment config file
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c',
                        '--config',
                        required=True,
                        type=str,
                        help='Experiment YAML config file')
    parser.add_argument('-f',
                        '--fasta',
                        required=True,
                        type=str,
                        help='Input FASTA file')
    parser.add_argument('-o',
                        '--output',
                        required=True,
                        type=str,
                        help='Output h5py file with embedded ordering values for each sequence')
    args = parser.parse_args()
    args.exp_dir = Path(args.config).parent
    with open(args.config, 'r') as config_file:
        args.config = yaml.load(config_file, Loader=yaml.FullLoader)

    for path in args.config['files']:
        args.config['files'][path] = str(args.exp_dir / args.config['files'][path])
    return args


if __name__ == '__main__':
    main()
