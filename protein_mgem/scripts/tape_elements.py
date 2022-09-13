import os
from collections import OrderedDict
from typing import Union, List, Tuple, Any, Dict, Iterable
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from torch.utils.data import Dataset
from tape.datasets import pad_sequences
from tape.registry import registry
from tape.tokenizers import TAPETokenizer
from tape.models.modeling_bert import *

from scripts.general import util
from scripts.general.data import load_input_data_as_df, split_on_unique_inputs
from scripts.general.preprocess import fasta_to_seq_df
from scripts.tape_fixes import *


@registry.register_task('learn_abundance')
class SequenceAbundanceDataset(Dataset):
    """
    Dataset consisting of protein sequences and abundances.
    Relies on a config.yaml placed in data_path to specify input files.

    This class has the IUPAC tokenizer vocabulary hardcoded but in principle any
    sequence data can be used, provided a derived class uses an appropriate tokenizer.

    The Box-Cox lambda to transform the abundances is hardcoded -0.05155
    for reproducibility. The value was determined automatically by the EM procedure
    used by the scipy boxcox function.

    The train:valid:test ratios are hardcoded.
    """
    def init_tokenizer(self):
        self.tokenizer = TAPETokenizer(vocab='iupac')

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = True):
        """
        tokenizer and in_memory are not used in this derived class but specified
        for compatibility with TAPE usage
        """
        self.init_tokenizer()
        sequences, abundances = self.prepare_data(data_path)

        random_seed = 42
        (sequences_train, sequences_test,
         abundances_train, abundances_test) = split_on_unique_inputs(sequences, abundances,
                                                                     test_size=0.2,
                                                                     shuffle=True,
                                                                     random_state=random_seed)
        (sequences_valid, sequences_test,
         abundances_valid, abundances_test) = split_on_unique_inputs(sequences_test, abundances_test,
                                                                     test_size=0.5,
                                                                     shuffle=True,
                                                                     random_state=random_seed)
        if split == 'train':
            data = (sequences_train.reshape(-1, 1), abundances_train.reshape(-1, 1))
        elif split == 'valid':
            data = (sequences_valid.reshape(-1, 1), abundances_valid.reshape(-1, 1))
        elif split == 'test':
            data = (sequences_test.reshape(-1, 1), abundances_test.reshape(-1, 1))
        else:
            raise ValueError(f'Unrecognized split: {split}.')
        self.data = np.concatenate(data, axis=1)

    def prepare_data(self, data_path: str) -> tuple:
        """
        Load and process data, applying any preprocessing deemed appropriate.
        Separated since may change between TAPE tasks
        """
        seq_and_abundances = load_input_data_as_df(Path(data_path) / 'config.yaml')
        sequences = self.preprocess_sequences(seq_and_abundances['seq'].values)
        abundances = self.preprocess_targets(seq_and_abundances['Median_molecules_per_cell'].values)
        return sequences, abundances

    def preprocess_sequences(self, sequences: Iterable) -> Iterable:
        return sequences

    def preprocess_targets(self, targets: Iterable) -> Iterable:
        """
        Box-Cox-transform target values, then clip negative values to zero.
        The latter is done to set counter-examples, introduced with value
        0 < eps < 1 in the input data (Box-Cox only defined for positives).
        """
        targets_transformed = stats.boxcox(targets, lmbda=-0.05155)
        targets_transformed[targets_transformed < 0] = 0
        return targets_transformed

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        sequence, abundance = self.data[item]
        token_ids = self.tokenizer.encode(sequence)
        attention_mask = np.ones_like(token_ids)
        return token_ids, attention_mask, abundance

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        """
        Convert the variable length sequences into a batch of torch tensors.
        Token ids and mask should be padded with zeros.
        This takes in a list of outputs from the dataset's __getitem__
        method.
        """
        input_ids, attention_mask, abundance = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        attention_mask = torch.from_numpy(pad_sequences(attention_mask, 0))
        abundance = torch.FloatTensor(abundance).unsqueeze(1)

        output = {'input_ids': input_ids,
                  'input_mask': attention_mask,
                  'targets': abundance}

        return output


@registry.register_task('learn_human_abundance')
class HumanSequenceAbundanceDataset(SequenceAbundanceDataset):
    def prepare_data(self, data_path: str) -> tuple:
        """
        Load and process data, applying any preprocessing deemed appropriate.
        Separated since may change between TAPE tasks
        """
        with Path(Path(data_path) / 'config.yaml').open('r') as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        data_root = os.path.expanduser(config['data_root'])
        fasta_fname = Path(data_root) / config['protein_sequence']
        abundances_fname = Path(data_root) / config['protein_abundance']
        seq_and_abundances = pd.merge(fasta_to_seq_df(fasta_fname),
                                      pd.read_csv(abundances_fname),
                                      on='swissprot_ac')[
            ['seq', 'copy_number']
        ]
        print(f'Num. seq: {seq_and_abundances.shape[0]}')
        sequences = self.preprocess_sequences(seq_and_abundances['seq'].values)
        abundances = self.preprocess_targets(seq_and_abundances['copy_number'].values)
        return sequences, abundances

    def preprocess_targets(self, targets: Iterable) -> Iterable:
        """
        Box-Cox-transform target values, then clip negative values to zero.
        The latter is done to set counter-examples, introduced with value
        0 < eps < 1 in the input data (Box-Cox only defined for positives).
        """
        targets_transformed, lmbda = stats.boxcox(targets)
        print(f'INFO: tape_elements.preprocess_targets(): Targets Box-Cox lambda = {lmbda}')
        targets_transformed[targets_transformed < 0] = 0
        return targets_transformed


@registry.register_task('learn_human_tissue_abundance')
class HumanTissueSequenceAbundanceDataset(HumanSequenceAbundanceDataset):
    def prepare_data(self, data_path: str) -> tuple:
        """
        Load and process data, applying any preprocessing deemed appropriate.
        Separated since may change between TAPE tasks
        """
        with Path(Path(data_path) / 'config.yaml').open('r') as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        data_root = os.path.expanduser(config['data_root'])
        fasta_fname = Path(data_root) / config['protein_sequence']
        abundances_fname = Path(data_root) / config['protein_abundance']
        model_tissue = config['tissue']

        seq_and_abundances = pd.merge(fasta_to_seq_df(fasta_fname),
                                      pd.read_csv(abundances_fname),
                                      on='swissprot_ac')[
            ['tissue', 'seq', 'copy_number']
        ]
        seq_and_abundances = seq_and_abundances.query(f'tissue == "{model_tissue}"')
        print(f'Filtered on tissue: {model_tissue}')
        print(f'Num. seq: {seq_and_abundances.shape[0]}')

        sequences = self.preprocess_sequences(seq_and_abundances['seq'].values)
        abundances = self.preprocess_targets(seq_and_abundances['copy_number'].values)
        return sequences, abundances

    def preprocess_targets(self, targets: Iterable) -> Iterable:
        """
        Box-Cox-transform target values, then clip negative values to zero.
        The latter is done to set counter-examples, introduced with value
        0 < eps < 1 in the input data (Box-Cox only defined for positives).
        """
        targets_transformed = stats.boxcox(targets)[0]
        targets_transformed[targets_transformed < 0] = 0
        return targets_transformed


@registry.register_task('learn_mrna_abundance_from_regulatory_regions')
class RegulatoryRegionsMRNADataset(SequenceAbundanceDataset):
    def init_tokenizer(self):
        self.tokenizer = DNATokenizer()

    def prepare_data(self, data_path: str) -> tuple:
        """
        Load and process data, applying any preprocessing deemed appropriate.
        Separated since may change between TAPE tasks
        """
        with (Path(data_path) / 'config.yaml').open('r') as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        data_root = os.path.expanduser(config['data_root'])
        seq_and_abundances = pd.merge(
            pd.read_csv(Path(data_root) / config['gene_sequence']),
            pd.read_csv(Path(data_root) / config['mrna_abundance']),
            on='gene_id'
        )
        sequences = self.preprocess_sequences(seq_and_abundances)
        abundances = seq_and_abundances['tpm_median_boxcox'].values
        return sequences, abundances

    def preprocess_sequences(self, sequences: pd.DataFrame) -> Iterable:
        """
        Concatenate region sequences, separated by space, to be split later.
        """
        regions_seq = sequences.apply(
            lambda row: ' '.join(
                [row['prom_seq'], row['utr5_seq'], row['utr3_seq'], row['term_seq']]
            ),
            axis='columns'
        )
        return regions_seq.values

    def preprocess_targets(self, targets: Iterable) -> Iterable:
        """NOP, since targets preprocessed"""
        return targets


@registry.register_task('learn_mrna_abundance_from_gene_regions')
class FullGeneMRNADataset(RegulatoryRegionsMRNADataset):
    def init_tokenizer(self):
        self.tokenizer = CodonExtendedTokenizer()

    def preprocess_sequences(self, sequences: pd.DataFrame) -> Iterable:
        """
        Concatenate region sequences, separated by space, to be split later.
        """
        regions_seq = sequences.apply(
            lambda row: ' '.join(
                [row['prom_seq'], row['utr5_seq'], row['cds_seq'], row['utr3_seq'], row['term_seq']]
            ),
            axis='columns'
        )
        return regions_seq.values


@registry.register_task('learn_prot_abundance_from_regulatory_regions')
class RegulatoryRegionsProteinAbundanceDataset(SequenceAbundanceDataset):
    def init_tokenizer(self):
        self.tokenizer = CodonExtendedTokenizer()

    def prepare_data(self, data_path: str) -> tuple:
        """
        Load and process data, applying any preprocessing deemed appropriate.
        Separated since may change between TAPE tasks
        """
        with (Path(data_path) / 'config.yaml').open('r') as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        data_root = os.path.expanduser(config['data_root'])

        seq_and_abundances = pd.merge(
            pd.merge(
                pd.read_csv(Path(data_root) / config['gene_sequence']),
                pd.read_csv(Path(data_root) / config['gene_id_correspondence']),
                left_on = 'gene_id', right_on='OLN'
            ),
            pd.read_csv(Path(data_root) / config['protein_abundance']),
            on='swissprot_ac'
        )
        sequences = self.preprocess_sequences(seq_and_abundances)
        abundances = self.preprocess_targets(seq_and_abundances['Median_molecules_per_cell'].values)
        return sequences, abundances

    def preprocess_sequences(self, sequences: pd.DataFrame) -> Iterable:
        """
        Concatenate region sequences, separated by space, to be split later.
        """
        regions_seq = sequences.apply(
            lambda row: ' '.join(
                [row['prom_seq'], row['utr5_seq'], row['utr3_seq'], row['term_seq']]
            ),
            axis='columns'
        )
        return regions_seq.values

    def preprocess_targets(self, targets: Iterable) -> Iterable:
        targets_transformed, boxcox_lambda = stats.boxcox(targets)
        logger = util.get_logger('preprocess_targets()')
        logger.info(f'Protein abundance Box-Cox lambda = {boxcox_lambda}')
        return targets_transformed


CODON_ASCII_OFFSET = 63
codon_symbols = [(chr(i), 5 + i - CODON_ASCII_OFFSET)
                 for i in range(CODON_ASCII_OFFSET, CODON_ASCII_OFFSET + 64)]

CODON_VOCAB = OrderedDict(
    [("<pad>", 0), ("<mask>", 1), ("<cls>", 2), ("<sep>", 3), ("<unk>", 4)]
     + codon_symbols)

codon_symbols_extended = [
    ("A", 5), ("C", 6), ("G", 7), ("T", 8)
] + [
    (chr(c), i + 8)
    for i, c in enumerate(list(range(33, 61)) + list(range(87, 90)) + list(range(93, 126)))
]

CODON_VOCAB_EXTENDED = OrderedDict(
    [("<pad>", 0), ("<mask>", 1), ("<cls>", 2), ("<sep>", 3), ("<unk>", 4)]
     + codon_symbols_extended)

DNA_VOCAB =  OrderedDict(
    [("<pad>", 0), ("<mask>", 1), ("<cls>", 2), ("<sep>", 3), ("<unk>", 4),
     ("A", 5), ("C", 6), ("G", 7), ("T", 8)]
)


TOY_VOCAB_1 = OrderedDict(
    [("<pad>", 0), ("<mask>", 1), ("<cls>", 2), ("<sep>", 3), ("<unk>", 4),
     ("A", 5), ("B", 6), ("C", 7), ("D", 8)])


class CodonTokenizer(TAPETokenizer):
    """
    Tokenizer for ASCII-encoded codons.
    Includes the same special tokens as the IUPAC vocab.
    """
    def __init__(self):
        self.vocab = CODON_VOCAB
        self.tokens = list(self.vocab.keys())
        self._vocab_type = 'codons'
        assert self.start_token in self.vocab and self.stop_token in self.vocab


class CodonExtendedTokenizer(TAPETokenizer):
    """
    Tokenizer for ASCII-encoded codons and DNA letters.
    Includes the same special tokens as the IUPAC vocab.
    """
    def __init__(self):
        self.vocab = CODON_VOCAB_EXTENDED
        self.tokens = list(self.vocab.keys())
        self._vocab_type = 'codons_and_dna'
        assert self.start_token in self.vocab and self.stop_token in self.vocab

    def add_special_tokens(self, token_ids: List[str]) -> List[str]:
        """
        Replace regions separator (' ') with [SEP] tokens
        """
        cls_token = [self.start_token]
        sep_token = [self.stop_token]
        token_ids = map(lambda token: self.stop_token if token == ' ' else token,
                        token_ids)
        return cls_token + list(token_ids) + sep_token


class DNATokenizer(CodonExtendedTokenizer):
    def __init__(self):
        self.vocab = DNA_VOCAB
        self.tokens = list(self.vocab.keys())
        self._vocab_type = 'dna'
        assert self.start_token in self.vocab and self.stop_token in self.vocab


class ToyVocabTokenizer(TAPETokenizer):
    """
    Tokenizer for a toy vocabulary.
    Includes the same special tokens as the IUPAC vocab.
    """
    def __init__(self):
        self.vocab = TOY_VOCAB_1
        self.tokens = list(self.vocab.keys())
        self._vocab_type = 'toy'
        assert self.start_token in self.vocab and self.stop_token in self.vocab


@registry.register_task('learn_abundance_from_codons')
class CodonSequenceAbundanceDataset(SequenceAbundanceDataset):
    def init_tokenizer(self):
        self.tokenizer = CodonTokenizer()


@registry.register_task('learn_toy_data')
class ToySequenceAbundanceDataset(SequenceAbundanceDataset):
    def init_tokenizer(self):
        self.tokenizer = ToyVocabTokenizer()


if version.parse(torch.__version__) > version.parse('1.4'):
    tape_model_name = 'transformer_parallel'
    tape_model_class = ProteinBertForValuePrediction_parallel
else:
    tape_model_name = 'transformer'
    tape_model_class = ProteinBertForValuePrediction


# Register the learning tasks using the Dataset classes defined above.
registry.register_task_model('learn_abundance', tape_model_name,
                             tape_model_class)
registry.register_task_model('learn_human_abundance', tape_model_name,
                             tape_model_class)
registry.register_task_model('learn_human_tissue_abundance', tape_model_name,
                             tape_model_class)
registry.register_task_model('learn_abundance_from_codons', tape_model_name,
                             tape_model_class)
registry.register_task_model('learn_mrna_abundance_from_regulatory_regions', tape_model_name,
                             tape_model_class)
registry.register_task_model('learn_mrna_abundance_from_gene_regions', tape_model_name,
                             tape_model_class)
registry.register_task_model('learn_prot_abundance_from_regulatory_regions', tape_model_name,
                             tape_model_class)
registry.register_task_model('learn_toy_data', tape_model_name,
                             tape_model_class)
