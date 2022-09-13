"""
Collection of preprocessing functions.
Whatever's not provided by standard ML libs
"""
import gzip
from pathlib import Path
from typing import Union

from Bio import SeqIO
import pandas as pd


def fasta_to_seq_df(filename: Union[str, Path],
                    id_name:str = 'swissprot_ac',
                    id_extract_func=None) -> pd.DataFrame:
    """
    Read an (optionally gzipped) FASTA file into a DataFrame.

    A custom seq id extraction function cane be given, otherwise the default
    splits the ID at | and takes the second element (following UniProt convention),
    e.g. sp|Pxxxx|info  -> id: Pxxxx

    :param id_name: What to name the ID column.
    """
    filename = str(filename)
    if filename.endswith('.gz'):
        with gzip.open(filename, 'rt') as zipped_fasta:
            seq_records = list(SeqIO.parse(zipped_fasta, 'fasta'))
    else:
        seq_records = list(SeqIO.parse(str(filename), 'fasta'))

    if id_extract_func is None:
        def id_extract_func(record):
            return record.id.split('|')[1]

    seq_df = pd.DataFrame(
        [(id_extract_func(rec), str(rec.seq)) for rec in seq_records],
        columns=[id_name,'seq']
    )
    return seq_df
