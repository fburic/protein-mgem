import os
from pathlib import Path

import pandas as pd
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql.types import *

from scripts.general import preprocess


conf = SparkConf().setAppName('interpro_filter')
conf = conf.setMaster("local[*]")
conf = conf.set('spark.local.dir', os.getenv('TMPDIR'))
conf = conf.set('spark.executor.memory', '2500M').\
    set('spark.driver.memory', '10G')

sc = SparkContext(conf=conf)
sqlc = SQLContext(sc)

yeast_ids = preprocess.fasta_to_seq_df(
    'data/seq/scerevisiae_aminoacid_uniprot_20200120.fasta'
)['swissprot_ac'].values

schema = StructType([StructField("uniprot_ac", StringType()),
                     StructField("entry_ac", StringType()),
                     StructField("entry_name", StringType()),
                     StructField("db_id", StringType()),
                     StructField("start", StringType()),
                     StructField("end", StringType())])
ip_prots = sqlc.read.csv('data/interpro/protein2ipr.tsv', sep='\t', schema=schema, header=False)

ip_prots = ip_prots.filter(ip_prots['uniprot_ac'].isin(set(yeast_ids)))
(
    ip_prots
        .write.format('com.databricks.spark.csv')
        .option('header', 'true').csv('data/interpro/yeast_protein2ipr')
)


csv_files = list(Path('yeast_protein2ipr').glob('*.csv'))
shards = [pd.read_csv(fpath) for fpath in csv_files]
yeast_ip = pd.concat(shards, axis=0, ignore_index=True)
yeast_ip.to_csv('yeast_protein2ipr.csv.gz', index=False, compression='gzip')
