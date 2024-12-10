import os
from pathlib import Path
import glob
import tqdm
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", type=str, help = "input directory")
parser.add_argument("--sequence_dir", type=str, help = "directory with sequence files")
parser.add_argument("-o", "--output", type=str, help = "output directory")
parser.add_argument("--tmp", type = str, default =  "tmp")

parser.add_argument("--overlap", type=int, default=1000, help="Minimul overlap between experiments")
parser.add_argument("--threashold", type=float, default = 0.8, help="Minimum pairwise correlation")
parser.add_argument("--min_clique", type=int, default = 3, help = "Minimum clique size for data set")

from Bio import SeqIO

import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx

import matplotlib as mpl
import matplotlib.pyplot as plt

def read_paxdb_dataset(path) -> pd.DataFrame:
    dset = pd.read_csv(path, sep = '\t', header=11)
    dset = dset.rename(columns = {'#string_external_id': 'external_id'})
    dset['external_id'] = dset['external_id'].map(str)
    dset['external_id'] = dset['external_id'].map(lambda s: s.split('.')[1])
    return dset

def read_experiments(DIR) -> list:
    organism_dir = Path(DIR)
    experiment_list = list(map(read_paxdb_dataset, organism_dir.glob('*.txt')))
    return experiment_list, list(organism_dir.glob('*.txt'))

def calc_pairwise_corr(experiment_list, args) -> pd.DataFrame:

    n_entries = len(experiment_list)
    paxdb_corr = np.zeros((n_entries, n_entries)) * np.nan

    for i in range(n_entries - 1):
        for j in range(i + 1, n_entries):
            match = pd.merge(experiment_list[i], experiment_list[j], on='external_id', suffixes=('_i', '_j'))
            if match.shape[0] > args.overlap:
                corr, pvalue = stats.pearsonr(match['abundance_i'], match['abundance_j'])
                if pvalue < 0.05:
                    paxdb_corr[i, j] = corr
    for i in range(n_entries - 1):
        for j in range(i + 1, n_entries):
            paxdb_corr[j,i] = paxdb_corr[i, j]
            
    df = pd.DataFrame(paxdb_corr>=args.threashold,
                      index=np.arange(paxdb_corr.shape[0]),    # 1st column as index
                      columns=np.arange(paxdb_corr.shape[1]))
    return df

def calc_max_clique(df) -> dict:
    net = nx.from_pandas_adjacency(df)
    l = nx.find_cliques(net)
    max_clique_df = {"size":0,'index':[]}
    for i in l:
        if len(i) > max_clique_df["size"]:
            max_clique_df["size"] = len(i)
            max_clique_df["index"] = i
    return max_clique_df

def read_sequences(file):
    seq_df = {}
    for rec in SeqIO.parse(file,"fasta"):
        seq_df[rec.id]=str(rec.seq)
    return seq_df

def calc_median_abundance(experiment_list, max_clique_df):
    df_subset =  pd.concat([experiment_list[i] for i in max_clique_df["index"]])
    return df_subset.groupby(by="external_id").median()

def write_fasta(seq_df, df_median, organism):
    failed_seq = {"id":[]}
    cnt = 0
    with open(os.path.join(args.output,f"{organism}_annot.fasta"), "w") as f:
        for row in df_median.reset_index().iterrows():
            id_ = f"{organism}.{row[1]['external_id']}"
            if id_ in seq_df:
                f.write(f">{row[1]['external_id']} {row[1]['abundance']}\n{seq_df[id_]}\n")
                cnt += 1
            else:
                failed_seq["id"].append(row[1]['external_id'])
    return cnt, failed_seq
                
def write_log(status, organism):
    with open(os.path.join(args.tmp,f"{organism}_log.txt"), "w") as f:
        f.write(f"Organism: {organism} finished with status: {status}")

def create_data_sets(DIR):
    try:
           # print(f"starting to compute {DIR}")
        organism = DIR.split('/')[-1]
        experiment_list, experiment_name_list = read_experiments(DIR)
        if len(experiment_list) < 3:
            status = f"Failed: number of experiments is {len(experiment_list)}"
            write_log(status, organism)
            return 1
    
        df_correlation = calc_pairwise_corr(experiment_list, args)
        max_clique_df = calc_max_clique(df_correlation)
    
        if max_clique_df['size'] < 3:
            status = f"Failed: number of correlated is {max_clique_df['size']}"
            write_log(status, organism)
            return 2
    
        sequence_file = f"{args.sequence_dir}/fasta.v11.5.{organism}.fa"
        seq_df = read_sequences(sequence_file)
        df_median = calc_median_abundance(experiment_list, max_clique_df)
        n_seq, _ = write_fasta(seq_df, df_median, organism)
        #print([experiment_list[i] for i in max_clique_df['index']])
        status = f"Success! Used: {max_clique_df['size']} ({[experiment_name_list[i] for i in max_clique_df['index']]})"
        write_log(status, organism)
    except:
        print(f"failed for organism: { DIR.split('/')[-1]}")
    return 0



def main(args):
    
    if not os.path.isdir(args.tmp):
        os.mkdir(args.tmp)
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    
    organism_dirs = glob.glob(f"{args.input}/*")

    with Pool(8) as p:
        p.map(create_data_sets, organism_dirs)
    log_files = glob.glob(f"{args.tmp}/*.txt")
    statuses = [] 
    for file in log_files:
        with open(file, "r") as f:
            statuses.append(f.read())
    with open(os.path.join(args.output, "log.txt"), "w") as f:
        for status in statuses:
            f.write(f"{status}\n")
    print(f"All done!")
    return 0

if __name__ == "__main__":
    
    args = parser.parse_args()
    main(args)
