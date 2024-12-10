import transformers

import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
import numpy as np

import pickle

from transformers import pipeline
from transformers import BertForMaskedLM, BertTokenizer, pipeline, AutoTokenizer, AutoModelForMaskedLM
import esm


import torch
import os


from Bio import SeqIO

import argparse

parser = argparse.ArgumentParser( 
    """ This script will read a fasta file """)

parser.add_argument("-i", "--input", type = str, required = True, help = 
                    "Path to fasta file to be converted")
parser.add_argument("-o", "--output", type = str, default = "./embedings.pkl", help = 
                   "Path to output ")
parser.add_argument("-m", "--model", type = str, default = "ESM", help = 
                   "What model to generate embeddings ")

NON_STANDARD_AMINO = ["B", "U", "Z", "X", "*"]


def read_fasta(file):
    
    df_fasta = { "id":[], 'seq':[], "TM":[]}
    list_data = []
    for rec in SeqIO.parse(file, "fasta"):
        sequence = str(rec.seq)
        
        if any(amino in sequence for amino in NON_STANDARD_AMINO):
            print(f"Sequences can not contain non standard amino acids Removing: {rec.id}")
            continue

            #raise ValueError (f"Sequences can not contain non standard amino acids")
        if len(rec.seq) > 2048:
            print(f"Sequences can not contain more than 1024 amino acids Removing: {rec.id} with {len(rec.seq)} aas")
            continue
        df_fasta["id"].append(rec.id)
        df_fasta["seq"].append(str(rec.seq))
        df_fasta["TM"].append( 1.0 )#float(rec.description.split(" ")[-1]))
        list_data.append((rec.id, str(rec.seq)))
    return df_fasta, list_data



def creat_embedings(data, df_fasta):

    ## Set transformer parameters 
    #transformers.logging.set_verbosity_error()
    if type(data) == dict:

        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
        model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert").to(device="cuda:0")
    
        #tokenizer = BertTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        #model = BertForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device="cuda:0")
    
        pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer, framework="pt",device="cuda:0")
    
        embeddings_list = []
        # Exctract CLS embeddings from ESM
        for idx, prot in enumerate(df_fasta["seq"]):
            this_embedding = pipe(' '.join(prot))
            #this_embedding = pipe(prot)
    
            avg_token = np.mean(np.array(this_embedding[0]), axis=0)
            embeddings_list.append(avg_token)
    
            if idx % 100 ==0:
                print(f"Done with {idx} Embeddings")
        assert len(embeddings_list) == len(df_fasta["seq"])
        df_fasta["Embedding"] = embeddings_list
        return df_fasta
    else:
        # Load ESM-2 model
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model.cuda()
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results
        embs = []
        batch_size = 2
        n_iter = len(data)//batch_size
        for i in range(n_iter):
            #print(f"Starting batch {i}")
            
            batch = data[i*batch_size:(i+1)*batch_size] if i < n_iter - 2 else data[i*batch_size:]
            
            batch_labels, batch_strs, batch_tokens = batch_converter(batch)
            
            # Move batch_tokens to the GPU
            #batch_tokens = batch_tokens
            
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    
    
    
            with torch.no_grad():
                results = model(batch_tokens.cuda(), repr_layers=[33], return_contacts=False)

            token_representations = results["representations"][33].to(device="cpu")
    
            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            sequence_representations = []
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
            #print(sequence_representations)
            #print(sequence_representations[0])
            embs += sequence_representations
            
        df_fasta["Embedding"] = embs
        return df_fasta


        
def write_pickel(df_fasta, output):
    with open(output, 'wb') as f:
        pickle.dump(df_fasta, f)
    

def main(args):
    
    df_fasta, list_data = read_fasta(args.input)
    #df_fasta = creat_embedings(df_fasta)
    if args.model == "ESM":
        df_fasta = creat_embedings(list_data, df_fasta)
    else:
        df_fasta = creat_embedings(df_fasta, df_fasta)
    write_pickel(df_fasta, args.output)
    print("Done!")
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)






#    else:
#        print("Using new emb")
#        model, alphabet = torch.hub.load("facebookresearch/esm", "esm1_t34_670M_UR50S")
#        batch_converter = alphabet.get_batch_converter()
#        batch_labels, batch_strs, batch_tokens = batch_converter(list_data)
#        with torch.no_grad():
#            results = model(batch_tokens.to(device="cpu"), repr_layers=[33], return_contacts=False)
#        result = np.mean(results['representations'][33].numpy(), axis = 1)
#        result = [list(emb) for emb in result]
#        df_fasta["Embedding"] = result
#        return df_fasta











