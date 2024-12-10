import os
import pickle as pkl
from pathlib import Path
import glob
import tqdm
from multiprocessing import Pool
import argparse





import random
random.seed(0)

import pandas as pd
import numpy as np
np.random.seed(0)

from scipy.stats import pearsonr
from scipy import stats
from scipy.stats import spearmanr
from scipy.stats import boxcox
from scipy.special import inv_boxcox

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics  import mean_squared_error, r2_score, root_mean_squared_error, median_absolute_error


import torch
torch.manual_seed(0)
torch.use_deterministic_algorithms(False)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torcheval.metrics import R2Score



parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", type=str, help = "input directory")


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(1280, 512)  # 1280 from embedding dimension
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.do1 = nn.Dropout(0.3)
        

    def forward(self, input):

        f1 = self.do1(F.relu(self.fc1(input)))
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f2 = F.relu(self.fc2(f1))
        # Gaussian layer OUTPUT: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
        output = self.fc3(f2)
        return output

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def Train_and_Eval(model,optimizer,criterion, data_train, data_test, metric, n_epochs = 100,name="blabla.pt", test_data = None,testloader_experiment =None, args=None):

    train_stat = {"Epoch":[],
                "Training_loss":[],
                 "Training_r2":[],
                 "Test_loss":[],
                 "Test_r2":[]}

    best_model=1000

    for epoch in range(n_epochs):
    
        running_loss = 0.0
        model.train()
        metric.reset()
        for i, data in enumerate(data_train, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device="cuda")
            labels = labels.to(device="cuda")
            
            labels = labels.reshape((32,1))#.squeeze(0)
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
    
            metric.update(outputs, labels)
        train_r2 = metric.compute().item()
        train_loss = running_loss / i

        train_stat["Training_loss"].append(train_loss)
        train_stat["Training_r2"].append(train_r2)
        
        running_loss = 0
        model.eval()
        metric.reset()
        for i, data in enumerate(data_test, 0):
            # get the inputs; data is a list of [inputs, labels]
            with torch.no_grad():
                inputs, labels = data
                inputs = inputs.to(device="cuda")
                labels = labels.to(device="cuda")
                
                #labels = labels.squeeze(0)
                labels = labels.reshape((32,1))
        
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            # print statistics
                running_loss += loss.item()
                metric.update(outputs, labels)
        
        test_r2 = metric.compute().item()
        test_loss = running_loss / i
        train_stat["Test_loss"].append(test_loss)
        train_stat["Test_r2"].append(test_r2)
        train_stat["Epoch"].append(epoch)
        if test_loss < best_model:
            torch.save(model.state_dict(), f"{args.input}/best_model.pt")
            best_model = test_loss

    
        print(f'Epoch: {epoch:3d}, Training Loss: {train_loss:2.4f}, Train R2 {train_r2:2.4f}, Test Loss: {test_loss:2.4f}, Test R2 {test_r2:2.4f}')
    train_stat 
    checkpoint = torch.load(f"{args.input}/best_model.pt")
    model = Net()
    model.load_state_dict(checkpoint)
    model.eval()
    for i, data in enumerate(test_data, 0):
        inputs, labels = data
        train_stat["Test_Eval"] = model(inputs)
        train_stat["Test_True"] = labels

    train_stat["Test_experiment_Eval"] = []
    train_stat["Test_experiment_True"] = []
    for experiment in testloader_experiment:
        for i, data in enumerate(experiment, 0):
            inputs, labels = data
            train_stat["Test_experiment_Eval"].append(model(inputs))
            train_stat["Test_experiment_True"].append(labels)
    
    return train_stat

def train_model( data_df, args): 
    batch_size = 32
    print("Start training")
    
    train_set = data_df["train"]
    test_set = data_df["test"]
    
    g = torch.Generator()
    g.manual_seed(0)

    # Initiate Data loaders
    trainloader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=2,worker_init_fn=seed_worker,generator=g,drop_last=True)
    testloader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=True,num_workers=2,worker_init_fn=seed_worker,generator=g,drop_last=True)
    testloader_2 = torch.utils.data.DataLoader(test_set,batch_size=len(test_set),shuffle=False,num_workers=2,worker_init_fn=seed_worker,generator=g,drop_last=False)
    testloader_experiment = [torch.utils.data.DataLoader(test_set_experiment,batch_size=len(test_set_experiment),shuffle=False,num_workers=2,worker_init_fn=seed_worker,generator=g,drop_last=False) for test_set_experiment in data_df["experiment_test"]]
    # Initiate models
    model = Net()
    model.to(device="cuda")
    criterion = nn.MSELoss()
    metric = R2Score()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train and evaluate model
    train_stat = Train_and_Eval(model,optimizer,criterion, data_train=trainloader, data_test=testloader, metric=metric, n_epochs = 30, name = "Skippy", test_data=testloader_2, testloader_experiment = testloader_experiment, args=args)
    # 
    data_df["train_stat"] = train_stat 
    return data_df

def read_paxdb_dataset(path) -> pd.DataFrame:
    organism = args.input.split("/")[-1]
    dset = pd.read_csv(path, sep = '\t', header=11)
    dset = dset.rename(columns = {'#string_external_id': 'id'})
    dset['id'] = dset['id'].map(str)
    if organism != "ho2018":
        dset['id'] = dset['id'].map(lambda s: s.split('.')[1])
    return dset

def load_train_test_split(args):
    organism = args.input.split("/")[-1]
    training_partition_file = f"{args.input}/train.fa"
    train_ids = pd.read_csv(training_partition_file, sep=" ", names=["id", "count"])
    if organism != "ho2018":
        train_ids["id"] = train_ids.apply(lambda x: x["id"].split(".")[1], axis = 1)
    train_ids = train_ids["id"].to_list()
    
    test_partition_file = f"{args.input}/test.fa"
    test_ids = pd.read_csv(test_partition_file, sep=" ", names=["id", "count"])
    if organism != "ho2018":
        test_ids["id"] = test_ids.apply(lambda x: x["id"].split(".")[1], axis = 1)
    test_ids = test_ids["id"].to_list()

    return train_ids, test_ids

def load_abundance(args):
    experiment_files = glob.glob(f"{args.input}/experiments/*.txt")
    experiment_DataFrame_list = list(map(read_paxdb_dataset, experiment_files))
    print(f"Loaded: {len(experiment_DataFrame_list)} experiments")
    return experiment_DataFrame_list
    
def calc_median_abundance(experiment_DataFrame_list):
    df_tmp = pd.concat(experiment_DataFrame_list)
    df_tmp = df_tmp.groupby(by="id").median().reset_index()
    print(f"Median data frame contains {df_tmp.shape[0]} proteins")
    return df_tmp

def extract_experiment_test_abundance(embedding_df, experiment_DataFrame_list, test_ids, lam):
    experiment_test, experiment_test_id = [], []
    
    for experiment in experiment_DataFrame_list:
        
        experiment_tmp = experiment.loc[experiment["id"].isin(test_ids)]
        experiment_tmp = embedding_df.merge(experiment_tmp, on = "id", how="inner")
        if experiment_tmp.shape[0] < 100:
            print(f"Skipping experiment with {experiment_tmp.shape[0]} overlapping sequences")
            continue
        X_test = experiment_tmp["Embedding"].to_list()
        y_test = experiment_tmp["abundance"].to_list()
        y_ids  = experiment_tmp["id"].to_list()
        y_test_transform = boxcox(y_test, lam)
        experiment_test.append(list(zip(X_test,y_test_transform.astype(np.float32))))
        experiment_test_id.append(y_ids)

    print(f"Prepared: {len(experiment_test)} Experiment test sets")
    return experiment_test, experiment_test_id

#def check_no_dot_id(df):
#    ids = df["id"].to_list()
#    ids_new = []
#    for id_ in ids:
#        if "." in id_:
            

def make_df(args):
    data_df = {}
    embedding_file = f"{args.input}/embedding.pkl"
    organism = args.input.split("/")[-1]
    assert embedding_file[-3:] == "pkl", "Embedding is Not pickle file"
            
    with open(embedding_file, "br") as f:
        embedding_df = pkl.load(f)
        embedding_df["Embedding"] = embedding_df["Embedding"][:len(embedding_df["id"])] # Make sure to remove last two copys of embeddings  ( bug from embedings calc script)

    embedding_df = pd.DataFrame(embedding_df)
    if organism != "ho2018":
        try:
            embedding_df["id"] = embedding_df.apply(lambda x: x["id"].split(".")[1], axis = 1)
        except:

            print("Not using species in id")
    print("Embedding df")
    print(embedding_df)
    
    abundance_df_list = load_abundance(args)
    abundance_median_df = calc_median_abundance(abundance_df_list)
    

    print("Abundance df")
    print(abundance_median_df)
    tmp_df = embedding_df.merge(abundance_median_df, on = "id", how="inner")

    print("Merge df")
    print(tmp_df)
    
    train_ids, test_ids = load_train_test_split(args)

    #print(f"train ids: {train_ids}")

    train_df = tmp_df.loc[tmp_df["id"].isin(train_ids)]
    

    #print("Train df")
    #print(train_df)

    X_train = train_df["Embedding"].to_list()
    y_train = train_df["abundance"].to_list()

    test_df = tmp_df.loc[tmp_df["id"].isin(test_ids)]
    X_test = test_df["Embedding"].to_list()
    y_test = test_df["abundance"].to_list()

    print(f"Y train : {y_train}")
    y_train_transform, lam = boxcox(y_train)
    y_test_transform       = boxcox(y_test, lam)

    print(f"Training set has {len(X_train)} samples and Test set has {len(X_test)} samples")
    
    data_df["lambda"] = lam
    data_df["train"] = list(zip(X_train,y_train_transform.astype(np.float32)))
    data_df["test"] = list(zip(X_test,y_test_transform.astype(np.float32)))
    data_df["test_labels"] = y_test
    data_df["train_labels"] = y_train
    experiment_test, experiment_test_id = extract_experiment_test_abundance(embedding_df, abundance_df_list, test_ids, lam)
    data_df["experiment_test"] = experiment_test
    data_df["experiment_test_ids"] = experiment_test_id
    return data_df

def write_df(data_df, args):
    cnt=0
    while (os.path.isfile(f"{args.input}/training_df_{cnt}.pkl")):
        cnt += 1

    with open(f"{args.input}/training_df_{cnt}.pkl", "bw") as f:
        pkl.dump(data_df, f)

    return 0 


def main(args):

    # Load data
    print("Load data")
    data_df = make_df(args)
 
    # Train and evaluate model
    print("Training")
    data_df = train_model( data_df, args)

    # Write data frame
    print("Writing data")
    write_df(data_df, args)

    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)