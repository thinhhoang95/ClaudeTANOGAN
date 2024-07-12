# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 10:16:01 2020
Modified on Mon Jul 11 by Thinh Hoang (dthoang@artimensium.com)

@author: Thinh Hoang (dthoang@artimensium.com)
"""


import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
from sklearn import preprocessing
from tqdm import tqdm

def unroll_ts(data, window_length, stride):
    """This will unroll the NGSIM time series into small sequences of length window_length with stride stride.
    From these, we can use the VAE network to generate the latent space embeddings for each sequence.
    """
    # data columns: Global_Time,Local_X,Local_Y,Velocity
    # data will be pandas dataframe, we must unroll by Vehicle_ID
    un_data = [] # unrolled data
    bond_data = [] # to store the links between different data points in un_data, providing a way to trace back to the original data
    seq_len = int(window_length)
    stride = int(stride)

    veh_ids = data['Vehicle_ID'].unique()
    
    veh_counter = 0
    last_row_id = -1
    row_id = 0
    while veh_counter < len(veh_ids):
        idx = 0 # index for the current vehicle
        # tqdm.write(f"Processed vehicle {veh_counter+1}/{len(veh_ids)}")
        veh_df = data[data['Vehicle_ID'] == veh_ids[veh_counter]]
        # subtract Velocity from the first Velocity value
        while idx < len(veh_df) - seq_len: # seq_len: length of the sliding window
            array_to_append = veh_df.iloc[idx:idx+seq_len].values[:,1:] # exclude Vehicle_ID
            un_data.append(array_to_append)
            idx += stride
            bond_data.append((row_id, last_row_id)) # store the link between the current row_id and the last_row_id
            last_row_id = row_id
            row_id += 1 # every append operation, we increment the row_id
            
            # print(f"IDX: {idx}")
            
        veh_counter += 1
        last_row_id = -1 # reset the last_row_id for the next vehicle
        # print(f"Number of sequences processed in this iteration: {len(un_data)}")
    return np.array(un_data), bond_data

import pickle

def create_lstm_data():
    data = pd.read_csv('NGSIM_Dataset/ngsim_sample.csv')
    unrolled_data, bond_data = unroll_ts(data, 32, 1)
    print(f"Unrolled data shape: {unrolled_data.shape}")
    print(f"Bond data length: {len(bond_data)}")

    # Write both unrolled_data and bond_data to files
    # Write unrolled_data to npy file
    np.save('NGSIM_Dataset/unrolled_data.npy', unrolled_data)
    # Write bond_data to a pickle file
    with open('NGSIM_Dataset/bond_data.pkl', 'wb') as f:
        pickle.dump(bond_data, f)
        
    print("Unrolled data and bond data have been written to files!")
    print("Files: unrolled_data.npy, bond_data.pkl")

if __name__ == '__main__':
    # create_lstm_data() # comment this out if embeddings have already been generated
    pass

from torch.utils.data import Dataset

class NGSIM_PreEmbeddings_Dataset(Dataset):
    def __init__(self, unroll_filepath, bond_filepath, column = None, batch_size=32):
        # Load up the unrolled data and bond data
        self.unrolled_data = np.load(unroll_filepath)
        if column is not None:
            self.unrolled_data = self.unrolled_data[:, :, column]
        
        # Reshape the unrolled data
        self.unrolled_data = np.expand_dims(self.unrolled_data, axis=1) # make (-1)x32 into (-1)x1x32
        
        with open(bond_filepath, 'rb') as f:
            self.bond_data = pickle.load(f)
            
        self.index = 0 # for iterating through the dataset
        self.batch_size = batch_size
            
    
    def __len__(self):
        return len(self.unrolled_data)
    
    def __getitem__(self, index):
        return self.unrolled_data[index], self.bond_data[index]
    
    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        if self.index < len(self.unrolled_data):
            batch = self.unrolled_data[self.index:self.index+self.batch_size]
            bond_batch = self.bond_data[self.index:self.index+self.batch_size]
            self.index += self.batch_size
            return batch, bond_batch
        raise StopIteration
    
def create_embeddings_sequences(embeddings_filepath, bond_filepath, batch_size=32, lookback_steps=1):
    # Load embeddings and bonds from files
    embeddings = np.load(embeddings_filepath)
    with open(bond_filepath, 'rb') as f:
        bond_data = pickle.load(f)
    
    # We will form a training set of sequence embeddings
    total_samples = len(embeddings)
    print(f"Total samples: {total_samples}")
    valid_samples = total_samples - lookback_steps
    
    # Pre-allocate arrays
    input_seqs = np.zeros((valid_samples, lookback_steps, embeddings.shape[1]), dtype=embeddings.dtype)
    output_seqs = np.zeros((valid_samples, lookback_steps, embeddings.shape[1]), dtype=embeddings.dtype)
    
    valid_idx = 0
    for i in range(lookback_steps, total_samples):
        current_embedding_id, previous_embedding_id = bond_data[i]

        if previous_embedding_id != -1:
            input_seqs[valid_idx] = embeddings[i-lookback_steps:i]
            output_seqs[valid_idx] = embeddings[i-lookback_steps+1:i+1]
            valid_idx += 1
    
    # Trim arrays to remove unused space
    return input_seqs[:valid_idx], output_seqs[:valid_idx]

if __name__ == '__main__':
    # input_seqs, output_seqs = create_embeddings_sequences('NGSIM_Dataset/embeddings.npy', 'NGSIM_Dataset/bond_data.pkl', batch_size=32, lookback_steps=1)
    # np.save('NGSIM_Dataset/input_seqs.npy', input_seqs)
    # np.save('NGSIM_Dataset/output_seqs.npy', output_seqs)
    # print("Input and output sequences have been saved to files!")
    pass
    
class NGSIM_Embeddings_Dataset(Dataset):
    def __init__(self, embeddings_filepath, bond_filepath, batch_size=32, lookback_steps=1):
        self.input_seqs, self.output_seqs = create_embeddings_sequences(embeddings_filepath, bond_filepath, batch_size, lookback_steps)
        
    def __len__(self):
        return len(self.input_seqs)
    
    def __getitem__(self, index):
        return self.input_seqs[index], self.output_seqs[index]
        
    def get_training_data(self):
        return torch.utils.data.DataLoader(self, batch_size=32, shuffle=True)