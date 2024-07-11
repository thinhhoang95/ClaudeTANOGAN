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
    # data columns: ['Vehicle_ID', 'Local_Y', 'Local_X', 'Global_Time']
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


create_lstm_data()
