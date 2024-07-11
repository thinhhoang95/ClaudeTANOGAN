# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 10:16:01 2020
Modified on Mon Jul 8 by Thinh Hoang (dthoang@artimensium.com)

@author: MA Bashar, Thinh Hoang (dthoang@artimensium.com)
"""


import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
from sklearn import preprocessing
from tqdm import tqdm

class NGSIMDataset(Dataset):
    def __init__(self, data_settings):
        """
        Args:
            data_settings (object): settings for loading data and preprocessing
        """
        self.train = data_settings.train
        # self.ano_span_count = 0 # self.ano_span_count is updated when read_data() function is is called
        
        df_x = self.read_data(data_file = data_settings.data_file, 
                       # label_file = data_settings.label_file, # no label_file for NGSIM dataset since we are training unsupervised
                       # key = data_settings.key, 
                       BASE = data_settings.BASE)
        
        # Extract the column of interest
        if data_settings.column_name is not None:
            df_x = df_x[['Vehicle_ID', data_settings.column_name]]
            print(f"Select column name: {data_settings.column_name}")
  
        # normalize the data
        df_x = self.normalize(df_x)
        
        # important parameters
        #self.window_length = int(len(df_x)*0.1/self.ano_span_count)
        self.window_length = 32
        self.stride = 1
        # if data_settings.train:
        #     self.stride = 
        # else:
        #     self.stride = self.window_length
        
        self.n_feature = len(df_x.columns) - 1 # exclude Vehicle_ID
        # self.n_feature = 3
        
        # x, y data
        x = df_x
        
        # adapt the datasets for the sequence data shape
        x = self.unroll(x)
        x = x.reshape((-1, 1, self.window_length))
        
        self.x = torch.from_numpy(x).float()
        
        self.data_len = x.shape[0]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.x[idx] # , self.y[idx]

    # Deprecated
    def unroll_legacy(self, data, labels):
        un_data = []
        un_labels = []
        seq_len = int(self.window_length)
        stride = int(self.stride)
        
        idx = 0
        while(idx < len(data) - seq_len):
            un_data.append(data.iloc[idx:idx+seq_len].values)
            un_labels.append(labels.iloc[idx:idx+seq_len].values)
            idx += stride
        return np.array(un_data), np.array(un_labels)
    
    # For NGSIM dataset with no label file
    def unroll(self, data):
        # data columns: ['Vehicle_ID', 'Local_Y', 'Local_X', 'Global_Time']
        # data will be pandas dataframe, we must unroll by Vehicle_ID
        un_data = []
        seq_len = int(self.window_length)
        stride = int(self.stride)

        veh_ids = data['Vehicle_ID'].unique()
        
        veh_counter = 0
        while veh_counter < len(veh_ids):
            idx = 0
            # tqdm.write(f"Processed vehicle {veh_counter+1}/{len(veh_ids)}")
            veh_df = data[data['Vehicle_ID'] == veh_ids[veh_counter]]
            # subtract Velocity from the first Velocity value
            while idx < len(veh_df) - seq_len: # seq_len: length of the sliding window
                array_to_append = veh_df.iloc[idx:idx+seq_len].values[:,1:] # exclude Vehicle_ID
                un_data.append(array_to_append)
                idx += stride
                
                # print(f"IDX: {idx}")
                
            veh_counter += 1
            # print(f"Number of sequences processed in this iteration: {len(un_data)}")
        return np.array(un_data)
    
    # For NGSIM dataset with no label file
    def read_data(self, data_file=None, BASE=''):
        df_x = pd.read_csv(BASE+data_file)
        return df_x
    
    # Deprecated
    def read_data_legacy(self, data_file=None, label_file=None, key=None, BASE=''):
        with open(BASE+label_file) as FI:
            j_label = json.load(FI)
        ano_spans = j_label[key]
        self.ano_span_count = len(ano_spans)
        df_x = pd.read_csv(BASE+data_file)
        df_x, df_y = self.assign_ano(ano_spans, df_x)
            
        return df_x, df_y
    
    def assign_ano_legacy(self, ano_spans=None, df_x=None):
        df_x['timestamp'] = pd.to_datetime(df_x['timestamp'])
        y = np.zeros(len(df_x))
        for ano_span in ano_spans:
            ano_start = pd.to_datetime(ano_span[0])
            ano_end = pd.to_datetime(ano_span[1])
            for idx in df_x.index:
                if df_x.loc[idx, 'timestamp'] >= ano_start and df_x.loc[idx, 'timestamp'] <= ano_end:
                    y[idx] = 1.0
        return df_x, pd.DataFrame(y)
    
    def normalize(self, df_x=None):
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        if 'Local_X' in df_x.columns:
            df_x_scaled = min_max_scaler.fit_transform(df_x['Local_X'].values.reshape(-1, 1))
            df_x['Local_X'] = pd.DataFrame(df_x_scaled)
        if 'Local_Y' in df_x.columns:
            df_y_scaled = min_max_scaler.fit_transform(df_x['Local_Y'].values.reshape(-1, 1))
            df_x['Local_Y'] = pd.DataFrame(df_y_scaled)
        if 'Velocity' in df_x.columns:
            df_v_scaled = min_max_scaler.fit_transform(df_x['Velocity'].values.reshape(-1, 1))
            df_x['Velocity'] = pd.DataFrame(df_v_scaled)
        return df_x
    
    def get_train_data(self, config):
        return torch.utils.data.DataLoader(self, batch_size=config['batch_size'], shuffle=True)
    
    
# settings for data loader
class DataSettings:
    
    def __init__(self):
        # location of datasets and category
        end_name = 'ngsim_sample.csv' # dataset name
        data_file = end_name # dataset category and dataset name
        # key = 'realKnownCause/'+end_name # This key is used for reading anomaly labels
        
        self.BASE = '/Users/thinhhoang/Documents/ClaudeTANOGAN/NGSIM_Dataset'
        # check if self.BASE has the last '/'
        if self.BASE[-1] != '/':
            self.BASE += '/'
        # self.label_file = 'labels\\combined_windows.json'
        self.data_file = data_file
        # self.key = key
        self.train = True
        self.window_length = 60
        self.column_name = 'Local_Y'
    
    
def main():
    data_settings = DataSettings()
    # define dataset object and data loader object for NAB dataset
    dataset = NGSIMDataset(data_settings=data_settings)
    print(f'Dataset X shape: {dataset.x.shape}') # check the dataset shape
    
if __name__=='__main__':
    main()