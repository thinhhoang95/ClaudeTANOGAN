import torch
from torch.utils.data import DataLoader, TensorDataset

class DataGenerator:
    def __init__(self, train_configurations):
        config = train_configurations
        self.config = config
        # Here you would load and preprocess your data
        # VAE
        self.train_data = torch.randn(1000, config['n_channel'], 32)
        self.val_data = torch.randn(200, config['n_channel'], 32)

    def get_train_data(self):
        return DataLoader(TensorDataset(self.train_data), batch_size=self.config['batch_size'], shuffle=True)

    def get_val_data(self):
        return DataLoader(TensorDataset(self.val_data), batch_size=self.config['batch_size'], shuffle=False)