import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.config = config
        self.input_dims = config['l_win'] * config['n_channel']
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.fc_mu = nn.Linear(config['num_hidden_units'] * 3, config['code_size'])
        self.fc_log_var = nn.Linear(config['num_hidden_units'] * 3, config['code_size'])

    def _build_encoder(self):
        layers = []
        in_channels = self.config['n_channel']
        out_channels = self.config['num_hidden_units'] // 16

        for _ in range(3):  # Simplified for brevity, adjust as needed
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.LeakyReLU())
            in_channels = out_channels
            out_channels *= 2

        layers.append(nn.Conv1d(in_channels, self.config['num_hidden_units'], kernel_size=4, stride=1))
        layers.append(nn.LeakyReLU())
        
        return nn.Sequential(*layers)

    def _build_decoder(self):
        layers = []
        in_channels = self.config['num_hidden_units']
        
        for i in range(3):  # Simplified for brevity, adjust as needed
            out_channels = in_channels // 2
            layers.append(nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ))
            layers.append(nn.LeakyReLU())
            in_channels = out_channels

        # Final layer to match the original number of channels
        layers.append(nn.ConvTranspose1d(
            in_channels,
            self.config['n_channel'],
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        ))
        
        return nn.Sequential(*layers)

    def encode(self, x):
        print(f'Input shape: {x.shape}')
        #x = x.unsqueeze(1)  # Add channel dimension
        # print(f'Input shape after unsqueeze: {x.shape}')
        h = self.encoder(x)
        print(f'Output shape after encoder: {h.shape}')
        h = h.view(h.size(0), -1)
        print(f'Output shape after view: {h.shape}')
        mux, lvarx = self.fc_mu(h), self.fc_log_var(h)
        print(f'Mu shape: {mux.shape}, Log Var shape: {lvarx.shape}')
        return mux, lvarx

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        print('---')
        print(f'Input shape: {z.shape}')
        z = z.view(z.size(0), -1, 1, 1)
        print(f'Input shape after view: {z.shape}')
        return self.decoder(z)

    def forward(self, x):
        print(f'Forward First Line Input shape: {x.shape}')
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(config['code_size'], config['num_hidden_units_lstm'], batch_first=True)
        self.lstm2 = nn.LSTM(config['num_hidden_units_lstm'], config['num_hidden_units_lstm'], batch_first=True)
        self.lstm3 = nn.LSTM(config['num_hidden_units_lstm'], config['code_size'], batch_first=True)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        return x

class VAE_LSTM_Model:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = VAE(config).to(self.device)
        self.lstm = LSTM(config).to(self.device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=config['learning_rate_vae'])
        self.lstm_optimizer = torch.optim.Adam(self.lstm.parameters(), lr=config['learning_rate_lstm'])

    def vae_loss(self, recon_x, x, mu, log_var):
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD

    def train_vae(self, data):
        self.vae.train()
        for epoch in range(self.config['num_epochs_vae']):
            for batch in data:
                x = batch.to(self.device)
                print(f'Batch shape: {x.shape}')
                self.vae_optimizer.zero_grad()
                recon_batch, mu, log_var = self.vae(x)
                loss = self.vae_loss(recon_batch, x, mu, log_var)
                loss.backward()
                self.vae_optimizer.step()
            print(f'VAE Epoch: {epoch}, Loss: {loss.item()}')

    def train_lstm(self, data):
        self.lstm.train()
        criterion = nn.MSELoss()
        for epoch in range(self.config['num_epochs_lstm']):
            for batch in data:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                self.lstm_optimizer.zero_grad()
                output = self.lstm(x)
                loss = criterion(output, y)
                loss.backward()
                self.lstm_optimizer.step()
            print(f'LSTM Epoch: {epoch}, Loss: {loss.item()}')

    def generate_embeddings(self, data):
        self.vae.eval()
        embeddings = []
        with torch.no_grad():
            for batch in data:
                x = batch.to(self.device)
                mu, _ = self.vae.encode(x)
                embeddings.append(mu.cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def predict_sequence(self, initial_sequence):
        self.vae.eval()
        self.lstm.eval()
        with torch.no_grad():
            initial_embedding = self.vae.encode(initial_sequence.to(self.device))[0]
            predicted_sequence = self.lstm(initial_embedding.unsqueeze(0))
            reconstructed_sequence = self.vae.decode(predicted_sequence.squeeze(0))
        return reconstructed_sequence.cpu().numpy()

# Example usage:
# config = {...}  # Define your configuration
# model = VAE_LSTM_Model(config)
# train_data = ...  # Prepare your data
# model.train_vae(train_data)
# lstm_data = ...  # Prepare LSTM training data
# model.train_lstm(lstm_data)
# test_sequence = ...  # Prepare a test sequence
# predicted_sequence = model.predict_sequence(test_sequence)