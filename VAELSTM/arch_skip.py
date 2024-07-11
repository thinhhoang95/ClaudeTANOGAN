import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.input_size = config['window_dim']
        self.hidden_dim = config['hidden_dim']
        self.latent_dim = config['latent_dim']

        # Encoder
        n_conv_layers = int(np.log2(self.hidden_dim) - 5)  # because we start with 32 filters
        self.conv1 = nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv_layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        for i in range(n_conv_layers):
            in_channels = 2 ** (i + 5)
            out_channels = 2 ** (i + 6)
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            self.skip_connections.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2))

        self.fc_mu = nn.Linear(self.hidden_dim * 1, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim * 1, self.latent_dim)

        # Decoder
        self.fc_decoder = nn.Linear(self.latent_dim, self.hidden_dim)
        self.deconv1 = nn.ConvTranspose1d(2 ** (n_conv_layers + 5), 2 ** (n_conv_layers + 4), kernel_size=4, stride=2, padding=1)
        self.deconv_layers = nn.ModuleList()
        self.skip_connections_decoder = nn.ModuleList()

        for i in range(n_conv_layers-1):
            in_channels = 2 ** (n_conv_layers - i + 4)
            out_channels = 2 ** (n_conv_layers - i + 3)
            self.deconv_layers.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            self.skip_connections_decoder.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1, stride=2, output_padding=1))

        self.deconv3 = nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1)

    def encode(self, x):
        h = torch.relu(self.conv1(x))
        for i, conv in enumerate(self.conv_layers):
            h_conv = conv(h)
            h_skip = self.skip_connections[i](h)
            h = torch.relu(h_conv + h_skip)
        h = h.view(-1, self.hidden_dim * 1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = torch.relu(self.fc_decoder(z)).view(-1, self.hidden_dim, 1)
        h = torch.relu(self.deconv1(h))
        for i, deconv in enumerate(self.deconv_layers):
            h_deconv = deconv(h)
            h_skip = self.skip_connections_decoder[i](h)
            h = torch.relu(h_deconv + h_skip)
        return self.deconv3(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


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
        # self.lstm = LSTM(config).to(self.device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=config['learning_rate_vae'])
        self.vae_scheduler = torch.optim.lr_scheduler.StepLR(self.vae_optimizer, step_size=5, gamma=0.1)
        # self.lstm_optimizer = torch.optim.Adam(self.lstm.parameters(), lr=config['learning_rate_lstm'])

        # Tboard
        self.writer = SummaryWriter(log_dir=self.config['tensorboard_log_dir'])

    def vae_loss(self, recon_x, x, mu, log_var):
        print('Shape of x:', x.shape)
        print('Shape of recon_x:', recon_x.shape)
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD

    def train_vae(self, data):
        self.vae.train()
        for epoch in range(self.config['n_epochs_vae']):
            for batch_idx, batch in enumerate(data):
                x = batch.to(self.device)
                # print(f'Batch shape: {x.shape}')
                self.vae_optimizer.zero_grad()
                recon_batch, mu, log_var = self.vae(x)
                loss = self.vae_loss(recon_batch, x, mu, log_var)
                loss.backward()
                self.vae_optimizer.step()
                # Log batch loss
                self.writer.add_scalar('Loss/batch', loss.item(), epoch * len(data) + batch_idx)
            print(f'VAE Epoch: {epoch}, Loss: {loss.item()}')
            self.vae_scheduler.step()
            # Log epoch loss
            #avg_epoch_loss = epoch_loss / len(data)
            #self.writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch)

            # Log model parameters
            for name, param in self.vae.named_parameters():
                self.writer.add_histogram(f'Parameters/{name}', param, epoch)

        self.writer.close()
            

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