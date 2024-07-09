import torch
import torch.nn as nn

class LSTMGenerator(nn.Module):
    """An LSTM based generator. It expects a sequence of noise vectors as input.

    Args:
        in_dim: Input noise dimensionality
        out_dim: Output dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms

    Input: noise of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, out_dim)
    """

    def __init__(self, in_dim, out_dim, device=None):
        super().__init__()
        self.out_dim = out_dim
        self.device = device

        self.lstm0 = nn.LSTM(in_dim, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        
        self.linear = nn.Sequential(nn.Linear(in_features=128, out_features=out_dim), nn.Tanh())

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(1, batch_size, 32).to(self.device)
        c_0 = torch.zeros(1, batch_size, 32).to(self.device)

        recurrent_features, (h_1, c_1) = self.lstm0(input, (h_0, c_0))
        recurrent_features, (h_2, c_2) = self.lstm1(recurrent_features)
        recurrent_features, _ = self.lstm2(recurrent_features)
        
        outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, 128))
        outputs = outputs.view(batch_size, seq_len, self.out_dim)
        return outputs, recurrent_features
    
class LSTMGeneratorX(nn.Module):
    def __init__(self, noise_dim, hidden_dim, output_dim, n_layers, dropout_rate=0.2):
        super(LSTMGeneratorX, self).__init__()
        
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=noise_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_rate if n_layers > 1 else 0
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Tanh to bound the output between -1 and 1
        )

    def forward(self, noise):
        # noise shape: (batch_size, seq_len, noise_dim)
        batch_size, seq_len, _ = noise.shape
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(noise.device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(noise.device)
        
        # Pass through LSTM
        lstm_out, (hn, cn) = self.lstm(noise, (h0, c0))
        
        # Pass through output layer
        output = self.output_layer(lstm_out)
        
        return output, lstm_out
    
class LSTMGenerator2(nn.Module):
    """
    An LSTM based generator. It expects a sequence of noise vectors as input.

    Args:
        in_dim: Input noise dimensionality
        out_dim: Output dimensionality
        hidden_dim: Dimensionality of the hidden layer of LSTMs
        n_layers: Number of LSTM layers
    
    Input: noise of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, out_dim)
    """
    def __init__(self, in_dim, out_dim, hidden_dim=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, out_dim)
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, noise):
        batch_size, seq_len, _ = noise.size()
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(noise.device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(noise.device)
        
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(noise, (h0.detach(), c0.detach()))
        
        # Apply linear layer to each time step
        out = self.linear(out)
        return out


class LSTMDiscriminator(nn.Module):
    """An LSTM based discriminator. It expects a sequence as input and outputs a probability for each element. 

    Args:
        in_dim: Input noise dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms

    Inputs: sequence of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, 1)
    """

    def __init__(self, in_dim, device=None):
        super().__init__()
        self.device = device

        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=100, num_layers=1, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(100, 1), nn.Sigmoid())

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(1, batch_size, 100).to(self.device)
        c_0 = torch.zeros(1, batch_size, 100).to(self.device)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, 100))
        outputs = outputs.view(batch_size, seq_len, 1)
        return outputs, recurrent_features


if __name__ == "__main__":
    batch_size = 16
    seq_len = 32
    noise_dim = 100
    seq_dim = 4

    gen = LSTMGenerator(noise_dim, seq_dim)
    dis = LSTMDiscriminator(seq_dim)
    noise = torch.randn(8, 16, noise_dim)
    gen_out = gen(noise)
    dis_out = dis(gen_out)
    
    print("Noise: ", noise.size())
    print("Generator output: ", gen_out.size())
    print("Discriminator output: ", dis_out.size())
