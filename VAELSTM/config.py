train_configurations = {
    "window_dim": 32, # CAUTION: changing this value requires changing the model architecture
    "hidden_dim": 512,
    "latent_dim": 6,
    "learning_rate_vae": 1e-3,
    "batch_size": 32,
    "n_channel": 1,
    "n_epochs_vae": 100,
    "tensorboard_log_dir": "tbo",
    "num_hidden_units_lstm": 256,
    "code_size": 6, # equal to latent_dim
    "num_epochs_lstm": 100,
    "learning_rate_lstm": 1e-3
}