# config.py

# Preprocessing options
preprocessing_options = {
    'basics': False,                # mean, std, rms, kurtosis, skew
    'noise_reduction': False,
    'fft': False,
    'mfcc': False,
    'stft': True
}

# Noise reduction parameters
noise_reduction_params = {
    'n_fft': 1024,
    'hop_length': 512
}

# FFT parameters
fft_params = {
    'n_fft': 1024,
    'fmin': 500,
    'fmax': 85000
}

# MFCC parameters
mfcc_params = {
    'n_fft': 1024,
    'hop_length': 512,
    'n_mels': 50,
    'fmin': 500,
    'fmax': 85000,
    'n_mfcc': 10
}

# STFT parameters
stft_params = {
    'n_fft': 1024,
    'hop_length': 512,
    'win_length': 1024
}

# Model
model = {
    'GBDT': True,
    'RF': False
}

# GBDT model parameters
model_params_GBDT = {
    'task': 'CLASSIFICATION',
    'num_trees': 300,                              # Default: 300                              
    'growing_strategy': 'LOCAL',                   # Default: 'LOCAL'
    'max_depth': 6,                                # Default: 6
    'early_stopping': 'LOSS_INCREASE'              # Default: 'LOSS_INCREASE'
}

# RF model parameters
model_params_RF = {
    'task': 'CLASSIFICATION',
    'num_trees': 300,                              # Default: 300                              
    'growing_strategy': 'LOCAL',                   # Default: 'LOCAL'
    'max_depth': 16,                               # Default: 16
}

# Force 50%-50% dataset
force_balanced_dataset = True
