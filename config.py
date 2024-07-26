# config.py

# Preprocessing options
preprocessing_options = {
    'basics': False,          # mean, std, rms, kurtosis, skew
    'noise_reduction': False,
    'fft': True,
    'mfcc': False,
    'stft': False
}

# Noise reduction parameters
noise_reduction_params = {
    'n_fft': 2048,        # Number of FFT points
    'hop_length': 1024,   # Hop size between successive frames
    'prop_decrease': 0.5  # Property for noise reduction
}

# FFT parameters
fft_params = {
    'n_fft': 2048,  # Number of FFT points
    'fmin': 500,    # Minimum frequency (Hz)
    'fmax': 85000   # Maximum frequency (Hz)
}

# MFCC parameters
mfcc_params = {
    'n_fft': 2048,       # Number of FFT points
    'hop_length': 1024,  # Hop size between successive frames
    'n_mels': 128,        # Number of Mel filter banks
    'fmin': 500,         # Minimum frequency (Hz)
    'fmax': 85000,       # Maximum frequency (Hz)
    'n_mfcc': 20         # Number of MFCC coefficients
}

# STFT parameters
stft_params = {
    'n_fft': 2048,       # Number of FFT points
    'hop_length': 1024,  # Hop size between successive frames
}

# Force 50%-50% dataset
force_balanced_dataset = True


# Model
model = {
    'GBDT': True,
    'RF': False
}

# GBDT model parameters
model_params_GBDT = {
    'num_trees': 300,                              # Default: 300                              
    'growing_strategy': 'LOCAL',                   # Default: 'LOCAL'
    'max_depth': 6,                                # Default: 6
    'early_stopping': 'LOSS_INCREASE'              # Default: 'LOSS_INCREASE'
}

# RF model parameters
model_params_RF = {
    'num_trees': 300,                              # Default: 300                              
    'growing_strategy': 'LOCAL',                   # Default: 'LOCAL'
    'max_depth': 16,                               # Default: 16
}

