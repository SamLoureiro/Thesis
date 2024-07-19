# config.py

# Preprocessing options
preprocessing_options = {
    'basics': True,                # mean, std, rms, kurtosis, skew
    'noise_reduction': False,
    'fft': True,
    'mfcc': True,
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
    'fmax': 20000,
    'n_mfcc': 10
}

# STFT parameters
stft_params = {
    'n_fft': 1024,
    'hop_length': 512,
    'win_length': 1024
}

# ML model parameters
model_params = {
    'task': 'CLASSIFICATION',
    'num_trees': 500,
    'growing_strategy': 'BEST_FIRST_GLOBAL',
    'max_depth': 12
}

# Force 50%-50% dataset
force_balanced_dataset = True
