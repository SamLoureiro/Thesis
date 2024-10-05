# config.py

# Preprocessing options
preprocessing_options = {
    'sp': False,
    'noise_reduction': False,
    'fft': False,
    'mfcc': True,
    'stft': False,
    'sp_accel': False
}

# Noise reduction parameters
noise_reduction_params = {
    'n_fft': 2048,        # Number of FFT points
    'hop_length': 512,    # Hop size between successive frames
    'prop_decrease': 0.5  # Property for noise reduction
}

# FFT parameters
fft_params = {
    'n_fft': 2048,  # Number of FFT points  
    'fmin': 500,    # Minimum frequency (Hz)
    'fmax': 80000   # Maximum frequency (Hz)
}

# MFCC parameters
mfcc_params = {
    'n_fft': 2048,       # Number of FFT points
    'hop_length': 512,   # Hop size between successive frames
    'n_mels': 100,       # Number of Mel filter banks
    'fmin': 500,         # Minimum frequency (Hz)
    'fmax': 80000,       # Maximum frequency (Hz)
    'n_mfcc': 40         # Number of MFCC coefficients
}

# STFT parameters
stft_params = {
    'n_fft': 2048,       # Number of FFT points
    'hop_length': 512,   # Hop size between successive frames
    'fmin': 500,         # Minimum frequency (Hz)
    'fmax': 80000        # Maximum frequency (Hz)
    
# Frequency resolution = sampling rate / window size = 192000 / 2048 ≈ 93.75 Hz
# Time resolution = hop size / sampling rate = 512 / 192000 ≈ 0.00267 s
# This means you can observe changes in the frequency content every 2.67 milliseconds.
}

# Balance dataset options (if both are True, the smote will be used)

# Use Smote to balance dataset
smote = True

# Force 50%-50% dataset
force_balanced_dataset = False


# Model (Choose only one. If both are True, the GBDT model will be used)
model = {
    'GBDT': False,
    'RF': True
}

# GBDT model parameters
model_params_GBDT = {
    'num_trees': 100,       #100                                # Default: 300                              
    'growing_strategy': 'BEST_FIRST_GLOBAL',                    # Default: 'LOCAL'
    'max_depth': -1,                                            # Default: 6
    'early_stopping': 'LOSS_INCREASE',                          # Default: 'LOSS_INCREASE'
    'split_axis': 'SPARSE_OBLIQUE',                             # Default: 'AXIS_ALIGNED'
    'sparse_oblique_num_projections_exponent': 1.0,             # Default: 1.0  (only for 'SPARSE_OBLIQUE')
    'max_num_nodes': 100,                                       # Default: None
    'l1_regularization': 0.01,         #0.01                    # Default: 0.0
    'l2_regularization': 0.01,                                  # Default: 0.0
    'shrinkage': 0.01,                                          # Default: 0.1
}

# RF model parameters
model_params_RF = { 
    'num_trees': 300,                                           # Default: 300                              
    'growing_strategy': 'BEST_FIRST_GLOBAL',                    # Default: 'LOCAL'
    'max_depth': -1,                                            # Default: 16      -1 = no limit
    'split_axis': 'SPARSE_OBLIQUE',                             # Default: 'AXIS_ALIGNED'
    'sparse_oblique_num_projections_exponent': 1.0,             # Default: 1.0  (only for 'SPARSE_OBLIQUE')
    'max_num_nodes': 1000,                                      # Default: None              
    'winner_take_all': False                                    # Default: True                               
}

# Save model
save_model = False

# Save Metrics
save_metrics = False

# Load model
model_load = False  

# Save HTLM
save_html = False

# Cross-validation
cross_validation = False