'''
Developer Notes:

- This script used a datashape of number of samples x number of timesteps x number of features, unlike tge Dense_AE script that used a datashape of number of samples x number of features.
- Due to the specific nature of the data shape, the preprocessing was different from the Dense_AE script.
- Increasing the accelerometer's sample rate through interpolation to allow for more audio features only increased the model's comlexity and did not improve performance (RNNs were unable to train due to lack of RAM).
- The precision of the various models is the most penalized metric.
- The Dense_AE model with the same pre-processing as the supervised methods is the one with the best performance.
'''

import os
import time
import numpy as np
import warnings
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from scipy import interpolate
from scipy.signal.windows import hann
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, 
                             confusion_matrix, accuracy_score, roc_curve, classification_report, precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import keras
from keras import ops
from keras import layers
from keras.callbacks import EarlyStopping
from keras_tuner import HyperModel, HyperParameters, BayesianOptimization, Hyperband
from UN_CNN_Models import RNN_DEEP, RNN_SIMPLE, CNN_SIMPLE, CNN_DEEP, Attention_AE, vae_model_builder
from AE_Aux_Func import plot_metrics_vs_threshold, plot_precision_recall_curve, find_optimal_threshold_f1



# File paths
current_dir = os.getcwd()
file_paths = {
    'good_bearing_audio_m': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'GOOD', 'AUDIO'),
    'good_bearing_acel_s': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'GOOD', 'ACEL'),
    'good_bearing_acel_m': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'GOOD', 'ACEL'),
    'good_bearing_audio_s': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'GOOD', 'AUDIO'),
    'damaged_bearing_audio_s': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'DAMAGED', 'AUDIO'),
    'damaged_bearing_acel_s': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'DAMAGED', 'ACEL'),
    'damaged_bearing_audio_m': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'DAMAGED', 'AUDIO'),
    'damaged_bearing_acel_m': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'DAMAGED', 'ACEL'),
    'smooth_floor_audio': os.path.join(current_dir, 'Dataset_Piso', 'LISO', 'SAMPLES_1s', 'AUDIO'),
    'smooth_floor_acel': os.path.join(current_dir, 'Dataset_Piso', 'LISO', 'SAMPLES_1s', 'ACCEL'),
    'tiled_floor_audio': os.path.join(current_dir, 'Dataset_Piso', 'TIJOLEIRA', 'SAMPLES_1s', 'AUDIO'),
    'tiled_floor_acel': os.path.join(current_dir, 'Dataset_Piso', 'TIJOLEIRA', 'SAMPLES_1s', 'ACCEL'),
    'noise_profile_file': os.path.join(current_dir, 'Dataset_Piso', 'Noise.WAV')
}

def sort_key(file_path):
    """Helper function to sort files by numeric part in filename."""
    file_name = os.path.basename(file_path)
    numeric_part = ''.join(filter(str.isdigit, file_name))
    return int(numeric_part) if numeric_part else 0

def load_files(directory, file_extension):
    """Load and sort files from a directory based on file extension."""
    return sorted(
        [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(file_extension)],
        key=sort_key
    )

def proc_audio(audio_file, n_fft=2048, hop_length=512, num_samples=50, mfcc=True, stft=True):
    """Compute the Short-Time Fourier Transform (STFT), Mel-Frequency Cepstral Coefficients (MFCC), and resample accelerometer data."""
    audio, sr = librosa.load(audio_file, sr=192000)
    stft_matrix = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft_matrix)
    #print("STFT Magnitude Shape: ", magnitude.shape)
    avg_magnitude, std_magnitude, avg_magnitude_mfcc, std_magnitude_mfcc = None, None, None, None
    
    if stft:
        #print("STFT Magnitude Shape: ", magnitude.shape)
        avg_magnitude, std_magnitude = apply_moving_average(magnitude, num_samples)
        #print("Average Magnitude Shape: ", avg_magnitude.shape)
        #print("Standard Deviation Magnitude Shape: ", std_magnitude.shape)
    
    if mfcc:        
        # Find the number of Mel filter banks that can be computed without any empty filters
        # Unccomment the following code if the samples proprietaries are not known, or the pre-processing parameters were changed
        n_mels = 100
        mfcc_computed = False
        '''while not mfcc_computed:
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    # Mel-frequency filter bank
                    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=500, fmax=85000, window=librosa.filters.get_window('hann', n_fft), power=2.0)
                    
                    if len(w) > 0 and any("Empty filters detected" in str(warning.message) for warning in w):
                        raise librosa.util.exceptions.ParameterError("Empty filters detected")
                    mfcc_computed = True
            
            except librosa.util.exceptions.ParameterError as e:
                if 'Empty filters detected' in str(e):
                    n_mels -= 1  # Reduce n_mels and try again
                    print(f"Reducing n_mels to {n_mels} and retrying...")
                    if n_mels < 1:
                        raise ValueError("Unable to compute MFCCs with given parameters.")
                else:
                    raise  # Re-raise any other exceptions'''
        # Mel spectrogram       
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=500, fmax=85000, window=librosa.filters.get_window('hann', n_fft), power=2.0)
        
        # Convert to dB scale
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # MFCCs
        mfccs = librosa.feature.mfcc(S=mel_spectrogram, n_mfcc=40, n_mels=n_mels, fmin=500, fmax=85000)
        #print("MFCC Magnitude Shape: ", mfccs.shape)
        avg_magnitude_mfcc, std_magnitude_mfcc = apply_moving_average(mfccs, num_samples)
        #print("Average MFCC Magnitude Shape: ", avg_magnitude_mfcc.shape)
        #print("Standard Deviation MFCC Magnitude Shape: ", std_magnitude_mfcc.shape)
    
    
    # Print shapes for debugging
    #print("Shapes: ", avg_magnitude.shape, std_magnitude.shape, avg_magnitude_mfcc.shape, std_magnitude_mfcc.shape)
    
    return avg_magnitude, std_magnitude, avg_magnitude_mfcc, std_magnitude_mfcc



def apply_moving_average(magnitude, target_frames):
    """Apply moving average and compute standard deviation to the STFT magnitude matrix."""    
    num_freq_bins, num_time_frames = magnitude.shape
    if num_time_frames <= target_frames:
        raise ValueError("Number of time frames in magnitude is less than or equal to target_frames.")
    
    window_size = num_time_frames // target_frames
    remainder = num_time_frames % target_frames
    num_full_windows = num_time_frames // window_size

    averaged_magnitude = np.zeros((num_freq_bins, target_frames))
    std_dev_magnitude = np.zeros((num_freq_bins, target_frames))
    
    for i in range(target_frames):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        if i == target_frames - 1 and remainder != 0:
            end_idx = num_time_frames
        segment = magnitude[:, start_idx:end_idx]
        averaged_magnitude[:, i] = np.mean(segment, axis=1)
        std_dev_magnitude[:, i] = np.std(segment, axis=1)
    
    return averaged_magnitude, std_dev_magnitude


def extract_raw_accel_features(csv_file_path, target_frames=50):
    """Extract features from accelerometer data CSV files and resample to match target frames."""
    columns = ['timestamp', 'accX_l', 'accY_l', 'accZ_l', 'gyrX_l', 'gyrY_l', 'gyrZ_l', 
               'accX_r', 'accY_r', 'accZ_r', 'gyrX_r', 'gyrY_r', 'gyrZ_r']
    df = pd.read_csv(csv_file_path, usecols=columns)
    accel_data = df.to_numpy()
    
    # Extract the features
    num_samples, num_features = accel_data.shape

    # Determine the resampling ratio
    original_frames = num_samples
    if original_frames == target_frames:
        resampled_data = accel_data  # No need to resample if lengths are already equal
    else:
        # Create time axis for original and target data
        original_time = np.linspace(0, 1, original_frames)
        target_time = np.linspace(0, 1, target_frames)

        # Initialize the resampled data array
        resampled_data = np.zeros((target_frames, num_features))

        # Resample each feature individually
        for feature_idx in range(num_features):
            feature_data = accel_data[:, feature_idx]
            interpolator = interpolate.interp1d(original_time, feature_data, kind='linear', fill_value='extrapolate')
            resampled_data[:, feature_idx] = interpolator(target_time)
    
    # Round the resampled data to two decimal places
    resampled_data = np.round(resampled_data, 2)
    
    return resampled_data


def preprocess_data(stft=True, mfcc=True, target_frames=50):
    """Preprocess data for training and testing."""
    # Load data
    noise_audio =   (load_files(file_paths['smooth_floor_audio'], '.WAV') +
                     load_files(file_paths['tiled_floor_audio'], '.WAV'))
    noise_accel =   (load_files(file_paths['smooth_floor_acel'], '.csv') +
                     load_files(file_paths['tiled_floor_acel'], '.csv'))
    
    good_bearing_audio = (load_files(file_paths['good_bearing_audio_s'], '.WAV') +
                          load_files(file_paths['good_bearing_audio_m'], '.WAV'))
    
    damaged_bearing_audio = (load_files(file_paths['damaged_bearing_audio_s'], '.WAV') + 
                             load_files(file_paths['damaged_bearing_audio_m'], '.WAV'))
    
    good_bearing_accel = (load_files(file_paths['good_bearing_acel_s'], '.csv') +
                          load_files(file_paths['good_bearing_acel_m'], '.csv'))
    
    damaged_bearing_accel = (load_files(file_paths['damaged_bearing_acel_s'], '.csv') +
                             load_files(file_paths['damaged_bearing_acel_m'], '.csv'))
    
    print("\nNumber of Noise Samples:", len(noise_audio))
    print("Number of Good Bearing Samples:", len(good_bearing_audio))
    print("Number of Damaged Bearing Samples:", len(damaged_bearing_audio))
    print("Number of Total Bearing Samples:", len(good_bearing_audio) + len(damaged_bearing_audio))
    
    # Process data
    audio_stft_avg_features, audio_stft_std_features, audio_mfcc_avg_features, audio_mfcc_std_features, accel_features, labels = [], [], [], [], [], []
    
    start_time_pre_proc = time.time()
    
    # Process noise data
    for audio_file, accel_file in zip(noise_audio, noise_accel):        
        avg_magnitude, std_magnitude, avg_mfcc, std_mfcc = proc_audio(audio_file, stft=stft, mfcc=mfcc, num_samples=target_frames)
        if stft and mfcc:            
            audio_stft_avg_features.append(avg_magnitude)
            audio_stft_std_features.append(std_magnitude)
            audio_mfcc_avg_features.append(avg_mfcc)
            audio_mfcc_std_features.append(std_mfcc)
        elif stft:
            audio_stft_avg_features.append(avg_magnitude)
            audio_stft_std_features.append(std_magnitude)
        elif mfcc:
            audio_mfcc_avg_features.append(avg_mfcc)
            audio_mfcc_std_features.append(std_mfcc)

        accel_features.append(extract_raw_accel_features(accel_file, target_frames=target_frames))
        labels.append(0)
    
    # Process good bearing data
    for audio_file, accel_file in zip(good_bearing_audio, good_bearing_accel):
        avg_magnitude, std_magnitude, avg_mfcc, std_mfcc = proc_audio(audio_file, stft=stft, mfcc=mfcc, num_samples=target_frames)
        if stft and mfcc:
            audio_stft_avg_features.append(avg_magnitude)
            audio_stft_std_features.append(std_magnitude)
            audio_mfcc_avg_features.append(avg_mfcc)
            audio_mfcc_std_features.append(std_mfcc)
        elif stft:
            audio_stft_avg_features.append(avg_magnitude)
            audio_stft_std_features.append(std_magnitude)
        elif mfcc:
            audio_mfcc_avg_features.append(avg_mfcc)
            audio_mfcc_std_features.append(std_mfcc)
        
        accel_features.append(extract_raw_accel_features(accel_file, target_frames=target_frames))
        labels.append(1)
        
    # Process damaged bearing data
    for audio_file, accel_file in zip(damaged_bearing_audio, damaged_bearing_accel):
        avg_magnitude, std_magnitude, avg_mfcc, std_mfcc = proc_audio(audio_file, stft=stft, mfcc=mfcc, num_samples=target_frames)
        if stft and mfcc:
            audio_stft_avg_features.append(avg_magnitude)
            audio_stft_std_features.append(std_magnitude)
            audio_mfcc_avg_features.append(avg_mfcc)
            audio_mfcc_std_features.append(std_mfcc)
        elif stft:
            audio_stft_avg_features.append(avg_magnitude)
            audio_stft_std_features.append(std_magnitude)
        elif mfcc:
            audio_mfcc_avg_features.append(avg_mfcc)
            audio_mfcc_std_features.append(std_mfcc)
        
        accel_features.append(extract_raw_accel_features(accel_file, target_frames=target_frames))
        labels.append(2)
    
    end_time_pre_proc = time.time()
    
    pre_proc_time = end_time_pre_proc - start_time_pre_proc
    
    methods_string = ""
    
    # Convert lists to numpy arrays and reshape
    if stft and mfcc:
        audio_stft_avg_features = np.array(audio_stft_avg_features)
        audio_stft_std_features = np.array(audio_stft_std_features)
        audio_mfcc_avg_features = np.array(audio_mfcc_avg_features)
        audio_mfcc_std_features = np.array(audio_mfcc_std_features)
        accel_features = np.array(accel_features)
        audio_stft_avg_features_reshaped = audio_stft_avg_features.transpose(0, 2, 1)
        audio_stft_std_features_reshaped = audio_stft_std_features.transpose(0, 2, 1)
        audio_mfcc_avg_features_reshaped = audio_mfcc_avg_features.transpose(0, 2, 1)
        audio_mfcc_std_features_reshaped = audio_mfcc_std_features.transpose(0, 2, 1)
        combined_features = np.concatenate((accel_features, audio_stft_avg_features_reshaped, audio_stft_std_features_reshaped, audio_mfcc_avg_features_reshaped, audio_mfcc_std_features_reshaped), axis=2)
        methods_string = "stft_mfcc"
    elif stft:
        audio_stft_avg_features = np.array(audio_stft_avg_features)
        audio_stft_std_features = np.array(audio_stft_std_features)
        accel_features = np.array(accel_features)
        audio_stft_avg_features_reshaped = audio_stft_avg_features.transpose(0, 2, 1)
        audio_stft_std_features_reshaped = audio_stft_std_features.transpose(0, 2, 1)
        combined_features = np.concatenate((accel_features, audio_stft_avg_features_reshaped, audio_stft_std_features_reshaped), axis=2)
        methods_string = "stft"
    elif mfcc:
        audio_mfcc_avg_features = np.array(audio_mfcc_avg_features)
        audio_mfcc_std_features = np.array(audio_mfcc_std_features)
        accel_features = np.array(accel_features)
        audio_mfcc_avg_features_reshaped = audio_mfcc_avg_features.transpose(0, 2, 1)
        audio_mfcc_std_features_reshaped = audio_mfcc_std_features.transpose(0, 2, 1)
        combined_features = np.concatenate((accel_features, audio_mfcc_avg_features_reshaped, audio_mfcc_std_features_reshaped), axis=2)
        methods_string = "mfcc"
    else:
        accel_features = np.array(accel_features)
        combined_features = np.concatenate((accel_features), axis=2)
        methods_string = "only_accel"
    
    # Normalize features
    scaler = StandardScaler()
    num_samples, time_steps, num_features = combined_features.shape
    combined_features_reshaped = combined_features.reshape(-1, num_features)
    combined_features_normalized = scaler.fit_transform(combined_features_reshaped).reshape(num_samples, time_steps, num_features)
    
    print("\nAverage Preprocessing Time per Sample:", (end_time_pre_proc - start_time_pre_proc) / len(labels))
    
    print("\nCombined Features Shape:", combined_features_normalized.shape)
    

    return combined_features_normalized, labels, methods_string, pre_proc_time

def remove_unused_trials(keras_tuner_dir, project_name, best_trial):
    """Removes all trials except for the best one."""
    trials_dir = os.path.join(keras_tuner_dir, project_name)
    
    for subdir in os.listdir(trials_dir):
        trial_dir = os.path.join(trials_dir, subdir)
        
        # Only consider files/directories with an "_" in the name
        if "_" in subdir:
            trial_id = subdir.split('_')[-1]
            if trial_id != best_trial.trial_id:
                print(f"Deleting trial {trial_id}")
                shutil.rmtree(trial_dir)
            else:
                print(f"Keeping best trial {trial_id}")
        else:
            print(f"Skipping {subdir}")

def main():
    
    '''
    A target frames value above 50 will expand the sample rate of the accelerometer to the target frames value by perfoming linear interpolation.
    A target frames value bellow 751 (for the STFT default values) will reduce the number of audio features per timestamp, by applying a moving average and standard deviation to the STFT magnitude matrix.
    High target frames values will increase the model's complexity and may lead to memory errors.
    '''
    
    target_frames_shape = 50
    stft = True
    mfcc = True
    
    # Data preprocessing
    features, labels, methods, pre_proc_time = preprocess_data(stft=stft, mfcc=mfcc, target_frames=target_frames_shape)
    
    # Noise samples (considered as regular data)
    noise_samples = features[np.array(labels) == 0]    
    noise_labels = np.array([0] * noise_samples.shape[0])  # All labels in training set are 0
 
    # Bearing samples (considered as anomalies)
    bearings_samples = features[np.array(labels) != 0]
    bearings_labels = np.array(labels)[np.array(labels) != 0] 
    
    # Split bearing samples into good and damaged bearings (For later use)
    good_bearing_samples = features[np.array(labels) == 1]
    good_bearing_labels = np.array([1] * good_bearing_samples.shape[0])
    
    damaged_bearing_samples = features[np.array(labels) == 2]
    damaged_bearing_labels = np.array([2] * damaged_bearing_samples.shape[0])   
    
    # Split noise data into training and validation sets
    X_train, X_val_complete, y_train, y_val_complete = train_test_split(noise_samples, noise_labels, test_size=0.1, random_state=42)
    
    # Further split labeled anomalies into validation and test sets
    bearings_val_complete, bearings_test_complete, bearings_val_labels_complete, bearings_test_labels_complete = train_test_split(bearings_samples, bearings_labels, test_size=0.5, random_state=42)
    
    val_size = min(len(X_val_complete), len(bearings_val_complete))
    
    X_val = X_val_complete[:val_size]
    y_val = y_val_complete[:val_size]
    
    # In an ideal situation, the test set should be independent of the training and validation sets. 
    # However, due to the lack of noise data, the noise test set is created with random samples from the the dataset.
    test_size = min(len(bearings_test_labels_complete), len(noise_labels))
    
    # Force the test set and val set to be balanced to avoid class imbalance and bias in the evaluation and threshold parameterization
    bearings_val = bearings_val_complete[:val_size]
    bearings_test = bearings_test_complete[:test_size]
    bearings_val_labels = bearings_val_labels_complete[:val_size]
    bearings_test_labels = bearings_test_labels_complete[:test_size]
    
    # The remaining bearing samples are used for the final evaluation
    # The remaining bearing samples are used for the final evaluation
    only_bearings_eval = np.concatenate((bearings_val_complete[val_size:], bearings_test_complete[test_size:]))
    only_bearings_eval_labels = np.concatenate((bearings_val_labels[val_size:], bearings_test_labels[test_size:]))
    
    
    bearings_val_labels_ae = np.ones(len(bearings_val_labels))
    bearings_test_labels_ae = np.ones(len(bearings_test_labels)) 
    bearings_eval_labels_ae = np.ones(len(only_bearings_eval_labels))
    
    # X_train and X_val shape: (number of samples, 50, 2143) - where:

    # n - number of samples
    # 50 - timestamps of each sample
    # 2143 - features per timestamp
    
    # input_shape = (50, 2143)
    
    # Model building and training

    # Epochs and batch size
    epochs = 150
    batch_size = 64

    X_train_reshaphed = np.expand_dims(X_train, -1).astype("float32")
    
    input_shape = X_train.shape[1:]
    
    # Early stopping setup
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    
    '''# Define directories
    current_dir = os.getcwd()
    tuner_dir = os.path.join(current_dir, 'Hyperband_Tuning', 'VAE')
    project_name = 'AE_TB_' + str(batch_size) + 'bs_' + methods + '_' + str(target_frames_shape)
    
    tuner = Hyperband(lambda hp: vae_model_builder(hp, input_shape=input_shape),
                     objective='loss',
                     max_epochs=epochs,
                     factor=3,
                     directory=tuner_dir,
                     project_name=project_name)
    

        
    # If the tuner directory exists, try to load the best trial, otherwise perform Bayesian optimization and save and load the best trial
    try:
        # Load the best trial
        best_trial = tuner.oracle.get_best_trials(1)[0]
        # Load the best model
        autoencoder = tuner.get_best_models(num_models=1)[0]
        
    except IndexError:
        # Perform Bayesian optimization
        tuner.search(X_train_reshaphed, X_train_reshaphed, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], validation_split=0.1, verbose=1)
        # Load the best trial
        best_trial = tuner.oracle.get_best_trials(1)[0]
        # Alternatively, you can directly get the best model
        autoencoder = tuner.get_best_models(num_models=1)[0]
        pass

    except AttributeError as e:
        print(f"Error while accessing best trial attributes: {e}")
    

    print(f"Best trial: {best_trial.trial_id}")
    print(f"Best trial value: {best_trial.score}")

    # Best hyperparameters
    hyperparameters = best_trial.hyperparameters
    print(f"Best trial hyperparameters: {hyperparameters.values}")'''
    
    # Build the model
    autoencoder = vae_model_builder(input_shape=input_shape, latent_dim=16)
    
    # Train the model
    autoencoder.fit(X_train_reshaphed, X_train_reshaphed, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], validation_split=0.1, verbose=1)    
   
    # Reconstruction and threshold finding
    start_time = time.time()
    val_predictions = autoencoder.predict(X_val)
    inference_time = time.time() - start_time
    
    # Get the reconstruction error on the validation set
    val_errors = np.mean(np.square(X_val - val_predictions), axis=(1,2))

    # Get the reconstruction error on the labeled validation anomalies
    bearing_val_predictions = autoencoder.predict(bearings_val)
    bearing_val_errors = np.mean(np.square(bearings_val - bearing_val_predictions), axis=(1,2))
    
    # Tune the threshold based on precision-recall curve
    # Labels for the validation set should be all 0 (normal) and for the labeled validation anomalies should be 1 (anomalous)
    y_val_binary = np.array([0] * len(val_errors))  # Normal data
    y_bearing_binary = np.array([1] * len(bearing_val_errors))  # Anomalous data

    # Combine validation errors and bearing validation errors
    combined_errors = np.concatenate([val_errors, bearing_val_errors])
    combined_labels = np.concatenate([y_val_binary, y_bearing_binary])

    '''# Calculate precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(combined_labels, combined_errors)

    # Calculate the absolute difference between precision and recall
    diff = abs(precision - recall)

    # Find the index of the minimum difference
    optimal_idx = np.argmin(diff)   

    # Get the optimal threshold
    optimal_threshold = thresholds[optimal_idx]'''
    
    #optimal_threshold = find_optimal_threshold_f1(combined_errors, combined_labels)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(combined_labels, combined_errors)

    # Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)  # This gives you the threshold with the maximum difference between TPR and FPR
    optimal_threshold = thresholds[optimal_idx]

    print(f"Optimal Threshold (ROC): {optimal_threshold}")

    print(f"Optimal threshold: {optimal_threshold}")
    
    # Final evaluation on the test set
    bearing_test_predictions = autoencoder.predict(bearings_test)
    bearing_test_errors = np.mean(np.square(bearings_test - bearing_test_predictions), axis=(1,2))

    # Randomly select a predefined number of noise samples to include in the test set
    # In an ideal situation, this set would be independent set, i.e., not used for training or validation
    predefined_noise_samples_count = len(bearings_test)  # Set the desired number of noise samples
    random_noise_indices = np.random.choice(noise_samples.shape[0], predefined_noise_samples_count, replace=False)
    selected_noise_samples = noise_samples[random_noise_indices]
    selected_noise_labels = noise_labels[random_noise_indices]

    # Predict the reconstruction error for the selected noise samples
    noise_test_predictions = autoencoder.predict(selected_noise_samples)
    noise_test_errors = np.mean(np.square(selected_noise_samples - noise_test_predictions), axis=(1,2))

    # Combine the noise and bearing errors
    combined_test_errors = np.concatenate([bearing_test_errors, noise_test_errors])
    combined_test_labels = np.concatenate([bearings_test_labels_ae, selected_noise_labels])

    # Determine which samples are anomalies based on the optimal threshold
    test_anomalies = combined_test_errors > optimal_threshold

    # Calculate and print the final detection metrics
    predicted_labels = test_anomalies.astype(int)

    print(classification_report(combined_test_labels, predicted_labels, target_names=["Noise", "Anomaly"]))
    
    print("Reconstruction Errors:")
    print("Average Reconstruction Error for Validation Data (only noise):")
    print(f'Mean Reconstruction Error: {np.mean(val_errors)}')
    print(f'Standard Deviation of Reconstruction Error: {np.std(val_errors)}')
    print("Average Reconstruction Error for Validation Data (only bearings):")
    print(f'Mean Reconstruction Error: {np.mean(bearing_val_predictions)}')
    print(f'Standard Deviation of Reconstruction Error: {np.std(bearing_val_errors)}')
    print("\n")
    
    # Global Evaluation
    print("Evaluation:")
    print(f"Optimal Threshold: {optimal_threshold:.3f}")  
    print(f"Accuracy: {accuracy_score(combined_test_labels, predicted_labels):.3f}")
    print(f"Precision: {precision_score(combined_test_labels, predicted_labels):.3f}")
    print(f"Recall: {recall_score(combined_test_labels, predicted_labels):.3f}")
    print(f"F1 Score: {f1_score(combined_test_labels, predicted_labels):.3f}")
    print(f"AUC: {roc_auc_score(combined_test_labels, predicted_labels):.3f}")
    print(f"Average inference time per sample: {(inference_time / len(X_val)) * 1000:.3f} ms")
    print(f"Average processing time per sample: {(pre_proc_time / (len(y_train) + len(y_val)) * 1000):.3f} ms")


    # Count the number of noise samples and bearings samples in the test set
    unique, counts = np.unique(combined_test_labels, return_counts=True)
    label_counts = dict(zip(unique, counts))

    print(f"Number of noise samples (0) in the test set: {label_counts.get(0, 0)}")
    print(f"Number of bearing samples (1) in the test set: {label_counts.get(1, 0)}")

    cm = confusion_matrix(combined_test_labels, predicted_labels)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Noise', 'Anomaly'],
                yticklabels=['Noise', 'Anomaly'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Final test with left over bearing samples
    print("Final Evaluation:")
    only_bearings_eval_predictions = autoencoder.predict(only_bearings_eval)
    only_bearings_eval_errors = np.mean(np.square(only_bearings_eval - only_bearings_eval_predictions), axis=(1,2))
    detected_labels = (only_bearings_eval_errors > optimal_threshold).astype(int)
    detected_anomalies = np.count_nonzero(detected_labels)
    detected_anomalies_per = detected_anomalies / len(only_bearings_eval)
    print(f"Detected Anomalies: {detected_anomalies}")
    print(f"Detected Anomalies Percentage: {detected_anomalies_per}")
    
    
    # Thresholds vs metrics
    thresholds = np.linspace(min(combined_errors), max(combined_errors), 100)
    f1_scores_test = []
    precisions_test = []
    recalls_test = []
    accuracy_test = []
    roc_aucs_test = []
    for threshold in thresholds:
        y_test_pred = (combined_test_errors > threshold).astype(int)
        f1_scores_test.append(f1_score(combined_test_labels, y_test_pred))
        accuracy_test.append(accuracy_score(combined_test_labels, y_test_pred))
        precisions_test.append(precision_score(combined_test_labels, y_test_pred, zero_division=0))
        recalls_test.append(recall_score(combined_test_labels, y_test_pred))
        roc_aucs_test.append(roc_auc_score(combined_test_labels, combined_test_errors))
    
    # Plot metrics vs threshold
    plot_metrics_vs_threshold(thresholds, f1_scores_test, accuracy_test, precisions_test, recalls_test, roc_aucs_test, optimal_threshold)
        
if __name__ == "__main__":
    main()