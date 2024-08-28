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
from keras.callbacks import EarlyStopping
from keras_tuner import HyperModel, HyperParameters, BayesianOptimization
from UN_CNN_Models import RNN_DEEP, RNN_SIMPLE, CNN_SIMPLE, CNN_DEEP, Attention_AE
from AE_Aux_Func import plot_metrics_vs_threshold



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

def proc_audio(audio_file, n_fft=2048, hop_length=1024, num_samples=50, mfcc=True, stft=True):
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
        # Mel spectrogram       
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, n_mels=50, fmin=500, fmax=85000, window=librosa.filters.get_window('hann', n_fft), power=2.0)
        
        # Convert to dB scale
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # MFCCs
        mfccs = librosa.feature.mfcc(S=mel_spectrogram, n_mfcc=40, n_mels=106, fmin=500, fmax=85000)
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
    
    # Split data: train only on noise data (label 0), test on both noise and bearing data
    noise_samples = features[np.array(labels) == 0]
    noise_labels= np.array([0] * noise_samples.shape[0])  # All labels in training set are 0
    
    bearings_samples = features[np.array(labels) != 0]
    bearings_labels = np.array(labels)[np.array(labels) != 0] 
    
    good_bearing_samples = features[np.array(labels) == 1]
    good_bearing_labels = np.array([1] * good_bearing_samples.shape[0])
    
    damaged_bearing_samples = features[np.array(labels) == 2]
    damaged_bearing_labels = np.array([2] * damaged_bearing_samples.shape[0])   
    
    X_train, X_test_noise, y_train, y_test_noise = train_test_split(noise_samples, noise_labels, test_size=0.2, random_state=42)
    
    X_test_size = len(X_test_noise)
    
    n_bearing_samples = len(bearings_labels)
    bearings_labels_ae = np.ones(n_bearing_samples)
    
    X_test_bearings, y_test_bearings = shuffle(bearings_samples, bearings_labels, random_state=42)
    
    X_test = np.concatenate((X_test_noise, X_test_bearings[:X_test_size]), axis=0)
    y_test = np.concatenate((y_test_noise, bearings_labels_ae[:X_test_size]), axis=0)
    
    # X_train and X_test shape: (number of samples, 50, 2143) - where:

    # n - number of samples
    # 50 - timestamps of each sample
    # 2143 - features per timestamp
    
    # input_shape = (50, 2143)
    
    # Model building and training

    input_shape = X_train.shape[1:]
    # Epochs and batch size
    epochs = 100
    batch_size = 32

    # Define directories
    current_dir = os.getcwd()
    tuner_dir = os.path.join(current_dir, 'Bayesian_Tuning', 'RNN_SIMPLE')
    project_name = 'AE_TB_' + str(batch_size) + 'bs_' + methods + '_' + str(target_frames_shape)


    # Define the BayesianOptimization tuner
    tuner = BayesianOptimization(
        hypermodel=RNN_DEEP(input_shape),
        objective='val_loss',
        max_trials=10,
        executions_per_trial=1,
        directory=tuner_dir,
        project_name=project_name
    )

    # Early stopping setup
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    # If the tuner directory exists, try to load the best trial, otherwise perform Bayesian optimization and save and load the best trial
    try:
        # Load the best trial
        best_trial = tuner.oracle.get_best_trials(1)[0]
        
    except IndexError:
        # Perform Bayesian optimization
        tuner.search(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping], verbose=1)    
        # Load the best trial
        best_trial = tuner.oracle.get_best_trials(1)[0]
        pass

    except AttributeError as e:
        print(f"Error while accessing best trial attributes: {e}")


    print(f"Best trial: {best_trial.trial_id}")
    print(f"Best trial value: {best_trial.score}")

    # Best hyperparameters
    hyperparameters = best_trial.hyperparameters
    print(f"Best trial hyperparameters: {hyperparameters.values}")

    # Build the best model using the best hyperparameters
    autoencoder = RNN_SIMPLE(input_shape).build(hyperparameters)
    
    # Reconstruction and threshold finding
    X_train_pred = autoencoder.predict(X_train)
    start_time = time.time()
    X_test_pred = autoencoder.predict(X_test)
    inference_time = time.time() - start_time
    
    train_reconstruction_error = np.mean(np.abs(X_train - X_train_pred), axis=(1, 2))
    test_reconstruction_error = np.mean(np.abs(X_test - X_test_pred), axis=(1, 2))
    
    print("\nTrain Reconstruction Error:", train_reconstruction_error)
    print("Test Reconstruction Error:", test_reconstruction_error)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, test_reconstruction_error)

    # Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)  # This gives you the threshold with the maximum difference between TPR and FPR
    optimal_threshold = thresholds[optimal_idx]
    
    '''# Determine the optimal threshold using precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, test_reconstruction_error)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]'''
   
    y_test_pred = (test_reconstruction_error > optimal_threshold).astype(int)
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['BEARING', 'NOISE']))
    print("\n")    
    
    # Global Evaluation
    print("Evaluation:")
    print(f"Optimal Threshold: {optimal_threshold:.3f}")  
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_test_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_test_pred):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_test_pred):.3f}")
    print(f"AUC: {roc_auc_score(y_test, test_reconstruction_error):.3f}")
    print(f"Average inference time per sample: {(inference_time / len(X_test)) * 1000:.3f} ms")
    print(f"Average processing time per sample: {(pre_proc_time / (len(y_train) + len(y_test)) * 1000):.3f} ms")

    # Count the number of noise samples and bearings samples in the test set
    unique, counts = np.unique(y_test, return_counts=True)
    label_counts = dict(zip(unique, counts))

    print(f"Number of noise samples (0) in the test set: {label_counts.get(0, 0)}")
    print(f"Number of bearing samples (1) in the test set: {label_counts.get(1, 0)}")

    cm = confusion_matrix(y_test, y_test_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Thresholds vs metrics
    
    # Use the IQR method to filter out outliers
    Q1 = np.percentile(test_reconstruction_error, 25)
    Q3 = np.percentile(test_reconstruction_error, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_errors = test_reconstruction_error[(test_reconstruction_error >= lower_bound) & (test_reconstruction_error <= upper_bound)]

    # Generate thresholds within the filtered range
    thresholds = np.linspace(min(filtered_errors), max(filtered_errors), 100)
    #thresholds = np.linspace(min(test_reconstruction_error), max(test_reconstruction_error), 100)
    f1_scores_test = []
    precisions_test = []
    recalls_test = []
    accuracy_test = []
    roc_aucs_test = []
    #f1_scores_train = []
    #accuracy_train = []
    #precision_train = []
    #recalls_train = []
    #roc_aucs_train = []

    for threshold in thresholds:
        y_test_pred = (test_reconstruction_error > threshold).astype(int)
        f1_scores_test.append(f1_score(y_test, y_test_pred))
        accuracy_test.append(accuracy_score(y_test, y_test_pred))
        precisions_test.append(precision_score(y_test, y_test_pred, zero_division=0))
        recalls_test.append(recall_score(y_test, y_test_pred))
        roc_aucs_test.append(roc_auc_score(y_test, test_reconstruction_error))
        #y_train_pred = (train_reconstruction_error > threshold).astype(int)
        #f1_scores_train.append(f1_score(y_train, y_train_pred))
        #accuracy_train.append(accuracy_score(y_train, y_train_pred))
        #precision_train.append(precision_score(y_train, y_train_pred, zero_division=0))
        #recalls_train.append(recall_score(y_train, y_train_pred))
        #roc_aucs_train.append(roc_auc_score(y_train, train_reconstruction_error))
    
    # Plot metrics vs threshold
    plot_metrics_vs_threshold(thresholds, f1_scores_test, accuracy_test, precisions_test, recalls_test, roc_aucs_test, optimal_threshold)


    remove_unused_trials(tuner_dir, project_name, best_trial)

if __name__ == "__main__":
    main()
