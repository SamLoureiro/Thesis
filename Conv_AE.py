import os
import time
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, 
                             confusion_matrix, accuracy_score)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import (LeakyReLU, Input, Dense, BatchNormalization, Dropout, LSTM, 
                          RepeatVector, TimeDistributed, Bidirectional)
from keras.models import Model

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

def stft(audio_file, n_fft=2048, hop_length=512, num_samples=50):
    """Compute the Short-Time Fourier Transform (STFT) and apply a moving average."""
    audio, sr = librosa.load(audio_file, sr=192000)
    stft_matrix = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft_matrix)
    return apply_moving_average(magnitude, num_samples)

def apply_moving_average(magnitude, target_frames):
    """Apply moving average to the STFT magnitude matrix."""
    num_freq_bins, num_time_frames = magnitude.shape
    if num_time_frames <= target_frames:
        raise ValueError("Number of time frames in magnitude is less than or equal to target_frames.")
    
    window_size = num_time_frames // target_frames
    remainder = num_time_frames % target_frames
    num_full_windows = num_time_frames // window_size

    averaged_magnitude = np.zeros((num_freq_bins, target_frames))
    
    for i in range(target_frames):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        if i == target_frames - 1 and remainder != 0:
            end_idx = num_time_frames
        averaged_magnitude[:, i] = np.mean(magnitude[:, start_idx:end_idx], axis=1)
    
    return averaged_magnitude

def extract_raw_accel_features(csv_file_path):
    """Extract features from accelerometer data CSV files."""
    columns = ['timestamp', 'accX_l', 'accY_l', 'accZ_l', 'gyrX_l', 'gyrY_l', 'gyrZ_l', 
               'accX_r', 'accY_r', 'accZ_r', 'gyrX_r', 'gyrY_r', 'gyrZ_r']
    df = pd.read_csv(csv_file_path, usecols=columns)
    return df.to_numpy()

def build_rnn_autoencoder(input_shape):
    """Build and compile an RNN autoencoder model."""
    inputs = Input(shape=input_shape)
    
    # Encoder
    x = Bidirectional(LSTM(128, activation='relu', return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    
    x = Bidirectional(LSTM(64, activation='relu', return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    
    encoded = LSTM(32, activation='relu', return_sequences=False)(x)
    encoded = Dropout(0.2)(encoded)
    encoded = BatchNormalization()(encoded)
    
    # Bottleneck
    bottleneck = Dense(16, activation='relu')(encoded)
    bottleneck = RepeatVector(input_shape[0])(bottleneck)
    
    # Decoder
    x = LSTM(32, activation='relu', return_sequences=True)(bottleneck)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    
    decoded = LSTM(128, activation='relu', return_sequences=True)(x)
    outputs = TimeDistributed(Dense(input_shape[1]))(decoded)
    
    autoencoder = Model(inputs, outputs)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

def find_optimal_threshold(reconstruction_error, y_true):
    """Find the optimal threshold for reconstruction error."""
    thresholds = np.linspace(min(reconstruction_error), max(reconstruction_error), 100)
    best_threshold = 0
    best_f1 = 0
    for threshold in thresholds:
        y_pred = (reconstruction_error > threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold

def preprocess_data():
    """Preprocess data for training and testing."""
    # Load data
    regular_audio = load_files(file_paths['good_bearing_audio_s'], '.WAV') + load_files(file_paths['good_bearing_audio_m'], '.WAV') + \
                    load_files(file_paths['smooth_floor_audio'], '.WAV') + load_files(file_paths['tiled_floor_audio'], '.WAV')
    regular_accel = load_files(file_paths['good_bearing_acel_s'], '.csv') + load_files(file_paths['good_bearing_acel_m'], '.csv') + \
                     load_files(file_paths['smooth_floor_acel'], '.csv') + load_files(file_paths['tiled_floor_acel'], '.csv')
    
    failures_audio = load_files(file_paths['damaged_bearing_audio_s'], '.WAV') + load_files(file_paths['damaged_bearing_audio_m'], '.WAV')
    failures_accel = load_files(file_paths['damaged_bearing_acel_s'], '.csv') + load_files(file_paths['damaged_bearing_acel_m'], '.csv')
    
    # Process data
    audio_features, accel_features, labels = [], [], []
    for audio_file, accel_file in zip(regular_audio, regular_accel):
        audio_features.append(stft(audio_file))
        accel_features.append(extract_raw_accel_features(accel_file))
        labels.append(0)
    
    for audio_file, accel_file in zip(failures_audio, failures_accel):
        audio_features.append(stft(audio_file))
        accel_features.append(extract_raw_accel_features(accel_file))
        labels.append(1)
    
    audio_features = np.array(audio_features)
    accel_features = np.array(accel_features)
    
    # Reshape and combine features
    audio_features_reshaped = audio_features.transpose(0, 2, 1)
    combined_features = np.concatenate((accel_features, audio_features_reshaped), axis=2)
    
    # Normalize features
    scaler = StandardScaler()
    num_samples, time_steps, num_features = combined_features.shape
    combined_features_reshaped = combined_features.reshape(-1, num_features)
    combined_features_normalized = scaler.fit_transform(combined_features_reshaped).reshape(num_samples, time_steps, num_features)
    
    return train_test_split(combined_features_normalized, labels, test_size=0.2, random_state=42)

def main():
    # Data preprocessing
    X_train, X_test, y_train, y_test = preprocess_data()
    
    # Model building and training
    input_shape = X_train.shape[1:]
    autoencoder = build_rnn_autoencoder(input_shape)
    
    print("Training autoencoder...")
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=64, validation_split=0.1, verbose=1)
    
    # Reconstruction and threshold finding
    X_train_pred = autoencoder.predict(X_train)
    X_test_pred = autoencoder.predict(X_test)
    
    train_reconstruction_error = np.mean(np.abs(X_train - X_train_pred), axis=(1, 2))
    test_reconstruction_error = np.mean(np.abs(X_test - X_test_pred), axis=(1, 2))
    
    optimal_threshold = find_optimal_threshold(train_reconstruction_error, y_train)
    
    y_test_pred = (test_reconstruction_error > optimal_threshold).astype(int)
    
    # Evaluation
    print("Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred)}")
    print(f"Precision: {precision_score(y_test, y_test_pred)}")
    print(f"Recall: {recall_score(y_test, y_test_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_test_pred)}")
    print(f"AUC: {roc_auc_score(y_test, test_reconstruction_error)}")
    
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

if __name__ == "__main__":
    main()
