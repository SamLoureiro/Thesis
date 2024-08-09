import os
import time
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from scipy.signal.windows import hann
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, 
                             confusion_matrix, accuracy_score, roc_curve, precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from keras.layers import (LeakyReLU, Input, Dense, BatchNormalization, Dropout, LSTM, 
                          RepeatVector, TimeDistributed, Bidirectional)
from keras.models import Model
import plotly.graph_objects as go




# Function to find the optimal threshold for reconstruction error
def find_optimal_threshold(reconstruction_error, y_true):
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

def reduce_dimensions(X, method='PCA'):
    if method == 'PCA':
        # Flatten the 3D array to 2D
        num_samples, time_steps, num_features = X.shape
        X_flattened = X.reshape(num_samples, time_steps * num_features)
        
        reducer = PCA(n_components=2)
        X_reduced = reducer.fit_transform(X_flattened)
        
    elif method == 't-SNE':
        # Flatten the 3D array to 2D
        num_samples, time_steps, num_features = X.shape
        X_flattened = X.reshape(num_samples, time_steps * num_features)
        
        reducer = TSNE(n_components=2, random_state=42)
        X_reduced = reducer.fit_transform(X_flattened)
        
    else:
        raise ValueError("Method should be 'PCA' or 't-SNE'")
    
    return X_reduced



# Function to plot the reduced data
def plot_reduced_data(X_reduced, y, y_pred=None, title="2D Map of Samples"):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                          c=y, cmap='coolwarm', alpha=0.6, edgecolors='w', s=50, label='True Label')
    if y_pred is not None:
        y_pred = np.ravel(y_pred)
        plt.scatter(X_reduced[y_pred == 1, 0], X_reduced[y_pred == 1, 1], 
                    c='red', marker='x', s=100, label='Detected Anomalies')
    plt.colorbar(scatter, label='True Label')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

def plot_metrics_vs_threshold(thresholds, f1_scores_test, precisions_test, recalls_test, roc_aucs_test,
                              f1_scores_train, precision_train, recalls_train, roc_aucs_train,
                              optimal_threshold):
    fig = go.Figure()

    # Add traces for test metrics
    fig.add_trace(go.Scatter(x=thresholds, y=f1_scores_test, mode='lines', name='F1 Score_Test'))
    fig.add_trace(go.Scatter(x=thresholds, y=precisions_test, mode='lines', name='Precision_Test'))
    fig.add_trace(go.Scatter(x=thresholds, y=recalls_test, mode='lines', name='Recall_Test'))
    fig.add_trace(go.Scatter(x=thresholds, y=roc_aucs_test, mode='lines', name='ROC-AUC_Test'))

    # Add traces for train metrics
    fig.add_trace(go.Scatter(x=thresholds, y=f1_scores_train, mode='lines', name='F1 Score_Train'))
    fig.add_trace(go.Scatter(x=thresholds, y=precision_train, mode='lines', name='Precision_Train'))
    fig.add_trace(go.Scatter(x=thresholds, y=recalls_train, mode='lines', name='Recall_Train'))
    fig.add_trace(go.Scatter(x=thresholds, y=roc_aucs_train, mode='lines', name='ROC-AUC_Train'))

    # Add vertical line for optimal threshold
    fig.add_vline(x=optimal_threshold, line=dict(color='red', width=2, dash='dash'),
                  annotation_text='Optimal Threshold', annotation_position='top right')

    # Update layout for better visualization
    fig.update_layout(
        title='Metrics vs. Threshold',
        xaxis_title='Threshold',
        yaxis_title='Score',
        legend_title='Metric',
        template='plotly_dark',
        showlegend=True
    )

    # Show the figure
    fig.show()

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

def stft(audio_file, n_fft=2048, hop_length=512, num_samples=50, mfcc = True, stft = True):
    """Compute the Short-Time Fourier Transform (STFT) and apply a moving average with standard deviation."""
    audio, sr = librosa.load(audio_file, sr=192000)
    stft_matrix = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft_matrix)
    if stft:
        avg_magnitude, std_magnitude = apply_moving_average(magnitude, num_samples)
    if mfcc:
        
        # Mel spectrogram       
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, n_mels=50, fmin=500, fmax=85000, window = hann(2048), power=2.0)
        
        # Convert to dB scale
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # MFCCs
        mfccs = librosa.feature.mfcc(S=mel_spectrogram, n_mfcc=40, n_mels=106, fmin=500, fmax=85000)
        #print(mfccs.shape)
        avg_magnitude_mfcc, std_magnitude_mfcc = apply_moving_average(mfccs, num_samples)
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
    autoencoder.compile(optimizer='adam', loss='msle')
    
    return autoencoder

def build_simpler_rnn_autoencoder(input_shape):
    """Build and compile a simpler RNN autoencoder model."""
    inputs = Input(shape=input_shape)
    
    # Encoder
    x = Bidirectional(LSTM(64, activation='relu', return_sequences=True))(inputs)
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
    autoencoder.compile(optimizer='adam', loss='msle')
    
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

def find_optimal_threshold_cost_based(reconstruction_error, y_true, cost_fp=2, cost_fn=0.5):
    thresholds = np.linspace(min(reconstruction_error), max(reconstruction_error), 100)
    best_threshold = 0
    best_cost = float('inf')
    
    for threshold in thresholds:
        y_pred = (reconstruction_error > threshold).astype(int)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        cost = cost_fp * fp + cost_fn * fn
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold
    
    return best_threshold


def preprocess_data():
    """Preprocess data for training and testing."""
    # Load data
    regular_audio = (load_files(file_paths['good_bearing_audio_s'], '.WAV') +
                     load_files(file_paths['good_bearing_audio_m'], '.WAV') +
                     load_files(file_paths['smooth_floor_audio'], '.WAV') +
                     load_files(file_paths['tiled_floor_audio'], '.WAV'))
    regular_accel = (load_files(file_paths['good_bearing_acel_s'], '.csv') +
                     load_files(file_paths['good_bearing_acel_m'], '.csv') +
                     load_files(file_paths['smooth_floor_acel'], '.csv') +
                     load_files(file_paths['tiled_floor_acel'], '.csv'))
    
    failures_audio = (load_files(file_paths['damaged_bearing_audio_s'], '.WAV') +
                      load_files(file_paths['damaged_bearing_audio_m'], '.WAV'))
    failures_accel = (load_files(file_paths['damaged_bearing_acel_s'], '.csv') +
                      load_files(file_paths['damaged_bearing_acel_m'], '.csv'))
    
    # Process data
    audio_features, audio_std_features, audio_mfcc_features, audio_mfcc_std_features, accel_features, labels = [], [], [], [], [], []
    
    for audio_file, accel_file in zip(regular_audio, regular_accel):
        avg_magnitude, std_magnitude, avg_mfcc, std_mfcc = stft(audio_file)
        audio_features.append(avg_magnitude)
        audio_std_features.append(std_magnitude)
        audio_mfcc_features.append(avg_mfcc)
        audio_mfcc_std_features.append(std_mfcc)
        accel_features.append(extract_raw_accel_features(accel_file))
        labels.append(0)
    
    for audio_file, accel_file in zip(failures_audio, failures_accel):
        avg_magnitude, std_magnitude, avg_mfcc, std_mfcc = stft(audio_file)
        audio_features.append(avg_magnitude)
        audio_std_features.append(std_magnitude)
        audio_mfcc_features.append(avg_mfcc)
        audio_mfcc_std_features.append(std_mfcc)
        accel_features.append(extract_raw_accel_features(accel_file))
        labels.append(1)
    
    audio_features = np.array(audio_features)
    audio_std_features = np.array(audio_std_features)
    audio_mfcc_features = np.array(audio_mfcc_features)
    audio_mfcc_std_features = np.array(audio_mfcc_std_features)
    accel_features = np.array(accel_features)
    
    # Reshape and combine features
    audio_features_reshaped = audio_features.transpose(0, 2, 1)
    audio_std_features_reshaped = audio_std_features.transpose(0, 2, 1)
    audio_mfcc_features_reshaped = audio_mfcc_features.transpose(0, 2, 1)
    audio_mfcc_std_features_reshaped = audio_mfcc_std_features.transpose(0, 2, 1)
    
    combined_features = np.concatenate((accel_features, audio_features_reshaped, audio_std_features_reshaped, audio_mfcc_features_reshaped, audio_mfcc_std_features_reshaped), axis=2)
    
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
    autoencoder = build_simpler_rnn_autoencoder(input_shape)
    
    print("Training autoencoder...")
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=64, validation_split=0.1, verbose=1)
    
    # Reconstruction and threshold finding
    X_train_pred = autoencoder.predict(X_train)
    X_test_pred = autoencoder.predict(X_test)
    
    train_reconstruction_error = np.mean(np.abs(X_train - X_train_pred), axis=(1, 2))
    test_reconstruction_error = np.mean(np.abs(X_test - X_test_pred), axis=(1, 2))
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, test_reconstruction_error)

    # Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)  # This gives you the threshold with the maximum difference between TPR and FPR
    optimal_threshold = thresholds[optimal_idx]

    print(f"Optimal Threshold (ROC): {optimal_threshold}")
    
    y_test_pred = (test_reconstruction_error > optimal_threshold).astype(int)

    # Evaluation
    print("Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred)}")
    print(f"Precision: {precision_score(y_test, y_test_pred)}")
    print(f"Recall: {recall_score(y_test, y_test_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_test_pred)}")
    print(f"AUC: {roc_auc_score(y_test, test_reconstruction_error)}")
    print(f"Optimal Threshold: {optimal_threshold}")


    ##################################################
    #F1-Score and AUC Results:
    
    
    
    
    # Count the number of good and damaged bearings in the test set
    unique, counts = np.unique(y_test, return_counts=True)
    label_counts = dict(zip(unique, counts))

    print(f"Number of good bearings (0) in the test set: {label_counts.get(0, 0)}")
    print(f"Number of damaged bearings (1) in the test set: {label_counts.get(1, 0)}")
    
    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Reduce dimensions and plot
    X_reduced = reduce_dimensions(X_test, method='t-SNE')
    plot_reduced_data(X_reduced, y_test)
    plot_reduced_data(X_reduced, y_test, y_test_pred)

    # Thresholds vs metrics
    thresholds = np.linspace(min(test_reconstruction_error), max(test_reconstruction_error), 100)
    f1_scores_test = []
    precisions_test = []
    recalls_test = []
    roc_aucs_test = []
    f1_scores_train= []
    precision_train = []
    recalls_train = []
    roc_aucs_train = []


    for threshold in thresholds:
        y_test_pred = (test_reconstruction_error > threshold).astype(int)
        f1_scores_test.append(f1_score(y_test, y_test_pred))
        precisions_test.append(precision_score(y_test, y_test_pred, zero_division=0))
        recalls_test.append(recall_score(y_test, y_test_pred))
        roc_aucs_test.append(roc_auc_score(y_test, test_reconstruction_error))
        y_train_pred = (train_reconstruction_error > threshold).astype(int)
        f1_scores_train.append(f1_score(y_train, y_train_pred))
        precision_train.append(precision_score(y_train, y_train_pred, zero_division=0))
        recalls_train.append(recall_score(y_train, y_train_pred))
        roc_aucs_train.append(roc_auc_score(y_train, train_reconstruction_error))

    '''plt.figure(figsize=(12, 8))
    plt.plot(thresholds, f1_scores_test, label='F1 Score_Test')
    plt.plot(thresholds, precisions_test, label='Precision_Test')
    plt.plot(thresholds, recalls_test, label='Recall_Test')
    plt.plot(thresholds, roc_aucs_test, label='ROC-AUC_Test')
    plt.plot(thresholds, f1_scores_train, label='F1 Score_Train')
    plt.plot(thresholds, precision_train, label='Precision_Train')
    plt.plot(thresholds, recalls_train, label='Recall_Train')
    plt.plot(thresholds, roc_aucs_train, label='ROC-AUC_Train')

    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()'''
    plot_metrics_vs_threshold(thresholds, f1_scores_test, precisions_test, recalls_test, roc_aucs_test,
                            f1_scores_train, precision_train, recalls_train, roc_aucs_train,
                            optimal_threshold)

if __name__ == "__main__":
    main()
