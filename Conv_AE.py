'''

Notas de desenvolvimento:

- Aumentar a sample rate do acelerómetro para permitir mais features de áudio apenas aumentou a comlexidade do modelo e não melhorou a performance (o RNN_Complex não conseguiu treinar por falta de RAM)
- A precisão dos vários modelos é a mais penalizada

'''

import os
import time
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from scipy.signal.windows import hann
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, 
                             confusion_matrix, accuracy_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from unsuper_time_models import RNN_Complex, RNN_Simple, CNN_Complex, CNN_Simple
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

def proc_audio(audio_file, n_fft=2048, hop_length=512, num_samples=100, mfcc=True, stft=True):
    """Compute the Short-Time Fourier Transform (STFT), Mel-Frequency Cepstral Coefficients (MFCC), and resample accelerometer data."""
    audio, sr = librosa.load(audio_file, sr=192000)
    stft_matrix = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft_matrix)
    #print("STFT Magnitude Shape: ", magnitude.shape)
    if stft:
        avg_magnitude, std_magnitude = apply_moving_average(magnitude, num_samples)
    
    if mfcc:
        # Mel spectrogram       
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, n_mels=50, fmin=500, fmax=85000, window=librosa.filters.get_window('hann', n_fft), power=2.0)
        
        # Convert to dB scale
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # MFCCs
        mfccs = librosa.feature.mfcc(S=mel_spectrogram, n_mfcc=40, n_mels=106, fmin=500, fmax=85000)
        #print("MFCC Magnitude Shape: ", mfccs.shape)
        avg_magnitude_mfcc, std_magnitude_mfcc = apply_moving_average(mfccs, num_samples)
    
    
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


def extract_raw_accel_features(csv_file_path, target_frames=100):
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
    
    start_time_pre_proc = time.time()
    
    for audio_file, accel_file in zip(regular_audio, regular_accel):
        avg_magnitude, std_magnitude, avg_mfcc, std_mfcc = proc_audio(audio_file)
        audio_features.append(avg_magnitude)
        audio_std_features.append(std_magnitude)
        audio_mfcc_features.append(avg_mfcc)
        audio_mfcc_std_features.append(std_mfcc)
        accel_features.append(extract_raw_accel_features(accel_file))
        labels.append(0)
    
    for audio_file, accel_file in zip(failures_audio, failures_accel):
        avg_magnitude, std_magnitude, avg_mfcc, std_mfcc = proc_audio(audio_file)
        audio_features.append(avg_magnitude)
        audio_std_features.append(std_magnitude)
        audio_mfcc_features.append(avg_mfcc)
        audio_mfcc_std_features.append(std_mfcc)
        accel_features.append(extract_raw_accel_features(accel_file))
        labels.append(1)
    
    end_time_pre_proc = time.time()
    
    
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
    #combined_features = np.concatenate((accel_features, audio_mfcc_features_reshaped, audio_mfcc_std_features_reshaped), axis=2)
    # Normalize features
    scaler = StandardScaler()
    num_samples, time_steps, num_features = combined_features.shape
    combined_features_reshaped = combined_features.reshape(-1, num_features)
    combined_features_normalized = scaler.fit_transform(combined_features_reshaped).reshape(num_samples, time_steps, num_features)
    
    print("\nAverage Preprocessing Time per Sample:", (end_time_pre_proc - start_time_pre_proc) / len(labels))
    
    print("\nCombined Features Shape:", combined_features_normalized.shape)
    
    return train_test_split(combined_features_normalized, labels, test_size=0.2, random_state=42)


def main():
    # Data preprocessing
    X_train, X_test, y_train, y_test = preprocess_data()
    
    # Model building and training
    input_shape = X_train.shape[1:]
    autoencoder = RNN_Simple(input_shape)
    
    print("Training autoencoder...")
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=64, validation_split=0.1, verbose=1)
    
    # Reconstruction and threshold finding
    X_train_pred = autoencoder.predict(X_train)
    start_time = time.time()
    X_test_pred = autoencoder.predict(X_test)
    inference_time = time.time() - start_time
    
    train_reconstruction_error = np.mean(np.abs(X_train - X_train_pred), axis=(1, 2))
    test_reconstruction_error = np.mean(np.abs(X_test - X_test_pred), axis=(1, 2))
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, test_reconstruction_error)

    # Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)  # This gives you the threshold with the maximum difference between TPR and FPR
    optimal_threshold = thresholds[optimal_idx]
   
    y_test_pred = (test_reconstruction_error > optimal_threshold).astype(int)

    # Evaluation
    print("Evaluation:")
    print(f"Optimal Threshold: {optimal_threshold:.3f}")  
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_test_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_test_pred):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_test_pred):.3f}")
    print(f"AUC: {roc_auc_score(y_test, test_reconstruction_error):.3f}")
    print(f"Average inference time per sample: {(inference_time / len(X_test)) * 1000:.3f} ms")
    
    
    
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

    # Calculate metrics for each threshold
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

    # Plot metrics vs threshold
    plot_metrics_vs_threshold(thresholds, f1_scores_test, precisions_test, recalls_test, roc_aucs_test,
                            f1_scores_train, precision_train, recalls_train, roc_aucs_train,
                            optimal_threshold)

if __name__ == "__main__":
    main()
