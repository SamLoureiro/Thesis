import os
import time
import warnings
import numpy as np
import pandas as pd
import librosa
import PreProc_Function_Conv as ppf
import config_Conv as config
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.layers import LeakyReLU, Input, Dense, BatchNormalization, Dropout
from keras.models import Model
from keras.callbacks import Callback


def stft(audio_file, n_fft = 2048, hop_length = 512, num_samples = 50):
    
  """Args:
    audio: The audio signal as a numpy array.
    sr: The sampling rate of the audio signal.
    n_fft: The FFT window size.
    hop_length: The hop length between STFT frames.
    num_samples: The number of samples to extract per frequency bin.

  Returns:
    A numpy array of shape (n_fft // 2 + 1, num_samples) containing the magnitude values.
  """
  
  audio, sr = librosa.load(audio_file, sr=192000)

  # Compute the STFT
  stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
  magnitude = np.abs(stft)

  # Calculate the number of time frames
  num_frames = magnitude.shape[1]

  # Calculate the step size for selecting time frames
  step_size = num_frames // num_samples

  # Select the desired time frames
  selected_frames = magnitude[:, ::step_size]

  return selected_frames

def apply_moving_average(magnitude, target_frames):
    num_freq_bins, num_time_frames = magnitude.shape
    if num_time_frames <= target_frames:
        raise ValueError("Number of time frames in magnitude is less than or equal to target_frames.")
    
    # Calculate the width of the moving average window
    window_size = num_time_frames // target_frames
    remainder = num_time_frames % target_frames
    
    # Pad with zeros if necessary to make the number of frames divisible by window_size
    if remainder != 0:
        padding = window_size - remainder
        magnitude = np.pad(magnitude, ((0, 0), (0, padding)), mode='constant')

    # Apply moving average
    num_frames_padded = magnitude.shape[1]
    averaged_magnitude = magnitude.reshape(num_freq_bins, target_frames, window_size).mean(axis=2)
    
    return averaged_magnitude

def extract_raw_accel_features(csv_file_path):

    # Define the columns to read from the CSV
    columns = ['timestamp', 'accX_l', 'accY_l', 'accZ_l', 'gyrX_l', 'gyrY_l', 'gyrZ_l', 
            'accX_r', 'accY_r', 'accZ_r', 'gyrX_r', 'gyrY_r', 'gyrZ_r']

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path, usecols=columns)

    # Convert the DataFrame to a NumPy array
    data_array = df.to_numpy()

    # Optionally, print the shape of the array to verify
    #print("Shape of the NumPy array:", data_array.shape)

    # Optionally, print the first few rows to verify the data
    #print("First few rows of the NumPy array:")
    #print(data_array[:5])
    return data_array

current_dir = os.getcwd()

good_bearing_dir_audio_m = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'GOOD', 'AUDIO')
good_bearing_dir_acel_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'GOOD', 'ACEL')
good_bearing_dir_acel_m = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'GOOD', 'ACEL')
good_bearing_dir_audio_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'GOOD', 'AUDIO')

damaged_bearing_dir_audio_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'DAMAGED', 'AUDIO')
damaged_bearing_dir_acel_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'DAMAGED', 'ACEL')
damaged_bearing_dir_audio_m = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'DAMAGED', 'AUDIO')
damaged_bearing_dir_acel_m = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'DAMAGED', 'ACEL')

smooth_floor_audio = os.path.join(current_dir, 'Dataset_Piso', 'LISO', 'SAMPLES_1s', 'AUDIO')
smooth_floor_acel = os.path.join(current_dir, 'Dataset_Piso', 'LISO','SAMPLES_1s', 'ACCEL')
tiled_floor_audio = os.path.join(current_dir, 'Dataset_Piso', 'TIJOLEIRA','SAMPLES_1s', 'AUDIO')
tiled_floor_acel = os.path.join(current_dir, 'Dataset_Piso', 'TIJOLEIRA','SAMPLES_1s', 'ACCEL')

# Define noise profile file
noise_profile_file = os.path.join(current_dir, 'Dataset_Piso', 'Noise.WAV')

# Helper function to sort files by the numeric part in the filename
def sort_key(file_path):
    file_name = os.path.basename(file_path)
    numeric_part = ''.join(filter(str.isdigit, file_name))
    return int(numeric_part) if numeric_part else 0

# Load list of audio and accelerometer files for AMR_MOVEMENT
good_bearing_files_audio_m = sorted(
    [os.path.join(good_bearing_dir_audio_m, file) for file in os.listdir(good_bearing_dir_audio_m) if file.endswith('.WAV')],
    key=sort_key
)
damaged_bearing_files_audio_m = sorted(
    [os.path.join(damaged_bearing_dir_audio_m, file) for file in os.listdir(damaged_bearing_dir_audio_m) if file.endswith('.WAV')],
    key=sort_key
)
good_bearing_files_acel_m = sorted(
    [os.path.join(good_bearing_dir_acel_m, file) for file in os.listdir(good_bearing_dir_acel_m) if file.endswith('.csv')],
    key=sort_key
)
damaged_bearing_files_acel_m = sorted(
    [os.path.join(damaged_bearing_dir_acel_m, file) for file in os.listdir(damaged_bearing_dir_acel_m) if file.endswith('.csv')],
    key=sort_key
)

# Load list of audio and accelerometer files for AMR_STOPPED
good_bearing_files_audio_s = sorted(
    [os.path.join(good_bearing_dir_audio_s, file) for file in os.listdir(good_bearing_dir_audio_s) if file.endswith('.WAV')],
    key=sort_key
)
damaged_bearing_files_audio_s = sorted(
    [os.path.join(damaged_bearing_dir_audio_s, file) for file in os.listdir(damaged_bearing_dir_audio_s) if file.endswith('.WAV')],
    key=sort_key
)
good_bearing_files_acel_s = sorted(
    [os.path.join(good_bearing_dir_acel_s, file) for file in os.listdir(good_bearing_dir_acel_s) if file.endswith('.csv')],
    key=sort_key
)
damaged_bearing_files_acel_s = sorted(
    [os.path.join(damaged_bearing_dir_acel_s, file) for file in os.listdir(damaged_bearing_dir_acel_s) if file.endswith('.csv')],
    key=sort_key
)

# Load list of audio and accelerometer files for smooth floor
smooth_floor_files_audio = sorted(
    [os.path.join(smooth_floor_audio, file) for file in os.listdir(smooth_floor_audio) if file.endswith('.WAV')],
    key=sort_key
)
smooth_floor_files_acel = sorted(
    [os.path.join(smooth_floor_acel, file) for file in os.listdir(smooth_floor_acel) if file.endswith('.csv')],
    key=sort_key
)

# Load list of audio and accelerometer files for tiled floor
tiled_floor_files_audio = sorted(
    [os.path.join(tiled_floor_audio, file) for file in os.listdir(tiled_floor_audio) if file.endswith('.WAV')],
    key=sort_key
)
tiled_floor_files_acel = sorted(
    [os.path.join(tiled_floor_acel, file) for file in os.listdir(tiled_floor_acel) if file.endswith('.csv')],
    key=sort_key
)

# Failure Dataset
failures_audio = damaged_bearing_files_audio_s + damaged_bearing_files_audio_m
failures_accel = damaged_bearing_files_acel_s + damaged_bearing_files_acel_m

# Regular Dataset
regular_audio = good_bearing_files_audio_s + good_bearing_files_audio_m + smooth_floor_files_audio + tiled_floor_files_audio
regular_accel = good_bearing_files_acel_s + good_bearing_files_acel_m + smooth_floor_files_acel + tiled_floor_files_acel

print(f"Number of regular samples: {len(regular_audio)}")
print(f"Number of failure samples: {len(failures_audio)}")
print(f"Total number of samples: {len(regular_audio) + len(failures_audio)}")


start_time_pre_proc = time.time()

print("Preprocessing data...")

# Lists to store features
audio_features = []
accel_features = []
labels = []

# Process good_bearing files
for audio_file, accel_file in zip(regular_audio, regular_accel):
    audio_features.append(stft(audio_file))
    accel_features.append(extract_raw_accel_features(accel_file))
    
    labels.append(0)  # 0 for good_bearing

# Process damaged_bearing files
for audio_file, accel_file in zip(failures_audio, failures_accel):
    audio_features.append(stft(audio_file))
    accel_features.append(extract_raw_accel_features(accel_file))
    
    labels.append(1)  # 1 for damaged_bearing

print("Data preprocessing complete.")

print("Time taken for preprocessing: {:.2f} seconds".format(time.time() - start_time_pre_proc))

print("Number of audio samples:", len(audio_features))
print("Number of accel samples:", len(audio_features))
print("Shape of audio features:", audio_features[0].shape)
print("Shape of accel features:", accel_features[0].shape)

