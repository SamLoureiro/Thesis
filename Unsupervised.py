import os
import time
import warnings
import numpy as np
import pandas as pd
import librosa
import PreProc_Function as ppf
import config
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


# Define directories
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
faiures_accel = damaged_bearing_files_acel_s + damaged_bearing_files_acel_m

# Regular Dataset
regular_audio = good_bearing_files_audio_s + good_bearing_files_audio_m + smooth_floor_files_audio + tiled_floor_files_audio
regular_accel = good_bearing_files_acel_s + good_bearing_files_acel_m + smooth_floor_files_acel + tiled_floor_files_acel


# Extract features for each file and combine them
combined_features = []
labels = []

# Create a string based on the methods that are on
methods_string = "_".join(method for method, value in config.preprocessing_options.items() if value)

start_time_pre_proc = time.time()

# Process good_bearing files
for audio_file, accel_file in zip(regular_audio, regular_accel):
    audio_features = ppf.extract_audio_features(audio_file, noise_profile_file, config.preprocessing_options)
    accel_features = ppf.extract_accel_features(accel_file)
    combined = {**audio_features, **accel_features}
    combined_features.append(combined)
    labels.append(0)  # 0 for good_bearing


n_samples_healthy = len(combined_features)
print(f"Number of samples (Healthy Bearing): {n_samples_healthy}")

count = 0
# Process damaged_bearing files
for audio_file, accel_file in zip(failures_audio, faiures_accel):
    audio_features = ppf.extract_audio_features(audio_file, noise_profile_file, config.preprocessing_options)
    accel_features = ppf.extract_accel_features(accel_file)
    combined = {**audio_features, **accel_features}
    combined_features.append(combined)
    labels.append(1)  # 1 for damaged_bearing

end_time_pre_proc = time.time()
    
n_samples_damaged = len(combined_features) - n_samples_healthy
print(f"Number of samples (Damaged Bearing): {n_samples_damaged}")

# Create DataFrame
combined_features_df = pd.DataFrame(combined_features)

# Normalize features
scaler = StandardScaler()
combined_features_normalized = scaler.fit_transform(combined_features_df)

# Convert labels to numpy array
y = np.array(labels)

# Shuffle the data and labels
combined_features_normalized, y = shuffle(combined_features_normalized, y, random_state=42)
