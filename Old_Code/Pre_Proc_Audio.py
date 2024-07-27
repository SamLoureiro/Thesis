import os
import wave
import math
import numpy as np
import pandas as pd
import librosa
import librosa.display
import scipy.signal
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.utils import class_weight, shuffle
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report


# Function to load and preprocess audio files with MFCC and Log-MFE

# Sampling rate (sr): Number of samples per second. In this case, sr=192000.
# Frame length (n_fft): The length of the FFT window, which is 2048 samples in this case.
# Hop length (hop_length): The number of samples between successive frames, which is 512 samples in this case
# Number of Time Frames = [(Total Number of Samples − Frame Length)/Hop Length​⌉ + 1 = [(1*192000 - 2048)/512] + 1 = 372
# 1/372 = 0.00268 seconds per frame

def preprocess_audio(file_path, sr=192000, n_mfcc=20, max_pad_len=372):
    # Load the audio file
    wave, sample_rate = librosa.load(file_path, sr=sr)

    # Calculate the maximum possible n_fft given the signal length
    max_n_fft = len(wave)  # Maximum n_fft cannot exceed the length of the signal

    # Choose n_fft dynamically based on the signal length
    n_fft = min(2048, max_n_fft)  # Adjust 2048 based on your signal characteristics

    # Parameters for MFCC computation
    hop_length = n_fft // 4  # or adjust as needed
    n_mels = 128  # Adjust based on your signal characteristics
    fmax = sr / 2  # Set fmax to Nyquist frequency (96 kHz)

    # Compute Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=wave, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, 
                                                     window=scipy.signal.get_window('hamming', n_fft), htk=False, 
                                                     center=True, pad_mode='reflect', power=2.0, 
                                                     n_mels=n_mels, fmax=fmax)
    # Convert power spectrogram to decibel (log) scale
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Pad or truncate the log mel spectrogram to a fixed length
    if log_mel_spectrogram.shape[1] > max_pad_len:
        log_mel_spectrogram = log_mel_spectrogram[:, :max_pad_len]
    elif log_mel_spectrogram.shape[1] < max_pad_len:
        pad_width = max_pad_len - log_mel_spectrogram.shape[1]
        log_mel_spectrogram = np.pad(log_mel_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Compute MFCCs using the log mel spectrogram
    mfcc = librosa.feature.mfcc(S=log_mel_spectrogram, sr=sample_rate, n_mfcc=n_mfcc)

    # Pad or truncate MFCCs to a fixed length
    if mfcc.shape[1] > max_pad_len:
        mfcc = mfcc[:, :max_pad_len]
    elif mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    
    return mfcc, log_mel_spectrogram

def feature_normalize(data):
    mu = np.mean(data, axis=0)    # Calculate mean
    sigma = np.std(data, axis=0)  # Calculate standard deviation
    return (data - mu) / sigma    # Normalize data

# Function to segment accelerometer data into overlapping windows
def segment_signal(data, label, window_size=10):
    segments = np.empty((0, window_size, data.shape[1] - 1))
    labels = np.empty((0))
    for start_index in range(0, len(data) - window_size + 1, int(window_size / 2)):  # Overlapping windows
        end_index = start_index + window_size
        # Extract accelerometer and gyroscope features for left and right sensors
        ax_l = data["accX_l"][start_index:end_index]
        ay_l = data["accY_l"][start_index:end_index]
        az_l = data["accZ_l"][start_index:end_index]
        gx_l = data["gyrX_l"][start_index:end_index]
        gy_l = data["gyrY_l"][start_index:end_index]
        gz_l = data["gyrZ_l"][start_index:end_index]
        ax_r = data["accX_r"][start_index:end_index]
        ay_r = data["accY_r"][start_index:end_index]
        az_r = data["accZ_r"][start_index:end_index]
        gx_r = data["gyrX_r"][start_index:end_index]
        gy_r = data["gyrY_r"][start_index:end_index]
        gz_r = data["gyrZ_r"][start_index:end_index]
        segments = np.vstack([segments, np.dstack([ax_l, ay_l, az_l, gx_l, gy_l, gz_l, ax_r, ay_r, az_r, gx_r, gy_r, gz_r])])
        labels = np.append(labels, label)
    return segments, labels

# Function to preprocess both audio and accelerometer data for sensor fusion
def preprocess_data(audio_file, accel_file, class_label, accel_window_size=10):
    # Preprocess the audio file to extract MFCC features
    try:
        max_pad_len = int((1*192000 - 2048)/512 + 1)
        max_audio_len = max_pad_len
        mfcc_features, mel_features = preprocess_audio(audio_file, max_pad_len)
    except ValueError as e:
        print(e)
        return None, None
    
    # Load the accelerometer data from the CSV file
    accel_data = pd.read_csv(accel_file)
    
    # Normalize each column in the accelerometer data
    for col in ['accX_l', 'accY_l', 'accZ_l', 'gyrX_l', 'gyrY_l', 'gyrZ_l', 
                'accX_r', 'accY_r', 'accZ_r', 'gyrX_r', 'gyrY_r', 'gyrZ_r']:
        accel_data[col] = feature_normalize(accel_data[col])
    
    # Segment the normalized accelerometer data into overlapping windows
    accel_segments, _ = segment_signal(accel_data, class_label, accel_window_size)
    
    # Add an extra dimension to the audio features to match the shape for Conv2D input
    mfcc_features = np.expand_dims(mfcc_features, axis=2)
    print("MFCC features: ", mfcc_features.shape)
    # Determine the number of segments obtained from the accelerometer data
    num_segments = accel_segments.shape[0]
    
    # Repeat the audio features to match the number of accelerometer segments
    # Here, each segment of accelerometer data will be paired with the same audio features
    mfcc_features_resampled = np.repeat(mfcc_features[:, :, np.newaxis], num_segments, axis=2)
    print("MFCC features resampled 1: ", mfcc_features_resampled.shape)
    # Move the new axis to the front to match the shape (num_segments, max_audio_len, n_mfcc, 1)
    mfcc_features_resampled = np.moveaxis(mfcc_features_resampled, 2, 0)

    print("MFCC features resampled 2: ", mfcc_features_resampled.shape)
    
    # Initialize an array to hold the combined features (audio and accelerometer)
    combined_features = np.zeros((num_segments, max(max_audio_len, accel_window_size), 13))
    print("Combined features shape: ", combined_features.shape)
    # Fill the first channel (index 0) of combined features with audio features
    combined_features[:, :mfcc_features_resampled.shape[1], 0] = mfcc_features_resampled[:, :, 0, 0]
    print("Combined features shape 2: ", combined_features.shape)
    # Fill the remaining channels (index 1 to 12) with accelerometer segments
    combined_features[:, :accel_segments.shape[1], 1:] = accel_segments
    print("Combined features shape 3: ", combined_features.shape)
    # Return the combined features and the corresponding class label
    return combined_features, class_label


# Directory paths for audio and accelerometer data
current_dir = os.getcwd()
tijoleira_dir_audio = os.path.join(current_dir, 'Dataset_Piso', 'TIJOLEIRA', 'SAMPLES_1s', 'AUDIO')
liso_dir_audio = os.path.join(current_dir, 'Dataset_Piso', 'LISO', 'SAMPLES_1s', 'AUDIO')
tijoleira_dir_acel = os.path.join(current_dir, 'Dataset_Piso', 'TIJOLEIRA', 'SAMPLES_1s', 'ACCEL')
liso_dir_acel = os.path.join(current_dir, 'Dataset_Piso', 'LISO', 'SAMPLES_1s', 'ACCEL')


# Helper function to sort files by the numeric part in the filename
def sort_key(file_path):
    file_name = os.path.basename(file_path)
    # Extract the numeric part from the filename for sorting
    numeric_part = ''.join(filter(str.isdigit, file_name))
    return int(numeric_part) if numeric_part else 0

# Load list of audio and accelerometer files
tijoleira_files_audio = sorted(
    [os.path.join(tijoleira_dir_audio, file) for file in os.listdir(tijoleira_dir_audio) if file.endswith('.WAV')],
    key=sort_key
)
liso_files_audio = sorted(
    [os.path.join(liso_dir_audio, file) for file in os.listdir(liso_dir_audio) if file.endswith('.WAV')],
    key=sort_key
)
tijoleira_files_acel = sorted(
    [os.path.join(tijoleira_dir_acel, file) for file in os.listdir(tijoleira_dir_acel) if file.endswith('.csv')],
    key=sort_key
)
liso_files_acel = sorted(
    [os.path.join(liso_dir_acel, file) for file in os.listdir(liso_dir_acel) if file.endswith('.csv')],
    key=sort_key
)

X = []
y = []
info = True

# Preprocess Tijoleira class data
for audio_file, accel_file in zip(tijoleira_files_audio, tijoleira_files_acel):
    sample_duration = 1
    if info:
        input_wav = wave.open(audio_file, 'rb')
        framerate = input_wav.getframerate()
        sample_width = input_wav.getsampwidth()
        channels = input_wav.getnchannels()
        total_frames = input_wav.getnframes()
        total_duration = total_frames / framerate
        print("Frame rate")
        print(framerate)
        print("Sample width")
        print(sample_width)
        print("Channels")
        print(channels)
        print("Total frames")
        print(total_frames)
        print("Total duration")
        print(total_duration)
        info = False 
    #print("Preprocessing file ", audio_file, " and ", accel_file)
    features, label = preprocess_data(audio_file, accel_file, 0)  # 0 for Tijoleira class
    if features is not None:
        X.append(features)
        y.append(label)

# Preprocess Liso class data
for audio_file, accel_file in zip(liso_files_audio, liso_files_acel):
    #print("Preprocessing file ", audio_file, " and ", accel_file)
    features, label = preprocess_data(audio_file, accel_file, 1)  # 1 for Liso class
    if features is not None:
        X.append(features)
        y.append(label)

# One-hot encode labels
y = to_categorical(y, num_classes=2)  # Assuming 2 classes (Tijoleira and Liso)

# Shuffle the data
X, y = shuffle(X, y, random_state=42)

print("Data preprocessing completed.")

print("X lenght: ", len(X))   # 949 samples
print("y lenght: ", len(y))   # 949 samples
print("X[] shape: ", X[0].shape)  # (9, 372, 13)
print("X[] shape: ", len(X[0][0]))  # (9, 372, 13)
print("X[] shape: ", len(X[1][1]))  # (9, 372, 13)
print("X[] shape: ", len(X[2][2]))   # (9, 372, 13)

# Convert X to numpy array
X = np.array(X)
