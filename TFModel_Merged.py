import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
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

# Function to preprocess audio files by extracting MFCCs
def preprocess_audio(audio_file, sr=192000, n_mfcc=40, max_pad_len=3500):
    wave, sample_rate = librosa.load(audio_file, sr=sr)  # Load audio file
    mfcc = librosa.feature.mfcc(y=wave, sr=sample_rate, n_mfcc=n_mfcc)  # Extract MFCC features
    if mfcc.shape[1] > max_pad_len:
        mfcc = mfcc[:, :max_pad_len]  # Truncate if necessary
    else:
        pad_width = max_pad_len - mfcc.shape[1]  # Pad if necessary
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

def feature_normalize(data):
    mu = np.mean(data, axis=0)    # Calculate mean
    sigma = np.std(data, axis=0)  # Calculate standard deviation
    return (data - mu) / sigma    # Normalize data

# Function to segment accelerometer data into overlapping windows
def segment_signal(data, label, window_size=100):
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
def preprocess_data(audio_file, accel_file, class_label, max_audio_len=3737, accel_window_size=100):
    # Preprocess the audio file to extract MFCC features
    audio_features = preprocess_audio(audio_file, max_pad_len=max_audio_len)
    
    # Load the accelerometer data from the CSV file
    accel_data = pd.read_csv(accel_file)
    
    # Normalize each column in the accelerometer data
    for col in ['accX_l', 'accY_l', 'accZ_l', 'gyrX_l', 'gyrY_l', 'gyrZ_l', 
                'accX_r', 'accY_r', 'accZ_r', 'gyrX_r', 'gyrY_r', 'gyrZ_r']:
        accel_data[col] = feature_normalize(accel_data[col])
    
    # Segment the normalized accelerometer data into overlapping windows
    accel_segments, _ = segment_signal(accel_data, class_label, accel_window_size)
    
    # Add an extra dimension to the audio features to match the shape for Conv2D input
    audio_features = np.expand_dims(audio_features, axis=2)
    
    # Determine the number of segments obtained from the accelerometer data
    num_segments = accel_segments.shape[0]
    
    # Repeat the audio features to match the number of accelerometer segments
    # Here, each segment of accelerometer data will be paired with the same audio features
    audio_features_resampled = np.repeat(audio_features[:, :, np.newaxis], num_segments, axis=2)
    
    # Move the new axis to the front to match the shape (num_segments, max_audio_len, n_mfcc, 1)
    audio_features_resampled = np.moveaxis(audio_features_resampled, 2, 0)
    
    # Initialize an array to hold the combined features (audio and accelerometer)
    combined_features = np.zeros((num_segments, max(max_audio_len, accel_window_size), 13))
    
    # Fill the first channel (index 0) of combined features with audio features
    combined_features[:, :audio_features_resampled.shape[1], 0] = audio_features_resampled[:, :, 0, 0]
    
    # Fill the remaining channels (index 1 to 12) with accelerometer segments
    combined_features[:, :accel_segments.shape[1], 1:] = accel_segments
    
    # Return the combined features and the corresponding class label
    return combined_features, class_label


# Directory paths for audio and accelerometer data
current_dir = os.getcwd()
tijoleira_dir_audio = os.path.join(current_dir, 'Dataset_Piso', 'TIJOLEIRA', 'SAMPLES', 'AUDIO')
liso_dir_audio = os.path.join(current_dir, 'Dataset_Piso', 'LISO', 'SAMPLES', 'AUDIO')
tijoleira_dir_acel = os.path.join(current_dir, 'Dataset_Piso', 'TIJOLEIRA', 'SAMPLES', 'ACEL')
liso_dir_acel = os.path.join(current_dir, 'Dataset_Piso', 'LISO', 'SAMPLES', 'ACEL')

# Load list of audio and accelerometer files
tijoleira_files_audio = [os.path.join(tijoleira_dir_audio, file) for file in os.listdir(tijoleira_dir_audio) if file.endswith('.WAV')]
liso_files_audio = [os.path.join(liso_dir_audio, file) for file in os.listdir(liso_dir_audio) if file.endswith('.WAV')]
tijoleira_files_acel = [os.path.join(tijoleira_dir_acel, file) for file in os.listdir(tijoleira_dir_acel) if file.endswith('.csv')]
liso_files_acel = [os.path.join(liso_dir_acel, file) for file in os.listdir(liso_dir_acel) if file.endswith('.csv')]

X = []
y = []

# Preprocess Tijoleira class data
for audio_file, accel_file in zip(tijoleira_files_audio, tijoleira_files_acel):
    features, label = preprocess_data(audio_file, accel_file, 0)  # 0 for Tijoleira class
    X.append(features)
    y.append(label)

# Preprocess Liso class data
for audio_file, accel_file in zip(liso_files_audio, liso_files_acel):
    features, label = preprocess_data(audio_file, accel_file, 1)  # 1 for Liso class
    X.append(features)
    y.append(label)

# One-hot encode labels
y = to_categorical(y, num_classes=2)  # Assuming 2 classes (Tijoleira and Liso)

# Shuffle the data
X, y = shuffle(X, y, random_state=42)

# Remove samples with incorrect dimensions
i = 0
while i < len(X):
    if len(X[i]) != 9:
        X.pop(i)
        y = np.delete(y, i, axis=0)
        continue
    i += 1

# Convert X to numpy array
X = np.array(X)

# Shape of X and y
print('X shape:', X.shape)
print('y shape:', y.shape)

# Expand dimensions for model input
X = np.expand_dims(X, axis=-1)

# Define the model
model = Sequential()

# Input shape is (num_segments, max(max_audio_len, accel_window_size), 13, 1)
input_shape = (X.shape[1], X.shape[2], X.shape[3])

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

# Flatten and dense layers
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

# Train the model
history = model.fit(X, y, validation_split=0.2, epochs=40, batch_size=16, callbacks=[early_stopping, reduce_lr])

# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')