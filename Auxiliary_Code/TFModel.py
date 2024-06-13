import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import scipy.signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Flatten, Concatenate, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Function to extract label from filename
def extract_label_from_filename(filename):
    if 'LISO' in filename.upper():
        return 0  # Label 0 for LISO
    elif 'TIJOLEIRA' in filename.upper():
        return 1  # Label 1 for TIJOLEIRA
    else:
        raise ValueError("Filename does not contain a valid label")

def load_wav_file(file_path):
    print(f"Loading WAV file: {file_path}")
    audio, sr = librosa.load(file_path, sr=192000)
    print(f"Loaded WAV file with shape: {audio.shape} and sample rate: {sr}")
    return audio

def load_csv_file(file_path):
    print(f"Loading CSV file: {file_path}")
    df = pd.read_csv(file_path)
    timestamps = df['timestamp'].values
    sensor_data = df[['accX_l', 'accY_l', 'accZ_l', 'gyrX_l', 'gyrY_l', 'gyrZ_l',
                      'accX_r', 'accY_r', 'accZ_r', 'gyrX_r', 'gyrY_r', 'gyrZ_r']].values
    print(f"Loaded CSV file with shape: {sensor_data.shape}")
    return timestamps, sensor_data

def extract_audio_features(audio, sr=192000):
    print(f"Extracting audio features from audio with shape: {audio.shape}")
    n_mfcc = 40  # Adjust as needed
    fmax = sr // 2  # Nyquist frequency
    n_fft = 4096  # FFT window size
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, fmax=fmax, window = scipy.signal.windows.hann(n_fft, False), 
                                     win_length = 4096, hop_length=int(0.01*sr), n_mels=128, fmin=0, htk=True, norm='ortho')
        mfccs = np.mean(mfccs.T, axis=0)
        print(f"Extracted MFCCs with shape: {mfccs.shape}")
    except Exception as e:
        print(f"Error extracting MFCCs: {e}")
        mfccs = np.zeros(n_mfcc)  # Default to zeros if extraction fails
    return mfccs

def load_dataset(audio_dir, sensor_dir):
    audio_files = sorted([os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.WAV')])
    sensor_files = sorted([os.path.join(sensor_dir, f) for f in os.listdir(sensor_dir) if f.endswith('.csv')])

    audio_features = []
    sensor_features = []
    labels = []

    for audio_file, sensor_file in zip(audio_files, sensor_files):
        audio = load_wav_file(audio_file)
        _, sensors = load_csv_file(sensor_file)
        label = extract_label_from_filename(audio_file)
        
        audio_feat = extract_audio_features(audio)
        
        audio_features.append(audio_feat)
        sensor_features.append(sensors)
        labels.append(label)

    # Pad sensor features to ensure consistent length
    max_len = max(sensor.shape[0] for sensor in sensor_features)
    sensor_features = pad_sequences(sensor_features, maxlen=max_len, padding='post', dtype='float32')
    
    print(f"Loaded dataset with {len(audio_features)} audio features and {len(sensor_features)} sensor features")
    return np.array(audio_features), sensor_features, np.array(labels)
# Load dataset
current_dir = os.getcwd()
audio_dir = os.path.join(current_dir, 'Dataset_Piso', 'AUDIO')
sensor_dir = os.path.join(current_dir, 'Dataset_Piso', 'ACEL')

audio_features, sensor_features, labels = load_dataset(audio_dir, sensor_dir)

print(f"Audio features shape: {audio_features.shape}")
print(f"Sensor features length: {len(sensor_features)}")
print(f"Labels shape: {labels.shape}")

# Standardize audio features
scaler_audio = StandardScaler()
audio_features = scaler_audio.fit_transform(audio_features)
print(f"Standardized audio features shape: {audio_features.shape}")

# Standardize sensor features
scaler_sensor = StandardScaler()
sensor_features = [scaler_sensor.fit_transform(sensor) for sensor in sensor_features]
print(f"Standardized sensor features shapes: {[sensor.shape for sensor in sensor_features]}")

# Split dataset into training and testing sets
X_audio_train, X_audio_test, X_sensor_train, X_sensor_test, y_train, y_test = train_test_split(
    audio_features, sensor_features, labels, test_size=0.2, random_state=42)

print(f"Training audio features shape: {X_audio_train.shape}")
print(f"Testing audio features shape: {X_audio_test.shape}")
print(f"Training sensor features length: {len(X_sensor_train)}")
print(f"Testing sensor features length: {len(X_sensor_test)}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing labels shape: {y_test.shape}")

# Create TensorFlow datasets
def generator(audio_data, sensor_data, labels):
    for audio, sensor, label in zip(audio_data, sensor_data, labels):
        yield (audio, sensor), label

train_dataset = tf.data.Dataset.from_generator(lambda: generator(X_audio_train, X_sensor_train, y_train),
                                               output_signature=(
                                                   (tf.TensorSpec(shape=(40,), dtype=tf.float32),
                                                    tf.TensorSpec(shape=(None, 12), dtype=tf.float32)),
                                                   tf.TensorSpec(shape=(), dtype=tf.int32)))

test_dataset = tf.data.Dataset.from_generator(lambda: generator(X_audio_test, X_sensor_test, y_test),
                                              output_signature=(
                                                  (tf.TensorSpec(shape=(40,), dtype=tf.float32),
                                                   tf.TensorSpec(shape=(None, 12), dtype=tf.float32)),
                                                  tf.TensorSpec(shape=(), dtype=tf.int32)))

# Batch and prefetch the datasets
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Build the model
def create_model():
    # Audio input branch
    audio_input = Input(shape=(40,))
    audio_branch = Dense(128, activation='relu')(audio_input)
    audio_branch = Dropout(0.3)(audio_branch)
    audio_branch = Dense(64, activation='relu')(audio_branch)

    # Sensor input branch
    sensor_input = Input(shape=(None, 12))
    sensor_branch = LSTM(128, activation='relu', return_sequences=True)(sensor_input)
    sensor_branch = LSTM(64, activation='relu')(sensor_branch)
    sensor_branch = Flatten()(sensor_branch)
    sensor_branch = Dense(128, activation='relu')(sensor_branch)
    sensor_branch = Dropout(0.3)(sensor_branch)
    sensor_branch = Dense(64, activation='relu')(sensor_branch)

    # Concatenate branches
    concatenated = Concatenate()([audio_branch, sensor_branch])
    concatenated = Dense(64, activation='relu')(concatenated)
    concatenated = Dropout(0.3)(concatenated)
    output = Dense(1, activation='sigmoid')(concatenated)

    model = Model(inputs=[audio_input, sensor_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

model = create_model()

# Train the model
print("Starting model training...")
history = model.fit(train_dataset, validation_data=test_dataset, epochs=20)
print("Model training finished.")

# Evaluate the model
print("Evaluating model...")
loss, accuracy = model.evaluate(test_dataset)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')
