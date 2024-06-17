import numpy as np
import pandas as pd
import librosa
import os
import scipy.signal
from scipy.stats import kurtosis, skew
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, concatenate
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import scipy



# Define directories
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

def find_max_n_mels(sr, n_fft, fmax):
    n_mels = 20
    while True:
        try:
            # Generate a mel filter bank
            librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax)
            n_mels += 1
            print(f"n_mels: {n_mels}")
        except:
            # If an exception is raised, the previous n_mels is the maximum valid value
            break
    return n_mels - 1


# Sampling rate (sr): Number of samples per second. In this case, sr=192000.
# Frame length (n_fft): The length of the FFT window, which is 2048 samples in this case.
# Hop length (hop_length): The number of samples between successive frames, which is 512 samples in this case
# Number of Time Frames = [(Total Number of Samples − Frame Length)/Hop Length​⌉ + 1 = [(1*192000 - 2048)/512] + 1 = 372

def extract_audio_features(file_path, sr=192000, fft_points=2048):
    y, sr = librosa.load(file_path, sr=sr)

    # Parameters for MFCC computation
    hop_length = fft_points // 4  # or adjust as needed
    n_mels = 120  # Adjust based on your signal characteristics
    fmax = sr / 2  # Set fmax to Nyquist frequency (96 kHz)
    
        
    # FFT between 20kHz and 96kHz
    fft_result = np.abs(fft(y, n=fft_points))
    freqs = np.fft.fftfreq(fft_points, 1/sr)
    fft_20k_96k = fft_result[(freqs >= 20000) & (freqs <= 96000)]
    
    # Ensure we only take positive frequencies
    fft_20k_96k = fft_20k_96k[:fft_points // 2]

    #max_n_mels = find_max_n_mels(sr, fft_points, fmax)
    #print(f"Maximum n_mels without empty responses: {max_n_mels}")
    
    # MFE Log
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=fft_points, hop_length=hop_length, 
                                                     window=scipy.signal.get_window('hamming', fft_points), htk=False, 
                                                     center=True, pad_mode='reflect', power=2.0, 
                                                     n_mels=n_mels, fmax=fmax)
    
    mfe_log = librosa.power_to_db(mel_spec, ref=np.max)
    
    mfccs = librosa.feature.mfcc(S=mfe_log, sr=sr, n_mfcc=40)

    # Combine features
    combined_features = np.hstack([
        mfccs.T.flatten(),
        fft_20k_96k.flatten(),
        mfe_log.flatten()
    ])
    
    return combined_features

def extract_accel_features(file_path):
    df = pd.read_csv(file_path)
    features = []
    for column in df.columns[1:]:  # skip timestamp
        data = df[column].values
        
        # Statistical features
        features.append(np.mean(data))
        features.append(np.std(data))
        features.append(np.sqrt(np.mean(data**2)))  # RMS
        features.append(kurtosis(data))
        features.append(skew(data))
        
        # Spectral features
        spectrum = np.abs(fft(data))[:len(data) // 2]
        features.append(np.mean(spectrum))
        features.append(np.std(spectrum))
        features.append(np.sqrt(np.mean(spectrum**2)))  # RMS of spectrum
        
    return features

# Extract features for each file and combine them
audio_features = []
accel_features = []
labels = []

# Process TIJOLEIRA files
for audio_file, accel_file in zip(tijoleira_files_audio, tijoleira_files_acel):
    audio_features.append(extract_audio_features(audio_file))
    accel_features.append(extract_accel_features(accel_file))
    labels.append(1)  # 1 for TIJOLEIRA

# Process LISO files
for audio_file, accel_file in zip(liso_files_audio, liso_files_acel):
    audio_features.append(extract_audio_features(audio_file))
    accel_features.append(extract_accel_features(accel_file))
    labels.append(0)  # 0 for LISO

# Convert lists to numpy arrays
audio_features = np.array(audio_features)
print(audio_features.shape)
accel_features = np.array(accel_features)
print(accel_features.shape)
labels = np.array(labels)

# Normalize accelerometer data
scaler = StandardScaler()
accel_features = scaler.fit_transform(accel_features)

# Split data into training and testing sets
X_audio_train, X_audio_test, X_accel_train, X_accel_test, y_train, y_test = train_test_split(audio_features, accel_features, labels, test_size=0.2, random_state=42)

# Convert labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Audio model
input_audio = Input(shape=(X_audio_train.shape[1], 1))
x_audio = Conv1D(32, kernel_size=3, activation='relu')(input_audio)
x_audio = MaxPooling1D(pool_size=2)(x_audio)
x_audio = Conv1D(64, kernel_size=3, activation='relu')(x_audio)
x_audio = MaxPooling1D(pool_size=2)(x_audio)
x_audio = Flatten()(x_audio)

# Accelerometer model
input_accel = Input(shape=(X_accel_train.shape[1], 1))
x_accel = Conv1D(32, kernel_size=3, activation='relu')(input_accel)
x_accel = MaxPooling1D(pool_size=2)(x_accel)
x_accel = Conv1D(64, kernel_size=3, activation='relu')(x_accel)
x_accel = MaxPooling1D(pool_size=2)(x_accel)
x_accel = Flatten()(x_accel)

# Combine models
combined = concatenate([x_audio, x_accel])
x = Dense(128, activation='relu')(combined)
x = Dropout(0.5)(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=[input_audio, input_accel], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Reshape audio data for CNN
X_audio_train = X_audio_train[..., np.newaxis]
X_audio_test = X_audio_test[..., np.newaxis]
X_accel_train = X_accel_train[..., np.newaxis]
X_accel_test = X_accel_test[..., np.newaxis]

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit([X_audio_train, X_accel_train], y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
y_pred = model.predict([X_audio_test, X_accel_test])
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_test_classes, y_pred_classes)
precision = precision_score(y_test_classes, y_pred_classes)
recall = recall_score(y_test_classes, y_pred_classes)
report = classification_report(y_test_classes, y_pred_classes, target_names=['LISO', 'TIJOLEIRA'])