import os
import numpy as np
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# Function to load and preprocess audio files with MFCC and Log-MFE
def preprocess_mfcc(file_path, sr=192000, n_mfcc=128, max_pad_len=3737):
    # Load the audio file
    wave, sample_rate = librosa.load(file_path, sr=sr)

    # Parameters
    n_mels = 128  # Starting with 128 Mel filters
    fmax = sr / 2  # Set fmax to Nyquist frequency (96 kHz)

    # Compute Mel filter bank
    mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=n_mels, fmax=fmax)

    # Check for empty filters
    empty_filters = np.sum(mel_filter_bank, axis=1) == 0
    if np.any(empty_filters):
        print(f"Warning: {np.sum(empty_filters)} empty filters detected. Consider adjusting n_mels or fmax.")

    # Adjust n_mels if needed
    while np.any(empty_filters):
        n_mels -= 1  # Decrease n_mels
        mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=n_mels, fmax=fmax)
        empty_filters = np.sum(mel_filter_bank, axis=1) == 0

    print(f"Number of Mel filters: {n_mels}")

    # Generate Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=wave, sr=sample_rate, n_fft=2048, hop_length=512, 
                                                     window=scipy.signal.get_window('hamming', 2048), htk=False, 
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

# Directory paths
current_dir = os.getcwd()
tijoleira_dir = os.path.join(current_dir, 'Dataset_Piso', 'TIJOLEIRA', 'SAMPLES', 'AUDIO')
liso_dir = os.path.join(current_dir, 'Dataset_Piso', 'LISO', 'SAMPLES', 'AUDIO')

# Load data
tijoleira_files = [os.path.join(tijoleira_dir, file) for file in os.listdir(tijoleira_dir) if file.endswith('.WAV')]
liso_files = [os.path.join(liso_dir, file) for file in os.listdir(liso_dir) if file.endswith('.WAV')]

X_mfcc = []
X_log_mfe = []
y = []

# Process Tijoleira files
for file in tijoleira_files:
    print('Processing:', file)
    mfcc, log_mfe = preprocess_mfcc(file)
    X_mfcc.append(mfcc)
    X_log_mfe.append(log_mfe)
    y.append(0)  # Label for TIJOLEIRA

# Process Liso files
for file in liso_files:
    print('Processing:', file)
    mfcc, log_mfe = preprocess_mfcc(file)
    X_mfcc.append(mfcc)
    X_log_mfe.append(log_mfe)
    y.append(1)  # Label for LISO

# Convert lists to numpy arrays
X_mfcc = np.array(X_mfcc)
X_log_mfe = np.array(X_log_mfe)
y = np.array(y)

print("MFCC shape:", X_mfcc.shape)
print("Log-MFE shape:", X_log_mfe.shape)

# MFCC shape: (98, 114, 3737)
# Log-MFE shape: (98, 114, 3737)

# Number of samples: 98 audio files.
# Feature dimension: 114 coefficients/features per frame (both MFCC and log-MFE).
# Time dimension: 3737 frames per audio sample, ensuring consistency in input size for the neural network.

# Reshape data for CNN input
X_mfcc = X_mfcc[..., np.newaxis]  # Adding a channel dimension
X_log_mfe = X_log_mfe[..., np.newaxis]  # Adding a channel dimension

# One-hot encode the labels
y = to_categorical(y, num_classes=2)

# Split the data into training and testing sets
X_train_mfcc, X_test_mfcc, y_train, y_test = train_test_split(X_mfcc, y, test_size=0.2, random_state=42, stratify=y)
X_train_log_mfe, X_test_log_mfe, _, _ = train_test_split(X_log_mfe, y, test_size=0.2, random_state=42, stratify=y)

# Build the CNN model
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train the model on MFCC data
input_shape = X_train_mfcc.shape[1:]
model_mfcc = build_model(input_shape)

history_mfcc = model_mfcc.fit(X_train_mfcc, y_train, epochs=20, batch_size=32, validation_data=(X_test_mfcc, y_test))

# Evaluate the model
print("Evaluating on MFCC data...")
test_loss_mfcc, test_acc_mfcc = model_mfcc.evaluate(X_test_mfcc, y_test)
print(f"Test accuracy (MFCC): {test_acc_mfcc:.4f}")

# Build and train the model on Log-MFE data
input_shape = X_train_log_mfe.shape[1:]
model_log_mfe = build_model(input_shape)

history_log_mfe = model_log_mfe.fit(X_train_log_mfe, y_train, epochs=20, batch_size=32, validation_data=(X_test_log_mfe, y_test))

# Evaluate the model
print("Evaluating on Log-MFE data...")
test_loss_log_mfe, test_acc_log_mfe = model_log_mfe.evaluate(X_test_log_mfe, y_test)
print(f"Test accuracy (Log-MFE): {test_acc_log_mfe:.4f}")

# Plotting training history for MFCC and Log-MFE
def plot_training_history(history, title):
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{title} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{title} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.tight_layout()
    plt.show()

plot_training_history(history_mfcc, 'MFCC')
plot_training_history(history_log_mfe, 'Log-MFE')
