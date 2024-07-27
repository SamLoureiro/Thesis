import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
import scipy
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and preprocess audio files
def preprocess_audio(file_path, sr=192000, n_mfcc=40, max_pad_len=3500):
    wave, sample_rate = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=wave, sr=sample_rate, n_mfcc=n_mfcc, n_fft=2048, hop_length=512, 
                                window=scipy.signal.get_window('hamming', 2048), center=True, 
                                pad_mode='reflect', power=2.0, n_mels=128, fmin=200, fmax=None, 
                                htk=False, norm='ortho', lifter=0)
    if mfcc.shape[1] > max_pad_len:
        mfcc = mfcc[:, :max_pad_len]
    else:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

# Directory paths
current_dir = os.getcwd()
tijoleira_dir = os.path.join(current_dir, 'Dataset_Piso', 'TIJOLEIRA', 'SAMPLES', 'AUDIO')
liso_dir = os.path.join(current_dir, 'Dataset_Piso', 'LISO', 'SAMPLES', 'AUDIO')

# Load data
tijoleira_files = [os.path.join(tijoleira_dir, file) for file in os.listdir(tijoleira_dir)]
liso_files = [os.path.join(liso_dir, file) for file in os.listdir(liso_dir)]

X = []
y = []

# Process Tijoleira files
for file in tijoleira_files:
    if file.endswith('.WAV'):
        print('Processing:', file)
        X.append(preprocess_audio(file))
        y.append(0)  # Label for TIJOLEIRA

# Process Liso files
for file in liso_files:
    if file.endswith('.WAV'):
        X.append(preprocess_audio(file))
        y.append(1)  # Label for LISO

X = np.array(X)
y = np.array(y)

# Shuffle the data
X, y = shuffle(X, y, random_state=42)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('It reached here!')

# Add a channel dimension for CNN input
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# Define the CNN model
def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create the model
input_shape = (X_train.shape[1], X_train.shape[2], 1)
model = create_model(input_shape)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('audio_classification_model.h5')

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print('Test accuracy:', test_accuracy)

# Generate predictions
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype("int32")

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
print('Confusion Matrix:\n', conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['TIJOLEIRA', 'LISO'], yticklabels=['TIJOLEIRA', 'LISO'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
class_report = classification_report(y_test, y_pred_classes, target_names=['TIJOLEIRA', 'LISO'])
print('Classification Report:\n', class_report)
