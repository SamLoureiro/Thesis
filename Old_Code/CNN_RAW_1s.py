import os
import numpy as np
import pandas as pd
import librosa
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, concatenate
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


# Define directories
current_dir = os.getcwd()

good_bearing_dir_audio_m = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'DAMAGED', 'AUDIO')
damaged_bearing_dir_audio_m = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'GOOD', 'AUDIO')
good_bearing_dir_acel_m = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'DAMAGED', 'ACEL')
damaged_bearing_dir_acel_m = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'GOOD', 'ACEL')

good_bearing_dir_audio_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'DAMAGED', 'AUDIO')
damaged_bearing_dir_audio_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'GOOD', 'AUDIO')
good_bearing_dir_acel_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'DAMAGED', 'ACEL')
damaged_bearing_dir_acel_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'GOOD', 'ACEL')

# Define noise profile file
noise_profile_file = os.path.join(current_dir, 'Dataset_Piso', 'Noise.WAV')

# Helper function to sort files by the numeric part in the filename
def sort_key(file_path):
    file_name = os.path.basename(file_path)
    # Extract the numeric part from the filename for sorting
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

# Combine audio files
good_bearing_files_audio = good_bearing_files_audio_m + good_bearing_files_audio_s
damaged_bearing_files_audio = damaged_bearing_files_audio_m + damaged_bearing_files_audio_s

# Combine accelerometer files
good_bearing_files_acel = good_bearing_files_acel_m + good_bearing_files_acel_s
damaged_bearing_files_acel = damaged_bearing_files_acel_m + damaged_bearing_files_acel_s

good_bearing_files_audio = sorted(good_bearing_files_audio, key=sort_key)
damaged_bearing_files_audio = sorted(damaged_bearing_files_audio, key=sort_key)
good_bearing_files_acel = sorted(good_bearing_files_acel, key=sort_key)
damaged_bearing_files_acel = sorted(damaged_bearing_files_acel, key=sort_key)


# Define feature extraction functions
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=192000)
    return y

def extract_accel_features(file_path):
    df = pd.read_csv(file_path)
    data = df[['accX_l', 'accY_l', 'accZ_l', 'gyrX_l', 'gyrY_l', 'gyrZ_l', 'accX_r', 'accY_r', 'accZ_r', 'gyrX_r', 'gyrY_r', 'gyrZ_r']].values
    return data

# Extract features for each file and combine them
audio_features = []
accel_features = []
labels = []

# Process good_bearing files
for audio_file, accel_file in zip(good_bearing_files_audio, good_bearing_files_acel):
    audio_features.append(extract_audio_features(audio_file))
    accel_features.append(extract_accel_features(accel_file))
    labels.append(1)  # 1 for good_bearing

# Process damaged_bearing files
for audio_file, accel_file in zip(damaged_bearing_files_audio, damaged_bearing_files_acel):
    audio_features.append(extract_audio_features(audio_file))
    accel_features.append(extract_accel_features(accel_file))
    labels.append(0)  # 0 for damaged_bearing

# Pad audio sequences
max_audio_length = max(len(a) for a in audio_features)
audio_features = [np.pad(a, (0, max_audio_length - len(a))) for a in audio_features]

for i in range(len(accel_features)):
    print(len(accel_features[i]))

# Convert lists to numpy arrays
audio_features = np.array(audio_features)
accel_features = np.array(accel_features)
labels = np.array(labels)

# Normalize accelerometer data
scaler = StandardScaler()
accel_features = scaler.fit_transform(accel_features.reshape(-1, accel_features.shape[-1])).reshape(accel_features.shape)

# Split data into training and testing sets
X_audio_train, X_audio_test, X_accel_train, X_accel_test, y_train, y_test = train_test_split(audio_features, accel_features, labels, test_size=0.2, random_state=42)

# Convert labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Audio model
input_audio = Input(shape=(max_audio_length, 1))
x_audio = Conv1D(32, kernel_size=3, activation='relu')(input_audio)
x_audio = MaxPooling1D(pool_size=2)(x_audio)
x_audio = Conv1D(64, kernel_size=3, activation='relu')(x_audio)
x_audio = MaxPooling1D(pool_size=2)(x_audio)
x_audio = Flatten()(x_audio)

# Accelerometer model
input_accel = Input(shape=(accel_features.shape[1], accel_features.shape[2]))
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
report = classification_report(y_test_classes, y_pred_classes, target_names=['damaged_bearing', 'good_bearing'])

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nClassification Report:")
print(report)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['damaged_bearing', 'good_bearing'], yticklabels=['damaged_bearing', 'good_bearing'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
# End of CNN_Merged_1s.py