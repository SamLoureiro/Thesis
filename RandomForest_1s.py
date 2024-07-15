import os
import numpy as np
import pandas as pd
import librosa
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
import scipy.signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle, compute_class_weight
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import noisereduce as nr

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

# Combine audio files
good_bearing_files_audio = good_bearing_files_audio_m + good_bearing_files_audio_s
damaged_bearing_files_audio = damaged_bearing_files_audio_m + damaged_bearing_files_audio_s

# Combine accelerometer files
good_bearing_files_acel = good_bearing_files_acel_m + good_bearing_files_acel_s
damaged_bearing_files_acel = damaged_bearing_files_acel_m + damaged_bearing_files_acel_s

# Sort files
good_bearing_files_audio = sorted(good_bearing_files_audio, key=sort_key)
damaged_bearing_files_audio = sorted(damaged_bearing_files_audio, key=sort_key)
good_bearing_files_acel = sorted(good_bearing_files_acel, key=sort_key)
damaged_bearing_files_acel = sorted(damaged_bearing_files_acel, key=sort_key)

# Define feature extraction functions
def extract_audio_features(file_path, noise_profile):
    y, sr = librosa.load(file_path, sr=192000)
    # Apply noise reduction
    # y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_profile, prop_decrease=0.2, n_fft=2048, hop_length=512)
    
    epsilon = 1e-10
    features = {
        'mean': np.mean(y),
        'std': np.std(y),
        'rms': np.sqrt(np.mean(y**2)),
        'kurtosis': kurtosis(y) if np.std(y) > epsilon else 0,
        'skew': skew(y) if np.std(y) > epsilon else 0
    }
    
    fft_result = np.abs(np.fft.fft(y, n=2048))
    freqs = np.fft.fftfreq(2048, 1/sr)
    fft_20k_96k = fft_result[(freqs >= 20000) & (freqs <= 96000)]
    fft_20k_96k = fft_20k_96k[:1024]
    # 96000/1024 = 93.75 Hz per bin -> 20kHz = 213.33 bins, 96kHz = 1024 bins 1024 - 214 = 811

    for i, value in enumerate(fft_20k_96k):
        features[f'fft_20k_96k_{i}'] = value
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_fft=2048, hop_length=512, 
                                 window=scipy.signal.get_window('hamming', 2048), htk=False, 
                                 center=True, pad_mode='reflect', power=2.0, 
                                 n_mels=120, fmax=sr/2)
    for i in range(mfccs.shape[0]):
        avg = np.mean(mfccs[i, :])
        std = np.std(mfccs[i, :])
        features[f'mfcc_avg_{i}'] = avg
        features[f'mfcc_std_{i}'] = std
    
    return features

def extract_accel_features(file_path):
    epsilon = 1e-10
    df = pd.read_csv(file_path)
    features = {}
    for prefix in ['accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ']:
        for side in ['l', 'r']:
            column = f'{prefix}_{side}'
            data = df[column]
            std_dev = np.std(data)
            features[f'{column}_mean'] = np.mean(data)
            features[f'{column}_std'] = std_dev
            features[f'{column}_rms'] = np.sqrt(np.mean(data**2))
            features[f'{column}_kurtosis'] = kurtosis(data) if std_dev > epsilon else 0
            features[f'{column}_skew'] = skew(data) if std_dev > epsilon else 0
    return features

# Extract features for each file and combine them
combined_features = []
labels = []

# Process good_bearing files
for audio_file, accel_file in zip(good_bearing_files_audio, good_bearing_files_acel):
    audio_features = extract_audio_features(audio_file, noise_profile_file)
    accel_features = extract_accel_features(accel_file)
    combined = {**audio_features, **accel_features}
    combined_features.append(combined)
    labels.append(1)  # 1 for good_bearing

# Process damaged_bearing files
for audio_file, accel_file in zip(damaged_bearing_files_audio, damaged_bearing_files_acel):
    audio_features = extract_audio_features(audio_file, noise_profile_file)
    accel_features = extract_accel_features(accel_file)
    combined = {**audio_features, **accel_features}
    combined_features.append(combined)
    labels.append(0)  # 0 for damaged_bearing

# Create DataFrame
combined_features_df = pd.DataFrame(combined_features)

# Normalize features
scaler = StandardScaler()
combined_features_normalized = scaler.fit_transform(combined_features_df)

# Convert labels to numpy array
y = np.array(labels)

# Shuffle the data and labels
combined_features_normalized, y = shuffle(combined_features_normalized, y, random_state=42)

# Extract features for each file and combine them
combined_features = []
labels = []

# Process good_bearing files
for audio_file, accel_file in zip(good_bearing_files_audio, good_bearing_files_acel):
    audio_features = extract_audio_features(audio_file, noise_profile_file)
    accel_features = extract_accel_features(accel_file)
    combined = {**audio_features, **accel_features}
    combined_features.append(combined)
    labels.append(1)  # 1 for good_bearing

# Process damaged_bearing files
for audio_file, accel_file in zip(damaged_bearing_files_audio, damaged_bearing_files_acel):
    audio_features = extract_audio_features(audio_file, noise_profile_file)
    accel_features = extract_accel_features(accel_file)
    combined = {**audio_features, **accel_features}
    combined_features.append(combined)
    labels.append(0)  # 0 for damaged_bearing

# Create DataFrame
combined_features_df = pd.DataFrame(combined_features)

# Normalize features
scaler = StandardScaler()
combined_features_normalized = scaler.fit_transform(combined_features_df)

# Convert labels to numpy array
y = np.array(labels)

# Shuffle the data and labels
combined_features_normalized, y = shuffle(combined_features_normalized, y, random_state=42)

# Train classifier
X_train, X_test, y_train, y_test = train_test_split(combined_features_normalized, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(class_weight='balanced')
clf.fit(X_train, y_train)

# Evaluate
scores = cross_val_score(clf, combined_features_normalized, y, cv=5)
print("Cross-validated scores:", scores)

# Print feature importances
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = combined_features_df.columns

print("Feature ranking:")
for i in range(len(importances)):
    print(f"{i + 1}. {feature_names[indices[i]]} ({importances[indices[i]]})")

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['DAMAGED', 'GOOD'])

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nClassification Report:")
print(report)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['DAMAGED', 'GOOD'], yticklabels=['DAMAGED', 'GOOD'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

