import os
import numpy as np
import IPython
import pandas as pd
import librosa
import noisereduce as nr
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
import scipy.signal
import tensorflow_decision_forests as tfdf
from sklearn.utils import shuffle, compute_class_weight
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV

# Define directories
current_dir = os.getcwd()
tijoleira_dir_audio = os.path.join(current_dir, 'Dataset_Piso', 'TIJOLEIRA', 'SAMPLES_1s', 'AUDIO')
liso_dir_audio = os.path.join(current_dir, 'Dataset_Piso', 'LISO', 'SAMPLES_1s', 'AUDIO')
tijoleira_dir_acel = os.path.join(current_dir, 'Dataset_Piso', 'TIJOLEIRA', 'SAMPLES_1s', 'ACCEL')
liso_dir_acel = os.path.join(current_dir, 'Dataset_Piso', 'LISO', 'SAMPLES_1s', 'ACCEL')
noise_profile_file = os.path.join(current_dir, 'Dataset_Piso', 'Noise.WAV')

# Helper function to sort files by the numeric part in the filename
def sort_key(file_path):
    file_name = os.path.basename(file_path)
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

# Define feature extraction functions
def extract_audio_features(file_path, noise_profile):
    # Load audio file
    y, sr = librosa.load(file_path, sr=192000)
    # Apply noise reduction
    y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_profile, n_fft=2048, hop_length=512)

    features = {
        'mean': np.mean(y),
        'std': np.std(y),
        'rms': np.sqrt(np.mean(y**2)),
        'kurtosis': kurtosis(y) if np.std(y) != 0 else 0,
        'skew': skew(y) if np.std(y) != 0 else 0
    }
    
    # FFT between 20kHz and 96kHz
    fft_result = np.abs(np.fft.fft(y, n=2048))
    freqs = np.fft.fftfreq(2048, 1/sr)
    fft_20k_96k = fft_result[(freqs >= 20000) & (freqs <= 96000)]
    fft_20k_96k = fft_20k_96k[:1024]  # Take the positive half of the FFT

    # Add FFT features
    for i, value in enumerate(fft_20k_96k):
        features[f'fft_20k_96k_{i}'] = value
    
    # MFCCs
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
    df = pd.read_csv(file_path)
    features = {}
    for prefix in ['accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ']:
        for side in ['l', 'r']:
            column = f'{prefix}_{side}'
            data = df[column]
            features[f'{column}_mean'] = np.mean(data)
            features[f'{column}_std'] = np.std(data)
            features[f'{column}_rms'] = np.sqrt(np.mean(data**2))
            features[f'{column}_kurtosis'] = kurtosis(data) if np.std(data) != 0 else 0
            features[f'{column}_skew'] = skew(data) if np.std(data) != 0 else 0
    return features

# Extract features for each file and combine them
combined_features = []
labels = []

# Process dataset files with unbalanced data
for audio_file, accel_file in zip(tijoleira_files_audio, tijoleira_files_acel):
    audio_features = extract_audio_features(audio_file, noise_profile_file)
    accel_features = extract_accel_features(accel_file)
    combined = {**audio_features, **accel_features}
    combined_features.append(combined)
    labels.append(1)  # 1 for TIJOLEIRA

for audio_file, accel_file in zip(liso_files_audio, liso_files_acel):
    audio_features = extract_audio_features(audio_file, noise_profile_file)
    accel_features = extract_accel_features(accel_file)
    combined = {**audio_features, **accel_features}
    combined_features.append(combined)
    labels.append(0)  # 0 for LISO

# Process dataset with balance data
'''
count = 0
max_count = min(len(tijoleira_files_audio), len(liso_files_audio))
# Process TIJOLEIRA files
for audio_file, accel_file in zip(tijoleira_files_audio, tijoleira_files_acel):
    audio_features = extract_audio_features(audio_file)
    accel_features = extract_accel_features(accel_file)
    combined = {**audio_features, **accel_features}
    combined_features.append(combined)
    labels.append(1)  # 1 for TIJOLEIRA
    count+=1
    if(count == max_count):
        break

count = 0
# Process LISO files
for audio_file, accel_file in zip(liso_files_audio, liso_files_acel):
    audio_features = extract_audio_features(audio_file)
    accel_features = extract_accel_features(accel_file)
    combined = {**audio_features, **accel_features}
    combined_features.append(combined)
    labels.append(0)  # 0 for LISO
    count+=1
    if(count == max_count):
        break
'''

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
#clf = tfdf(class_weight='balanced')
clf = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.CLASSIFICATION)
clf.fit(X_train, y_train)

clf.compile(loss='binary_crossentropy', metrics=["accuracy"])

# Evaluate the model
evaluation = clf.evaluate(X_test, y_test, return_dict=True)

print("\n")

for name, value in evaluation.items():
  print(f"{name}: {value:.4f}")

print('Test accuracy:', evaluation)

tfdf.model_plotter.plot_model_in_colab(clf, tree_idx=0, max_depth=3)

# The input features
clf.make_inspector().features()

# The feature importances
clf.make_inspector().variable_importances()

# The training logs
clf.make_inspector().training_logs()


logs = clf.make_inspector().training_logs()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy (out-of-bag)")

plt.subplot(1, 2, 2)
plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Logloss (out-of-bag)")

plt.show()


# Predict on the test set
y_pred_probs = clf.predict(X_test)

# Convert predicted probabilities to binary class labels
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['LISO', 'TIJOLEIRA'])

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(report)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['LISO', 'TIJOLEIRA'], yticklabels=['LISO', 'TIJOLEIRA'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

