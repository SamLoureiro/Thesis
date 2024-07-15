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
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define directories
current_dir = os.getcwd()

good_bearing_dir_audio_m = os.path.join(current_dir, 'Dataset_Bearings', 'BEARING_ONLY', 'V2', 'GOOD', 'AUDIO_1s')
damaged_bearing_dir_audio_m = os.path.join(current_dir, 'Dataset_Bearings', 'BEARING_ONLY', 'V2', 'SMALL_DAMAGE', 'AUDIO_1s')
#good_bearing_dir_audio_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'GOOD', 'AUDIO')
#damaged_bearing_dir_audio_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'DAMAGED', 'AUDIO')

# Define noise profile file
noise_profile_file = os.path.join(current_dir, 'Dataset_Bearings', 'BEARING_ONLY', 'V2', 'NOISE', 'RECORD_3.WAV')

# Helper function to sort files by the numeric part in the filename
def sort_key(file_path):
    file_name = os.path.basename(file_path)
    # Extract the numeric part from the filename for sorting
    numeric_part = ''.join(filter(str.isdigit, file_name))
    return int(numeric_part) if numeric_part else 0

# Load list of audio files for AMR_MOVEMENT
good_bearing_files_audio_m = sorted(
    [os.path.join(good_bearing_dir_audio_m, file) for file in os.listdir(good_bearing_dir_audio_m) if file.endswith('.WAV')],
    key=sort_key
)
damaged_bearing_files_audio_m = sorted(
    [os.path.join(damaged_bearing_dir_audio_m, file) for file in os.listdir(damaged_bearing_dir_audio_m) if file.endswith('.WAV')],
    key=sort_key
)

# Load list of audio files for AMR_STOPPED
'''good_bearing_files_audio_s = sorted(
    [os.path.join(good_bearing_dir_audio_s, file) for file in os.listdir(good_bearing_dir_audio_s) if file.endswith('.WAV')],
    key=sort_key
)
damaged_bearing_files_audio_s = sorted(
    [os.path.join(damaged_bearing_dir_audio_s, file) for file in os.listdir(damaged_bearing_dir_audio_s) if file.endswith('.WAV')],
    key=sort_key
)'''

# Combine audio files
good_bearing_files_audio = good_bearing_files_audio_m #+ good_bearing_files_audio_s
damaged_bearing_files_audio = damaged_bearing_files_audio_m #+ damaged_bearing_files_audio_s

#good_bearing_files_audio = sorted(good_bearing_files_audio, key=sort_key)
#damaged_bearing_files_audio = sorted(damaged_bearing_files_audio, key=sort_key)

# Define feature extraction function for audio
def extract_audio_features(file_path, noise_profile_file):
    # Load noise profile
    noise_profile, sr_noise = librosa.load(noise_profile_file, sr=192000)
    
    # Load audio file
    y, sr = librosa.load(file_path, sr=192000)
    # Apply noise reduction
    y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_profile, n_fft=2048, hop_length=512)
    
    epsilon = 1e-10

    # Extract features
    features = {
        'mean': np.mean(y),
        'std': np.std(y),
        'rms': np.sqrt(np.mean(y**2)),
        'kurtosis': kurtosis(y) if np.std(y) > epsilon else 0,
        'skew': skew(y) if np.std(y) > epsilon else 0
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

# Extract features for each file and combine them
combined_features = []
labels = []

count = 0
max_count = min(len(good_bearing_files_audio), len(damaged_bearing_files_audio))
# Process good_bearing files
for audio_file in good_bearing_files_audio:
    audio_features = extract_audio_features(audio_file, noise_profile_file)
    combined_features.append(audio_features)
    labels.append(0)  # 0 for good_bearing
    count += 1
    if count == max_count:
        break

count = 0
# Process damaged_bearing files
for audio_file in damaged_bearing_files_audio:
    audio_features = extract_audio_features(audio_file, noise_profile_file)
    combined_features.append(audio_features)
    labels.append(1)  # 1 for damaged_bearing
    count += 1
    if count == max_count:
        break

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

clf = tfdf.keras.GradientBoostedTreesModel(task=tfdf.keras.Task.CLASSIFICATION, num_trees=500, growing_strategy="BEST_FIRST_GLOBAL", max_depth=12)

clf.fit(X_train, y_train)

clf.compile(metrics=["accuracy"])

# Save the model
#model_save_path = os.path.join(current_dir, "saved_model")
#clf.save(model_save_path)

# Evaluate the model
evaluation = clf.evaluate(X_test, y_test, return_dict=True)

print("\n")

for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")

print('Test accuracy:', evaluation['accuracy'])

tfdf.model_plotter.plot_model_in_colab(clf, tree_idx=0, max_depth=3)

# Training logs
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

# Calculate additional metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Count the number of good and damaged bearings in the test set
unique, counts = np.unique(y_test, return_counts=True)
label_counts = dict(zip(unique, counts))

print(f"Number of good bearings (0) in the test set: {label_counts.get(0, 0)}")
print(f"Number of damaged bearings (1) in the test set: {label_counts.get(1, 0)}")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Classification report
print("Classification Report:")
target_names = ['GOOD', 'DAMAGED']
print(classification_report(y_test, y_pred, target_names=['DAMAGED', 'GOOD']))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
