import os
import numpy as np
import pandas as pd
import librosa
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


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

# Define feature extraction functions
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=192000)
    features = {
        'mean': np.mean(y),
        'std': np.std(y),
        'rms': np.sqrt(np.mean(y**2)),
        'kurtosis': kurtosis(y) if np.std(y) != 0 else 0,
        'skew': skew(y) if np.std(y) != 0 else 0
    }
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

# Process TIJOLEIRA files
for audio_file, accel_file in zip(tijoleira_files_audio, tijoleira_files_acel):
    audio_features = extract_audio_features(audio_file)
    accel_features = extract_accel_features(accel_file)
    combined = {**audio_features, **accel_features}
    combined_features.append(combined)
    labels.append(1)  # 1 for TIJOLEIRA

# Process LISO files
for audio_file, accel_file in zip(liso_files_audio, liso_files_acel):
    audio_features = extract_audio_features(audio_file)
    accel_features = extract_accel_features(accel_file)
    combined = {**audio_features, **accel_features}
    combined_features.append(combined)
    labels.append(0)  # 0 for LISO

# Create DataFrame
combined_features_df = pd.DataFrame(combined_features)

# Normalize features
scaler = StandardScaler()
combined_features_normalized = scaler.fit_transform(combined_features_df)

# Convert labels to numpy array
y = np.array(labels)

# Train classifier
X_train, X_test, y_train, y_test = train_test_split(combined_features_normalized, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
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
report = classification_report(y_test, y_pred, target_names=['LISO', 'TIJOLEIRA'])

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
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

