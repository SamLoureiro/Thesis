import os
import time
import warnings
import numpy as np
import pandas as pd
import librosa
import PreProc_Function as ppf
import config
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.layers import LeakyReLU
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import Callback


def find_optimal_threshold(reconstruction_error, y_true):
    thresholds = np.linspace(min(reconstruction_error), max(reconstruction_error), 100)
    best_threshold = 0
    best_f1 = 0
    for threshold in thresholds:
        y_pred = (reconstruction_error > threshold).astype(int)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


# Custom callback to track loss and accuracy
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        # Calculate accuracy as the proportion of correctly reconstructed samples
        val_accuracy = 1 - np.mean(np.abs(self.validation_data[0] - self.validation_data[1]) > 0.5)
        self.val_accuracies.append(val_accuracy)
        print(f'\nEpoch {epoch+1} - loss: {logs.get("loss")} - val_loss: {logs.get("val_loss")} - val_accuracy: {val_accuracy}')

# Improved autoencoder architecture
def build_autoencoder_v2(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128)(input_layer)
    encoded = LeakyReLU(alpha=0.1)(encoded)
    encoded = Dense(64)(encoded)
    encoded = LeakyReLU(alpha=0.1)(encoded)
    
    decoded = Dense(64)(encoded)
    decoded = LeakyReLU(alpha=0.1)(decoded)
    decoded = Dense(128)(decoded)
    decoded = LeakyReLU(alpha=0.1)(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

# Dimensionality reduction function
def reduce_dimensions(X, method='PCA'):
    if method == 'PCA':
        reducer = PCA(n_components=2)
    elif method == 't-SNE':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Method should be 'PCA' or 't-SNE'")
    
    X_reduced = reducer.fit_transform(X)
    return X_reduced

# Function to plot the reduced data
def plot_reduced_data(X_reduced, y, y_pred=None, title="2D Map of Samples"):
    plt.figure(figsize=(10, 8))
    
    # Scatter plot for the true labels
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                          c=y, cmap='coolwarm', alpha=0.6, edgecolors='w', s=50, label='True Label')
    
    if y_pred is not None:
        # Ensure y_pred is a 1D array
        y_pred = np.ravel(y_pred)
        # Overlay predicted anomalies
        plt.scatter(X_reduced[y_pred == 1, 0], X_reduced[y_pred == 1, 1], 
                    c='red', marker='x', s=100, label='Detected Anomalies')

    plt.colorbar(scatter, label='True Label')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(title)
    plt.legend(loc='best')
    plt.show()


# Define directories
current_dir = os.getcwd()

good_bearing_dir_audio_m = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'GOOD', 'AUDIO')
good_bearing_dir_acel_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'GOOD', 'ACEL')
good_bearing_dir_acel_m = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'GOOD', 'ACEL')
good_bearing_dir_audio_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'GOOD', 'AUDIO')

damaged_bearing_dir_audio_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'DAMAGED', 'AUDIO')
damaged_bearing_dir_acel_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'DAMAGED', 'ACEL')
damaged_bearing_dir_audio_m = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'DAMAGED', 'AUDIO')
damaged_bearing_dir_acel_m = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'DAMAGED', 'ACEL')

smooth_floor_audio = os.path.join(current_dir, 'Dataset_Piso', 'LISO', 'SAMPLES_1s', 'AUDIO')
smooth_floor_acel = os.path.join(current_dir, 'Dataset_Piso', 'LISO','SAMPLES_1s', 'ACCEL')
tiled_floor_audio = os.path.join(current_dir, 'Dataset_Piso', 'TIJOLEIRA','SAMPLES_1s', 'AUDIO')
tiled_floor_acel = os.path.join(current_dir, 'Dataset_Piso', 'TIJOLEIRA','SAMPLES_1s', 'ACCEL')

# Define noise profile file
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

# Load list of audio and accelerometer files for smooth floor
smooth_floor_files_audio = sorted(
    [os.path.join(smooth_floor_audio, file) for file in os.listdir(smooth_floor_audio) if file.endswith('.WAV')],
    key=sort_key
)
smooth_floor_files_acel = sorted(
    [os.path.join(smooth_floor_acel, file) for file in os.listdir(smooth_floor_acel) if file.endswith('.csv')],
    key=sort_key
)

# Load list of audio and accelerometer files for tiled floor
tiled_floor_files_audio = sorted(
    [os.path.join(tiled_floor_audio, file) for file in os.listdir(tiled_floor_audio) if file.endswith('.WAV')],
    key=sort_key
)
tiled_floor_files_acel = sorted(
    [os.path.join(tiled_floor_acel, file) for file in os.listdir(tiled_floor_acel) if file.endswith('.csv')],
    key=sort_key
)

# Failure Dataset
failures_audio = damaged_bearing_files_audio_s + damaged_bearing_files_audio_m
faiures_accel = damaged_bearing_files_acel_s + damaged_bearing_files_acel_m

# Regular Dataset
regular_audio = good_bearing_files_audio_s + good_bearing_files_audio_m + smooth_floor_files_audio + tiled_floor_files_audio
regular_accel = good_bearing_files_acel_s + good_bearing_files_acel_m + smooth_floor_files_acel + tiled_floor_files_acel


# Extract features for each file and combine them
combined_features = []
labels = []

# Create a string based on the methods that are on
methods_string = "_".join(method for method, value in config.preprocessing_options.items() if value)

start_time_pre_proc = time.time()

# Process good_bearing files
for audio_file, accel_file in zip(regular_audio, regular_accel):
    audio_features = ppf.extract_audio_features(audio_file, noise_profile_file, config.preprocessing_options)
    accel_features = ppf.extract_accel_features(accel_file)
    combined = {**audio_features, **accel_features}
    combined_features.append(combined)
    labels.append(0)  # 0 for good_bearing


n_samples_healthy = len(combined_features)
print(f"Number of samples (Healthy Bearing): {n_samples_healthy}")

count = 0
# Process damaged_bearing files
for audio_file, accel_file in zip(failures_audio, faiures_accel):
    audio_features = ppf.extract_audio_features(audio_file, noise_profile_file, config.preprocessing_options)
    accel_features = ppf.extract_accel_features(accel_file)
    combined = {**audio_features, **accel_features}
    combined_features.append(combined)
    labels.append(1)  # 1 for damaged_bearing

end_time_pre_proc = time.time()
    
n_samples_damaged = len(combined_features) - n_samples_healthy
print(f"Number of samples (Damaged Bearing): {n_samples_damaged}")

# Create DataFrame
combined_features_df = pd.DataFrame(combined_features)

# Normalize features
scaler = StandardScaler()
combined_features_normalized = scaler.fit_transform(combined_features_df)

# Convert labels to numpy array
y = np.array(labels)

# Shuffle the data and labels
combined_features_normalized, y = shuffle(combined_features_normalized, y, random_state=42)

# Split data into training (normal) and test sets
X_train, X_test, y_train, y_test = train_test_split(combined_features_normalized, y, test_size=0.2, random_state=42)

# Build and train the autoencoder
input_dim = X_train.shape[1]
autoencoder, encoder = build_autoencoder_v2(input_dim)

# Create the LossHistory callback
#loss_history = LossHistory()

history = autoencoder.fit(X_train, X_train,
                 epochs=150,
                 batch_size=64,
                 validation_data=(X_test, X_test),
                 verbose=1)

history_dict = history.history

loss_value = history_dict['loss'][-1]
val_loss_value = history_dict['val_loss'][-1]

autoencoder.compile(loss='binary_crossentropy', metrics=["accuracy"])

# Predict reconstruction
X_test_pred = autoencoder.predict(combined_features_normalized)

# Calculate reconstruction error
reconstruction_error = np.mean(np.square(combined_features_normalized - X_test_pred), axis=1)

# Standardize reconstruction errors
scaler = StandardScaler()

# Compute reconstruction error on the test set
reconstruction_error = np.mean(np.square(combined_features_normalized - X_test_pred), axis=1)

# Find optimal threshold
optimal_threshold = find_optimal_threshold(reconstruction_error, y)

print(f"Optimal Threshold: {optimal_threshold}")

# Apply the optimal threshold
y_pred = (reconstruction_error > optimal_threshold).astype(int)

# Compute confusion matrix
cm = confusion_matrix(y, y_pred)

# Calculate Precision, Recall, F1 Score, and ROC-AUC
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
roc_auc = roc_auc_score(y, reconstruction_error)
accuracy = accuracy_score(y, y_pred)

# Print the metrics
print(f"Train Loss: {loss_value:.2f}")
print(f"Train Validation Loss: {val_loss_value:.2f}")
print("\n")
print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC-AUC: {roc_auc:.2f}")

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Reduce dimensions using t-SNE for better visualization
X_reduced = reduce_dimensions(combined_features_normalized, method='t-SNE')

# Plot the true labels
plot_reduced_data(X_reduced, y)

# Plot with detected anomalies
plot_reduced_data(X_reduced, y, y_pred)


thresholds = np.linspace(min(reconstruction_error), max(reconstruction_error), 100)
f1_scores = []
precisions = []
recalls = []
roc_aucs = []

for threshold in thresholds:
    y_pred = (reconstruction_error > threshold).astype(int)
    f1_scores.append(f1_score(y, y_pred))
    precisions.append(precision_score(y, y_pred))
    recalls.append(recall_score(y, y_pred))
    roc_aucs.append(roc_auc_score(y, reconstruction_error))

plt.figure(figsize=(12, 8))
plt.plot(thresholds, f1_scores, label='F1 Score')
plt.plot(thresholds, precisions, label='Precision')
plt.plot(thresholds, recalls, label='Recall')
plt.plot(thresholds, roc_aucs, label='ROC-AUC')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Metrics vs. Threshold')
plt.legend()
plt.grid(True)
plt.show()