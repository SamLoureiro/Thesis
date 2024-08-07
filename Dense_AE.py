import os
import time
import numpy as np
import pandas as pd
import librosa
import config
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, 
                             confusion_matrix, accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.layers import LeakyReLU, Input, Dense, BatchNormalization, Dropout
from keras.models import Model
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import seaborn as sns
import PreProc_Function as ppf


# Function to find the optimal threshold for reconstruction error
def find_optimal_threshold(reconstruction_error, y_true):
    thresholds = np.linspace(min(reconstruction_error), max(reconstruction_error), 100)
    best_threshold = 0
    best_f1 = 0
    for threshold in thresholds:
        y_pred = (reconstruction_error > threshold).astype(int)
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
        val_accuracy = 1 - np.mean(np.abs(self.validation_data[0] - self.validation_data[1]) > 0.5)
        self.val_accuracies.append(val_accuracy)
        print(f'\nEpoch {epoch+1} - loss: {logs.get("loss")} - val_loss: {logs.get("val_loss")} - val_accuracy: {val_accuracy}')


# Improved autoencoder architecture
def build_autoencoder_v2(input_dim):
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(256, activation='relu')(input_layer)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.3)(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.3)(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    
    # Bottleneck
    bottleneck = Dense(32, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(64, activation='relu')(bottleneck)
    decoded = BatchNormalization()(decoded)
    decoded = Dropout(0.3)(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dropout(0.3)(decoded)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, bottleneck)
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
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                          c=y, cmap='coolwarm', alpha=0.6, edgecolors='w', s=50, label='True Label')
    if y_pred is not None:
        y_pred = np.ravel(y_pred)
        plt.scatter(X_reduced[y_pred == 1, 0], X_reduced[y_pred == 1, 1], 
                    c='red', marker='x', s=100, label='Detected Anomalies')
    plt.colorbar(scatter, label='True Label')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(title)
    plt.legend(loc='best')
    plt.show()


# Define directories and file paths
def define_directories():
    current_dir = os.getcwd()
    directories = {
        'good_bearing': {
            'audio_m': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'GOOD', 'AUDIO'),
            'acel_m': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'GOOD', 'ACEL'),
            'audio_s': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'GOOD', 'AUDIO'),
            'acel_s': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'GOOD', 'ACEL'),
        },
        'damaged_bearing': {
            'audio_s': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'DAMAGED', 'AUDIO'),
            'acel_s': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'DAMAGED', 'ACEL'),
            'audio_m': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'DAMAGED', 'AUDIO'),
            'acel_m': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'DAMAGED', 'ACEL'),
        },
        'smooth_floor': {
            'audio': os.path.join(current_dir, 'Dataset_Piso', 'LISO', 'SAMPLES_1s', 'AUDIO'),
            'acel': os.path.join(current_dir, 'Dataset_Piso', 'LISO', 'SAMPLES_1s', 'ACCEL'),
        },
        'tiled_floor': {
            'audio': os.path.join(current_dir, 'Dataset_Piso', 'TIJOLEIRA', 'SAMPLES_1s', 'AUDIO'),
            'acel': os.path.join(current_dir, 'Dataset_Piso', 'TIJOLEIRA', 'SAMPLES_1s', 'ACCEL'),
        },
        'noise_profile': os.path.join(current_dir, 'Dataset_Piso', 'Noise.WAV')
    }
    return directories


# Helper function to sort files by numeric part in the filename
def sort_key(file_path):
    file_name = os.path.basename(file_path)
    numeric_part = ''.join(filter(str.isdigit, file_name))
    return int(numeric_part) if numeric_part else 0


# Load files and extract features
def load_and_extract_features(directories):
    combined_features = []
    labels = []

    methods_string = "_".join(method for method, value in config.preprocessing_options.items() if value)
    start_time_pre_proc = time.time()

    # Process regular bearings
    for audio_file, accel_file in zip(
        sorted([os.path.join(directories['good_bearing']['audio_s'], file) for file in os.listdir(directories['good_bearing']['audio_s']) if file.endswith('.WAV')], key=sort_key) +
        sorted([os.path.join(directories['good_bearing']['audio_m'], file) for file in os.listdir(directories['good_bearing']['audio_m']) if file.endswith('.WAV')], key=sort_key),
        sorted([os.path.join(directories['good_bearing']['acel_s'], file) for file in os.listdir(directories['good_bearing']['acel_s']) if file.endswith('.csv')], key=sort_key) +
        sorted([os.path.join(directories['good_bearing']['acel_m'], file) for file in os.listdir(directories['good_bearing']['acel_m']) if file.endswith('.csv')], key=sort_key)
    ):
        audio_features = ppf.extract_audio_features(audio_file, directories['noise_profile'], config.preprocessing_options)
        accel_features = ppf.extract_accel_features(accel_file)
        combined = {**audio_features, **accel_features}
        combined_features.append(combined)
        labels.append(0)  # 0 for good bearing

    n_samples_healthy = len(combined_features)
    print(f"Number of samples (Healthy Bearing): {n_samples_healthy}")

    # Process damaged bearings
    for audio_file, accel_file in zip(
        sorted([os.path.join(directories['damaged_bearing']['audio_s'], file) for file in os.listdir(directories['damaged_bearing']['audio_s']) if file.endswith('.WAV')], key=sort_key) +
        sorted([os.path.join(directories['damaged_bearing']['audio_m'], file) for file in os.listdir(directories['damaged_bearing']['audio_m']) if file.endswith('.WAV')], key=sort_key),
        sorted([os.path.join(directories['damaged_bearing']['acel_s'], file) for file in os.listdir(directories['damaged_bearing']['acel_s']) if file.endswith('.csv')], key=sort_key) +
        sorted([os.path.join(directories['damaged_bearing']['acel_m'], file) for file in os.listdir(directories['damaged_bearing']['acel_m']) if file.endswith('.csv')], key=sort_key)
    ):
        audio_features = ppf.extract_audio_features(audio_file, directories['noise_profile'], config.preprocessing_options)
        accel_features = ppf.extract_accel_features(accel_file)
        combined = {**audio_features, **accel_features}
        combined_features.append(combined)
        labels.append(1)  # 1 for damaged bearing

    end_time_pre_proc = time.time()
    n_samples_damaged = len(combined_features) - n_samples_healthy
    print(f"Number of samples (Damaged Bearing): {n_samples_damaged}")

    return pd.DataFrame(combined_features), np.array(labels), end_time_pre_proc - start_time_pre_proc


# Main execution
directories = define_directories()
combined_features_df, labels, pre_proc_time = load_and_extract_features(directories)

# Normalize features
scaler = StandardScaler()
combined_features_normalized = scaler.fit_transform(combined_features_df)

# Shuffle the data and labels
combined_features_normalized, labels = shuffle(combined_features_normalized, labels, random_state=42)

print(f"Preprocessing Time: {pre_proc_time:.2f} seconds")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(combined_features_normalized, labels, test_size=0.2, random_state=42)

# Build and train the autoencoder
input_dim = X_train.shape[1]
autoencoder, encoder = build_autoencoder_v2(input_dim)
history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Evaluate the model
history_dict = history.history
loss_value = history_dict['loss'][-1]
val_loss_value = history_dict['val_loss'][-1]

# Reconstruction and threshold finding
X_train_pred = autoencoder.predict(X_train)
X_test_pred = autoencoder.predict(X_test)

train_reconstruction_error = np.mean(np.abs(X_train - X_train_pred), axis=1)
test_reconstruction_error = np.mean(np.abs(X_test - X_test_pred), axis=1)

optimal_threshold = find_optimal_threshold(train_reconstruction_error, y_train)

y_test_pred = (test_reconstruction_error > optimal_threshold).astype(int)

# Evaluation
print("Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred)}")
print(f"Precision: {precision_score(y_test, y_test_pred)}")
print(f"Recall: {recall_score(y_test, y_test_pred)}")
print(f"F1 Score: {f1_score(y_test, y_test_pred)}")
print(f"AUC: {roc_auc_score(y_test, test_reconstruction_error)}")
print(f"Optimal Threshold: {optimal_threshold}")


##################################################
#F1-Score Results:
#STFT: 0.769   (0.775 AUC)
#MFCC: Detect everything as anomaly, need another method to find the optimal threshold. However, the model is not better than STFT based on the Metrics vs. Threshold' figure 
#FFT: Detect everything as anomaly, need another method to find the optimal threshold. However, the model is not better than STFT based on the Metrics vs. Threshold' figure 
#SP (basics): Detect everything as anomaly, need another method to find the optimal threshold. However, the model is not better than STFT based on the Metrics vs. Threshold' figure 
#SP + STFT: 0.777 (0.780 AUC)
#SP+FFT+MDCC+STFT: Detect everything as anomaly, need another method to find the optimal threshold. However, the model is not better than STFT based on the Metrics vs. Threshold' figure 
#SP+STFT+NR: 0.771

cm = confusion_matrix(y_test, y_test_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Reduce dimensions and plot
X_reduced = reduce_dimensions(X_test, method='t-SNE')
plot_reduced_data(X_reduced, y_test)
plot_reduced_data(X_reduced, y_test, y_test_pred)

# Thresholds vs metrics
thresholds = np.linspace(min(test_reconstruction_error), max(test_reconstruction_error), 100)
f1_scores_test = []
precisions_test = []
recalls_test = []
roc_aucs_test = []
f1_scores_train= []
precision_train = []
recalls_train = []
roc_aucs_train = []


for threshold in thresholds:
    y_test_pred = (test_reconstruction_error > threshold).astype(int)
    f1_scores_test.append(f1_score(y_test, y_test_pred))
    precisions_test.append(precision_score(y_test, y_test_pred, zero_division=0))
    recalls_test.append(recall_score(y_test, y_test_pred))
    roc_aucs_test.append(roc_auc_score(y_test, test_reconstruction_error))
    y_train_pred = (train_reconstruction_error > threshold).astype(int)
    f1_scores_train.append(f1_score(y_train, y_train_pred))
    precision_train.append(precision_score(y_train, y_train_pred, zero_division=0))
    recalls_train.append(recall_score(y_train, y_train_pred))
    roc_aucs_train.append(roc_auc_score(y_train, train_reconstruction_error))

plt.figure(figsize=(12, 8))
plt.plot(thresholds, f1_scores_test, label='F1 Score_Test')
plt.plot(thresholds, precisions_test, label='Precision_Test')
plt.plot(thresholds, recalls_test, label='Recall_Test')
plt.plot(thresholds, roc_aucs_test, label='ROC-AUC_Test')
plt.plot(thresholds, f1_scores_train, label='F1 Score_Train')
plt.plot(thresholds, precision_train, label='Precision_Train')
plt.plot(thresholds, recalls_train, label='Recall_Train')
plt.plot(thresholds, roc_aucs_train, label='ROC-AUC_Train')

plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Metrics vs. Threshold')
plt.legend()
plt.grid(True)
plt.show()
