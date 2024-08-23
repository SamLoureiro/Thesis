'''
Developer Notes:

- This script used a data shape of number of samples x number of features, unlike tge Conv_AE script that used a datashape of number of samples x number of timesteps x number of features.
- The model achieved the best results using STFT features, without MFCC - (When added the other methods the results were almost the same).
- MFCC performed poorly in this case, even when joined with other pre-processing methods.
- Increasing the batch size from 32 to 128 did not improve the model performance.

'''

import os
import shutil
import time
import numpy as np
import pandas as pd
import config
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, 
                             confusion_matrix, accuracy_score, roc_curve)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, BatchNormalization, Dropout, Lambda
from keras.models import Model
from keras_tuner import HyperModel, RandomSearch, HyperParameters, BayesianOptimization
import matplotlib.pyplot as plt
import seaborn as sns
import PreProc_Function as ppf
from AE_Aux_Func import reduce_dimensions, plot_reduced_data, plot_metrics_vs_threshold, find_optimal_threshold_f1


class AutoencoderHyperModel(HyperModel):
    def __init__(self, input_dim):
        self.input_dim = input_dim

    def build(self, hp):
        if not isinstance(hp, HyperParameters):
            raise ValueError("Expected 'hp' to be an instance of 'HyperParameters'.")

        noise_factor = hp.Float('noise_factor', min_value=0.1, max_value=0.5, step=0.05)
        units_1 = hp.Int('units_1', min_value=128, max_value=512, step=32)
        dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.05)
        units_2 = hp.Int('units_2', min_value=64, max_value=256, step=32)
        dropout_2 = hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.05)
        units_3 = hp.Int('units_3', min_value=32, max_value=128, step=32)
        bottleneck_units = hp.Int('bottleneck_units', min_value=16, max_value=64, step=8)
        dropout_3 = hp.Float('dropout_3', min_value=0.1, max_value=0.5, step=0.05)
        dropout_4 = hp.Float('dropout_4', min_value=0.1, max_value=0.5, step=0.05)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

        input_layer = Input(shape=(self.input_dim,))
        noisy_inputs = Lambda(lambda x: x + noise_factor * tf.random.normal(tf.shape(x)))(input_layer)
        
        # Encoder
        encoded = Dense(units_1, activation='relu')(noisy_inputs)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(dropout_1)(encoded)
        encoded = Dense(units_2, activation='relu')(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(dropout_2)(encoded)
        encoded = Dense(units_3, activation='relu')(encoded)
        
        # Bottleneck
        bottleneck = Dense(bottleneck_units, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(units_3, activation='relu')(bottleneck)
        decoded = BatchNormalization()(decoded)
        decoded = Dropout(dropout_3)(decoded)
        decoded = Dense(units_2, activation='relu')(decoded)
        decoded = BatchNormalization()(decoded)
        decoded = Dropout(dropout_4)(decoded)
        decoded = Dense(units_1, activation='relu')(decoded)
        decoded = Dense(self.input_dim, activation='sigmoid')(decoded)
        
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                            loss='mse')        
        return autoencoder


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
        sorted([os.path.join(directories['good_bearing']['audio_m'], file) for file in os.listdir(directories['good_bearing']['audio_m']) if file.endswith('.WAV')], key=sort_key) +
        sorted([os.path.join(directories['smooth_floor']['audio'], file) for file in os.listdir(directories['smooth_floor']['audio']) if file.endswith('.WAV')], key=sort_key) +
        sorted([os.path.join(directories['tiled_floor']['audio'], file) for file in os.listdir(directories['tiled_floor']['audio']) if file.endswith('.WAV')], key=sort_key),
        sorted([os.path.join(directories['good_bearing']['acel_s'], file) for file in os.listdir(directories['good_bearing']['acel_s']) if file.endswith('.csv')], key=sort_key) +
        sorted([os.path.join(directories['good_bearing']['acel_m'], file) for file in os.listdir(directories['good_bearing']['acel_m']) if file.endswith('.csv')], key=sort_key) +
        sorted([os.path.join(directories['smooth_floor']['acel'], file) for file in os.listdir(directories['smooth_floor']['acel']) if file.endswith('.csv')], key=sort_key) +
        sorted([os.path.join(directories['tiled_floor']['acel'], file) for file in os.listdir(directories['tiled_floor']['acel']) if file.endswith('.csv')], key=sort_key)
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

    return pd.DataFrame(combined_features), np.array(labels), end_time_pre_proc - start_time_pre_proc, methods_string


def remove_unused_trials(keras_tuner_dir, project_name, best_trial):
    """Removes all trials except for the best one."""
    trials_dir = os.path.join(keras_tuner_dir, project_name)
    
    for subdir in os.listdir(trials_dir):
        trial_dir = os.path.join(trials_dir, subdir)
        
        # Only consider files/directories with an "_" in the name
        if "_" in subdir:
            trial_id = subdir.split('_')[-1]
            if trial_id != best_trial.trial_id:
                print(f"Deleting trial {trial_id}")
                shutil.rmtree(trial_dir)
            else:
                print(f"Keeping best trial {trial_id}")
        else:
            print(f"Skipping {subdir}")


# Main execution
directories = define_directories()
combined_features_df, labels, pre_proc_time, pre_proc_string = load_and_extract_features(directories)

# Normalize features
scaler = StandardScaler()
combined_features_normalized = scaler.fit_transform(combined_features_df)

#print(f"Combined Features Shape: {combined_features_normalized.shape}")

# The data shape is (number of samples x number of features)

print(f"Preprocessing Time: {pre_proc_time:.3f} seconds")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(combined_features_normalized, labels, test_size=0.2, random_state=42)

# Build and train the autoencoder
input_dim = X_train.shape[1]

# Epochs and batch size
epochs = 100
batch_size = 32

# Define directories
current_dir = os.getcwd()
tuner_dir = os.path.join(current_dir, 'keras_tuner_dir')
project_name = 'AE_BayesianOptimization_' + str(batch_size) + 'bs_' + pre_proc_string


# Define the BayesianOptimization tuner
tuner = BayesianOptimization(
    hypermodel=AutoencoderHyperModel(input_dim),
    objective='val_loss',
    max_trials=20,
    executions_per_trial=1,
    directory=tuner_dir,
    project_name=project_name
)

# Early stopping setup
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# If the tuner directory exists, try to load the best trial, otherwise perform Bayesian optimization and save and load the best trial
try:
    # Load the best trial
    best_trial = tuner.oracle.get_best_trials(1)[0]
    print(f"Best trial: {best_trial.trial_id}")
    print(f"Best trial value: {best_trial.score}")

    # Best hyperparameters
    hyperparameters = best_trial.hyperparameters
    print(f"Best trial hyperparameters: {hyperparameters.values}")

    # Build the best model using the best hyperparameters
    autoencoder = AutoencoderHyperModel(input_dim).build(hyperparameters)
except IndexError:
    # Perform Bayesian optimization
    tuner.search(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping], verbose=1)
    
    best_trial = tuner.oracle.get_best_trials(1)[0]
    print(f"Best trial: {best_trial.trial_id}")
    print(f"Best trial value: {best_trial.score}")

    # Best hyperparameters
    hyperparameters = best_trial.hyperparameters
    print(f"Best trial hyperparameters: {hyperparameters.values}")

    # Build the best model using the best hyperparameters
    autoencoder = AutoencoderHyperModel(input_dim).build(hyperparameters)
    pass
except AttributeError as e:
    print(f"Error while accessing best trial attributes: {e}")


# Reconstruction and threshold finding
X_train_pred = autoencoder.predict(X_train)
start_time = time.time()
X_test_pred = autoencoder.predict(X_test)
inference_time = time.time() - start_time

train_reconstruction_error = np.mean(np.abs(X_train - X_train_pred), axis=1)
test_reconstruction_error = np.mean(np.abs(X_test - X_test_pred), axis=1)

# Calculate ROC curve
optimal_threshold = find_optimal_threshold_f1(test_reconstruction_error, y_test)

y_test_pred = (test_reconstruction_error > optimal_threshold).astype(int)

# Evaluation
print("Evaluation:")
print(f"Optimal Threshold: {optimal_threshold:.3f}")  
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_test_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_test_pred):.3f}")
print(f"AUC: {roc_auc_score(y_test, test_reconstruction_error):.3f}")
print(f"Average inference time per sample: {(inference_time / len(X_test)) * 1000:.3f} ms")
print(f"Average processing time per sample: {(pre_proc_time / len(combined_features_df) * 1000):.3f} ms")

# Count the number of good and damaged bearings in the test set
unique, counts = np.unique(y_test, return_counts=True)
label_counts = dict(zip(unique, counts))

print(f"Number of good bearings (0) in the test set: {label_counts.get(0, 0)}")
print(f"Number of damaged bearings (1) in the test set: {label_counts.get(1, 0)}")

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
accuracy_test = []
roc_aucs_test = []
f1_scores_train = []
accuracy_train = []
precision_train = []
recalls_train = []
roc_aucs_train = []

for threshold in thresholds:
    y_test_pred = (test_reconstruction_error > threshold).astype(int)
    f1_scores_test.append(f1_score(y_test, y_test_pred))
    accuracy_test.append(accuracy_score(y_test, y_test_pred))
    precisions_test.append(precision_score(y_test, y_test_pred, zero_division=0))
    recalls_test.append(recall_score(y_test, y_test_pred))
    roc_aucs_test.append(roc_auc_score(y_test, test_reconstruction_error))
    y_train_pred = (train_reconstruction_error > threshold).astype(int)
    f1_scores_train.append(f1_score(y_train, y_train_pred))
    accuracy_train.append(accuracy_score(y_train, y_train_pred))
    precision_train.append(precision_score(y_train, y_train_pred, zero_division=0))
    recalls_train.append(recall_score(y_train, y_train_pred))
    roc_aucs_train.append(roc_auc_score(y_train, train_reconstruction_error))

# Plot metrics vs threshold
plot_metrics_vs_threshold(thresholds, f1_scores_test, accuracy_test, precisions_test, recalls_test, roc_aucs_test,
                        f1_scores_train, accuracy_train, precision_train, recalls_train, roc_aucs_train,
                        optimal_threshold)


remove_unused_trials(tuner_dir, project_name, best_trial)