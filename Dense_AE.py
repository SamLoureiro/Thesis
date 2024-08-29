'''
Developer Notes:

- This script used a data shape of (number of samples x number of features), unlike tge Conv_AE script that used a datashape of (number of samples x number of timesteps x number of features).
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
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, 
                             confusion_matrix, accuracy_score, roc_curve, classification_report, precision_recall_curve)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, BatchNormalization, Dropout, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras_tuner import HyperModel, HyperParameters, BayesianOptimization
import matplotlib.pyplot as plt
import seaborn as sns
import PreProc_Function as ppf
from AE_Aux_Func import reduce_dimensions, plot_reduced_data, plot_metrics_vs_threshold, find_optimal_threshold_f1


class DENSE(HyperModel):
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
        autoencoder.compile(optimizer=Adam(learning_rate=learning_rate),
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

    # Process good bearings
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
        labels.append(1)  # 1 for good bearing

    n_samples_good_bearing = len(combined_features)
    print(f"Number of samples (Good Bearing): {n_samples_good_bearing}")

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
        labels.append(2)  # 2 for damaged bearing

    n_samples_damaged_bearing = len(combined_features) - n_samples_good_bearing
    print(f"Number of samples (Damaged Bearing): {n_samples_damaged_bearing}")

    # Process floor noise
    for audio_file, accel_file in zip(
        sorted([os.path.join(directories['smooth_floor']['audio'], file) for file in os.listdir(directories['smooth_floor']['audio']) if file.endswith('.WAV')], key=sort_key) +
        sorted([os.path.join(directories['tiled_floor']['audio'], file) for file in os.listdir(directories['tiled_floor']['audio']) if file.endswith('.WAV')], key=sort_key),
        sorted([os.path.join(directories['smooth_floor']['acel'], file) for file in os.listdir(directories['smooth_floor']['acel']) if file.endswith('.csv')], key=sort_key) +
        sorted([os.path.join(directories['tiled_floor']['acel'], file) for file in os.listdir(directories['tiled_floor']['acel']) if file.endswith('.csv')], key=sort_key)
    ):
        audio_features = ppf.extract_audio_features(audio_file, directories['noise_profile'], config.preprocessing_options)
        accel_features = ppf.extract_accel_features(accel_file)
        combined = {**audio_features, **accel_features}
        combined_features.append(combined)
        labels.append(0)  # 0 for floor noise

    n_samples_floor_noise = len(combined_features) - n_samples_good_bearing - n_samples_damaged_bearing
    print(f"Number of samples (Floor Noise): {n_samples_floor_noise}")

    end_time_pre_proc = time.time()

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


def main():

    # Main execution
    directories = define_directories()
    combined_features_normalized, labels, pre_proc_time, pre_proc_string = load_and_extract_features(directories)

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(combined_features_normalized)

    # Noise samples (considered as regular data)
    noise_samples = features[np.array(labels) == 0]    
    noise_labels = np.array([0] * noise_samples.shape[0])  # All labels in training set are 0

    # Bearing samples (considered as anomalies)
    bearings_samples = features[np.array(labels) != 0]
    bearings_labels = np.array(labels)[np.array(labels) != 0] 

    # Split bearing samples into good and damaged bearings (For later use)
    good_bearing_samples = features[np.array(labels) == 1]
    good_bearing_labels = np.array([1] * good_bearing_samples.shape[0])

    damaged_bearing_samples = features[np.array(labels) == 2]
    damaged_bearing_labels = np.array([2] * damaged_bearing_samples.shape[0])   

    # Split noise data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(noise_samples, noise_labels, test_size=0.1, random_state=42)

    # Further split labeled anomalies into validation and test sets
    bearings_train, bearings_test, bearings_train_labels, bearings_test_labels = train_test_split(bearings_samples, bearings_labels, test_size=0.5, random_state=42)

    bearings_train = bearings_train[:95]
    bearings_test = bearings_test[:95]
    bearings_train_labels = bearings_train_labels[:95]
    bearings_test_labels = bearings_test_labels[:95]


    bearings_train_labels_ae = np.ones(len(bearings_train_labels))
    bearings_test_labels_ae = np.ones(len(bearings_test_labels)) 

    # The data shape is (number of samples x number of features)

    print(f"Preprocessing Time: {pre_proc_time:.3f} seconds")

    input_shape = X_train.shape[1]
    
    print(f"Input shape: {input_shape}")
    
    # Epochs and batch size
    epochs = 100
    batch_size = 32

    # Define directories
    current_dir = os.getcwd()
    tuner_dir = os.path.join(current_dir, 'Bayesian_Tuning', 'DENSE')
    project_name = 'AE_TB_' + str(batch_size) + 'bs_' + pre_proc_string


    # Define the BayesianOptimization tuner
    tuner = BayesianOptimization(
        hypermodel=DENSE(input_shape),
        objective='val_loss',
        max_trials=20,
        executions_per_trial=1,
        directory=tuner_dir,
        project_name=project_name
    )

    # Early stopping setup
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    # If the tuner directory exists, try to load the best trial, otherwise perform Bayesian optimization and save and load the best trial
    try:
        # Load the best trial
        best_trial = tuner.oracle.get_best_trials(1)[0]
        
    except IndexError:
        # Perform Bayesian optimization
        tuner.search(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping], verbose=1)    
        # Load the best trial
        best_trial = tuner.oracle.get_best_trials(1)[0]
        pass

    except AttributeError as e:
        print(f"Error while accessing best trial attributes: {e}")


    print(f"Best trial: {best_trial.trial_id}")
    print(f"Best trial value: {best_trial.score}")

    # Best hyperparameters
    hyperparameters = best_trial.hyperparameters
    print(f"Best trial hyperparameters: {hyperparameters.values}")

    # Build the best model using the best hyperparameters
    autoencoder = DENSE(input_shape).build(hyperparameters)
    
    # Reconstruction and threshold finding
    start_time = time.time()
    val_predictions = autoencoder.predict(X_val)
    inference_time = time.time() - start_time
    
    # Get the reconstruction error on the validation set
    val_errors = np.mean(np.square(X_val - val_predictions), axis=1)   

        
    # Get the reconstruction error on the labeled validation anomalies
    bearing_val_predictions = autoencoder.predict(bearings_train)
    bearing_val_errors = np.mean(np.square(bearings_train - bearing_val_predictions), axis=1)
    
    # Tune the threshold based on precision-recall curve
    # Labels for the validation set should be all 0 (normal) and for the labeled validation anomalies should be 1 (anomalous)
    y_val_binary = np.array([0] * len(val_errors))  # Normal data
    y_bearing_binary = np.array([1] * len(bearing_val_errors))  # Anomalous data

    # Combine validation errors and bearing validation errors
    combined_errors = np.concatenate([val_errors, bearing_val_errors])
    combined_labels = np.concatenate([y_val_binary, y_bearing_binary])

    # Calculate precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(combined_labels, combined_errors)

    # Calculate the absolute difference between precision and recall
    diff = abs(precision - recall)

    # Find the index of the minimum difference
    optimal_idx = np.argmin(diff)

    # Get the optimal threshold
    optimal_threshold = thresholds[optimal_idx]
    
    #optimal_threshold = 0.837   
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    
    # Final evaluation on the test set
    bearing_test_predictions = autoencoder.predict(bearings_test)
    bearing_test_errors = np.mean(np.square(bearings_test - bearing_test_predictions), axis=1)

    # Randomly select a predefined number of noise samples to include in the test set
    predefined_noise_samples_count = len(bearings_test)  # Set the desired number of noise samples
    random_noise_indices = np.random.choice(noise_samples.shape[0], predefined_noise_samples_count, replace=False)
    selected_noise_samples = noise_samples[random_noise_indices]
    selected_noise_labels = noise_labels[random_noise_indices]

    # Predict the reconstruction error for the selected noise samples
    noise_test_predictions = autoencoder.predict(selected_noise_samples)
    noise_test_errors = np.mean(np.square(selected_noise_samples - noise_test_predictions), axis=1)

    # Combine the noise and bearing errors
    combined_test_data = np.concatenate([bearings_test, selected_noise_samples])
    combined_test_errors = np.concatenate([bearing_test_errors, noise_test_errors])
    combined_test_labels = np.concatenate([bearings_test_labels_ae, selected_noise_labels])

    # Determine which samples are anomalies based on the optimal threshold
    test_anomalies = combined_test_errors > optimal_threshold

    # Calculate and print the final detection metrics
    predicted_labels = test_anomalies.astype(int)

    print(classification_report(combined_test_labels, predicted_labels, target_names=["Noise", "Anomaly"]))
    
    print("Reconstruction Errors:")
    print("Average Reconstruction Error for Validation Data (only noise):")
    print(f'Mean Reconstruction Error: {np.mean(val_errors)}')
    print(f'Standard Deviation of Reconstruction Error: {np.std(val_errors)}')
    print("Average Reconstruction Error for Validation Data (only bearings):")
    print(f'Mean Reconstruction Error: {np.mean(bearing_val_predictions)}')
    print(f'Standard Deviation of Reconstruction Error: {np.std(bearing_val_errors)}')
    print("\n")
    
    # Global Evaluation
    print("Evaluation:")
    print(f"Optimal Threshold: {optimal_threshold:.3f}")  
    print(f"Accuracy: {accuracy_score(combined_test_labels, predicted_labels):.3f}")
    print(f"Precision: {precision_score(combined_test_labels, predicted_labels):.3f}")
    print(f"Recall: {recall_score(combined_test_labels, predicted_labels):.3f}")
    print(f"F1 Score: {f1_score(combined_test_labels, predicted_labels):.3f}")
    print(f"AUC: {roc_auc_score(combined_test_labels, predicted_labels):.3f}")
    print(f"Average inference time per sample: {(inference_time / len(X_val)) * 1000:.3f} ms")
    print(f"Average processing time per sample: {(pre_proc_time / (len(y_train) + len(y_val)) * 1000):.3f} ms")


    # Count the number of noise samples and bearings samples in the test set
    unique, counts = np.unique(combined_test_labels, return_counts=True)
    label_counts = dict(zip(unique, counts))

    print(f"Number of noise samples (0) in the test set: {label_counts.get(0, 0)}")
    print(f"Number of bearing samples (1) in the test set: {label_counts.get(1, 0)}")

    cm = confusion_matrix(combined_test_labels, predicted_labels)
    

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Noise', 'Anomaly'],
                yticklabels=['Noise', 'Anomaly'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Reduce dimensions and plot
    X_reduced = reduce_dimensions(combined_test_data, method='t-SNE')
    plot_reduced_data(X_reduced, combined_test_labels)
    plot_reduced_data(X_reduced, combined_test_labels, predicted_labels)
        
    # Thresholds vs metrics
    
    # Use the IQR method to filter out outliers
    Q1 = np.percentile(combined_test_errors, 25)
    Q3 = np.percentile(combined_test_errors, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_errors = combined_test_errors[(combined_test_errors >= lower_bound) & (combined_test_errors <= upper_bound)]

    # Generate thresholds within the filtered range
    thresholds = np.linspace(min(filtered_errors), max(filtered_errors), 100)
    f1_scores_test = []
    precisions_test = []
    recalls_test = []
    accuracy_test = []
    roc_aucs_test = []
    for threshold in thresholds:
        y_test_pred = (combined_test_errors > threshold).astype(int)
        f1_scores_test.append(f1_score(combined_test_labels, y_test_pred))
        accuracy_test.append(accuracy_score(combined_test_labels, y_test_pred))
        precisions_test.append(precision_score(combined_test_labels, y_test_pred, zero_division=0))
        recalls_test.append(recall_score(combined_test_labels, y_test_pred))
        roc_aucs_test.append(roc_auc_score(combined_test_labels, combined_test_errors))
    
    # Plot metrics vs threshold
    plot_metrics_vs_threshold(thresholds, f1_scores_test, accuracy_test, precisions_test, recalls_test, roc_aucs_test, optimal_threshold)

    remove_unused_trials(tuner_dir, project_name, best_trial)
    
if __name__ == "__main__":
    main()