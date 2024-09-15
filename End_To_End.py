import os
import shutil
import time
import numpy as np
import pandas as pd
import config
import random
import json 
import matplotlib.pyplot as plt
import seaborn as sns
import PreProc_Function as ppf
import ydf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, 
                             confusion_matrix, accuracy_score, roc_curve, classification_report, precision_recall_curve)
from sklearn.model_selection import train_test_split
from keras.models import load_model
from UN_CNN_Models import DENSE
from UN_CNN_Models_TuningVersion import DENSE_TUNING
from AE_Aux_Func import reduce_dimensions, plot_reduced_data, plot_metrics_vs_threshold, find_optimal_threshold_f1, save_metrics_to_csv



# Define directories and file paths
def define_directories():
    current_dir = os.getcwd()
    directories = {
        'good_bearing': {
            'audio_m': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'GOOD', 'AUDIO'),
            'acel_m': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'GOOD', 'ACEL'),
            'audio_s': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'GOOD', 'AUDIO'),
            'acel_s': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'GOOD', 'ACEL'),
            'audio_new_amr': os.path.join(current_dir, 'Dataset_Bearings', 'NEW_AMR', 'GOOD', 'AUDIO'),
            'accel_new_amr': os.path.join(current_dir, 'Dataset_Bearings', 'NEW_AMR', 'GOOD', 'ACCEL'),
        },
        'damaged_bearing': {
            'audio_s': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'DAMAGED', 'AUDIO'),
            'acel_s': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'DAMAGED', 'ACEL'),
            'audio_m': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'DAMAGED', 'AUDIO'),
            'acel_m': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'DAMAGED', 'ACEL'),
            'audio_new_amr': os.path.join(current_dir, 'Dataset_Bearings', 'NEW_AMR', 'DAMAGED', 'AUDIO'),
            'accel_new_amr': os.path.join(current_dir, 'Dataset_Bearings', 'NEW_AMR', 'DAMAGED', 'ACCEL'),
        },
        'smooth_floor': {
            'audio': os.path.join(current_dir, 'Dataset_Piso', 'LISO', 'SAMPLES_1s', 'AUDIO'),
            'acel': os.path.join(current_dir, 'Dataset_Piso', 'LISO', 'SAMPLES_1s', 'ACCEL'),
            'audio_new_amr': os.path.join(current_dir, 'Dataset_Piso', 'NEW_AMR', 'LISO', 'SAMPLES_1s', 'AUDIO'),
            'accel_new_amr': os.path.join(current_dir, 'Dataset_Piso', 'NEW_AMR', 'LISO', 'SAMPLES_1s', 'ACCEL')
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
        sorted([os.path.join(directories['good_bearing']['audio_m'], file) for file in os.listdir(directories['good_bearing']['audio_m']) if file.endswith('.WAV')], key=sort_key) + 
        sorted([os.path.join(directories['good_bearing']['audio_new_amr'], file) for file in os.listdir(directories['good_bearing']['audio_new_amr']) if file.endswith('.WAV')], key=sort_key),
        sorted([os.path.join(directories['good_bearing']['acel_s'], file) for file in os.listdir(directories['good_bearing']['acel_s']) if file.endswith('.csv')], key=sort_key) +
        sorted([os.path.join(directories['good_bearing']['acel_m'], file) for file in os.listdir(directories['good_bearing']['acel_m']) if file.endswith('.csv')], key=sort_key) +
        sorted([os.path.join(directories['good_bearing']['accel_new_amr'], file) for file in os.listdir(directories['good_bearing']['accel_new_amr']) if file.endswith('.csv')], key=sort_key)
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
        sorted([os.path.join(directories['damaged_bearing']['audio_m'], file) for file in os.listdir(directories['damaged_bearing']['audio_m']) if file.endswith('.WAV')], key=sort_key) +
        sorted([os.path.join(directories['damaged_bearing']['audio_new_amr'], file) for file in os.listdir(directories['damaged_bearing']['audio_new_amr']) if file.endswith('.WAV')], key=sort_key),
        sorted([os.path.join(directories['damaged_bearing']['acel_s'], file) for file in os.listdir(directories['damaged_bearing']['acel_s']) if file.endswith('.csv')], key=sort_key) +
        sorted([os.path.join(directories['damaged_bearing']['acel_m'], file) for file in os.listdir(directories['damaged_bearing']['acel_m']) if file.endswith('.csv')], key=sort_key) +
        sorted([os.path.join(directories['damaged_bearing']['accel_new_amr'], file) for file in os.listdir(directories['damaged_bearing']['accel_new_amr']) if file.endswith('.csv')], key=sort_key)
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
        sorted([os.path.join(directories['tiled_floor']['audio'], file) for file in os.listdir(directories['tiled_floor']['audio']) if file.endswith('.WAV')], key=sort_key) +
        sorted([os.path.join(directories['smooth_floor']['audio_new_amr'], file) for file in os.listdir(directories['smooth_floor']['audio_new_amr']) if file.endswith('.WAV')], key=sort_key),
        sorted([os.path.join(directories['smooth_floor']['acel'], file) for file in os.listdir(directories['smooth_floor']['acel']) if file.endswith('.csv')], key=sort_key) +
        sorted([os.path.join(directories['tiled_floor']['acel'], file) for file in os.listdir(directories['tiled_floor']['acel']) if file.endswith('.csv')], key=sort_key) +
        sorted([os.path.join(directories['smooth_floor']['accel_new_amr'], file) for file in os.listdir(directories['smooth_floor']['accel_new_amr']) if file.endswith('.csv')], key=sort_key)
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

    # Change the model name to the desired model
    model = 'DAE' 

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
    X_train, X_val_complete, y_train, y_val_complete = train_test_split(noise_samples, noise_labels, test_size=0.2, random_state=42)
    
    # Split noise data into training and validation sets
    X_val_complete, X_test_complete, y_val_complete, y_test_complete = train_test_split(X_val_complete, y_val_complete, test_size=0.35, random_state=42)
    
    # Further split labeled anomalies into validation and test sets
    bearings_val_complete, bearings_test_complete, bearings_val_labels_complete, bearings_test_labels_complete = train_test_split(bearings_samples, bearings_labels, test_size=0.5, random_state=42)
    
    print("\nNumber of Noise Training Samples:", len(X_train))
    print("Number of Noise Validation Samples:", len(X_val_complete))
    
    print("Number of Bearing Validation Samples:", len(bearings_test_complete))
    print("Number of Bearing Test Samples:", len(bearings_val_complete))   
    
    
    val_size = min(len(X_val_complete), len(bearings_val_complete))
    
    X_val = X_val_complete[:val_size]
    y_val = y_val_complete[:val_size]
    
    # In an ideal situation, the test set should be independent of the training and validation sets. 
    # However, due to the lack of noise data, the noise test set is created with random samples from the the dataset.
    #test_size = min(len(bearings_test_labels_complete), len(noise_labels))
    
    test_size = min(len(bearings_test_labels_complete), len(X_test_complete))
    
    X_test = X_test_complete[:test_size]
    y_test = y_test_complete[:test_size]
    
    # Force the test set and val set to be balanced to avoid class imbalance and bias in the evaluation and threshold parameterization
    bearings_val = bearings_val_complete[:val_size]
    bearings_test = bearings_test_complete[:test_size]
    bearings_val_labels = bearings_val_labels_complete[:val_size]
    bearings_test_labels = bearings_test_labels_complete[:test_size]
    
    # The remaining bearing samples are used for the final evaluation
    # The remaining bearing samples are used for the final evaluation
    only_bearings_eval = np.concatenate((bearings_val_complete[val_size:], bearings_test_complete[test_size:]))
    only_bearings_eval_labels = np.concatenate((bearings_val_labels[val_size:], bearings_test_labels[test_size:]))
    
    
    bearings_val_labels_ae = np.ones(len(bearings_val_labels))
    bearings_test_labels_ae = np.ones(len(bearings_test_labels)) 
    bearings_eval_labels_ae = np.ones(len(only_bearings_eval_labels))

    # The data shape is (number of samples x number of features)

    print(f"Preprocessing Time: {pre_proc_time:.3f} seconds")

    input_shape = X_train.shape[1]
    
    print(f"Input shape: {input_shape}")
    
    # Define directories
    current_dir = os.getcwd()    
    
    ##############################################################################################################
    
    # Autoencoder Model -> Distinguish between noise and bearing samples
    
    model_string = 'DAE'   
    model_name = f"{model_string}_{pre_proc_string}.keras"
    model_save_path = os.path.join(current_dir, 'AE_Models', 'NEW_AMR', 'Bayesian', model_name) 
    
    autoencoder = load_model(model_save_path, custom_objects={'DENSE_TUNING': DENSE_TUNING})
        
    # Full test set
    combined_X_test = np.concatenate([bearings_test, X_test])
    
    # Predict the reconstruction error for the test bearing samples
    bearing_test_predictions = autoencoder.predict(bearings_test)
    bearing_test_errors = np.mean(np.square(bearings_test - bearing_test_predictions), axis=1)
    
    # Predict the reconstruction error for the test noise samples
    noise_test_predictions = autoencoder.predict(X_test)
    noise_test_errors = np.mean(np.square(X_test - noise_test_predictions), axis=1)

    # Combine the noise and bearing errors
    combined_test_errors = np.concatenate([bearing_test_errors, noise_test_errors])
    combined_test_labels_ae = np.concatenate([bearings_test_labels_ae, y_test])
    combined_test_labels = np.concatenate([bearings_test_labels, y_test])
    
    optimal_threshold = 0.444

    # Determine which samples are anomalies based on the optimal threshold
    test_anomalies = combined_test_errors > optimal_threshold
    
    # Calculate and print the final detection metrics
    predicted_labels = test_anomalies.astype(int)    # e.g. [0, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    
    
    
    ##############################################################################################################
    
    # Yggdrasil GBDT Model -> Distinguish between good and damaged bearings
    
    model_string = 'GBDT'
    gbdt_save_path = os.path.join(current_dir, 'DTs_Models', 'GBDT', 'gbdt_stft')
    
    ygg = ydf.load_model(gbdt_save_path)
    
    pred_bearing_indices = np.where(predicted_labels == 1) 
    
    X_test_bearing = combined_X_test[pred_bearing_indices]
    
    X_test_bearing_labels = combined_test_labels[pred_bearing_indices]
    
    X_test_bearing = pd.DataFrame(X_test_bearing)
    
    pred_bearing_faults = ygg.predict(X_test_bearing)
    
    pred_bearing_faults_3_labels = pred_bearing_faults + 1   # Good Bearing (1), Damaged Bearing (2)
    
    print("Predicted Bearing Faults:", pred_bearing_faults_3_labels)
    print("Predicted Bearing Faults Length:", len(pred_bearing_faults_3_labels))
    
    print("True Bearing Faults:", X_test_bearing_labels)
    print("True Bearing Faults Length:", len(X_test_bearing_labels))

if __name__ == "__main__":
    main()