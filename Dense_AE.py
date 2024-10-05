'''
Developer Notes:

- This model classify the samples as noise or "bearing presence" based on the reconstruction error of the autoencoder.
- This script used a data shape of (number of samples x number of features), unlike tge Conv_AE script that used a datashape of (number of samples x number of timesteps x number of features).

'''

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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, 
                             confusion_matrix, accuracy_score, roc_curve, classification_report, precision_recall_curve)
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras_tuner import HyperModel, HyperParameters, BayesianOptimization, Hyperband
from UN_Models import VDAE, vdae_model_builder, DENSE, DENSE_SIMPLE
from UN_Models_TuningVersion import DENSE_TUNING, vdae_model_builder_Tuning, VDAE_Tuning, DENSE_TUNING_MFCC
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
            'audio_2g': os.path.join(current_dir, 'Dataset_Bearings', 'LAST_TEST', '2_good', 'AUDIO'),
            'accel_2g': os.path.join(current_dir, 'Dataset_Bearings', 'LAST_TEST', '2_good', 'ACCEL'),
            'audio_1g': os.path.join(current_dir, 'Dataset_Bearings', 'LAST_TEST', '1_good', 'AUDIO'),
            'accel_1g': os.path.join(current_dir, 'Dataset_Bearings', 'LAST_TEST', '1_good', 'ACCEL')
        },
        'damaged_bearing': {
            'audio_s': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'DAMAGED', 'AUDIO'),
            'acel_s': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'DAMAGED', 'ACEL'),
            'audio_m': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'DAMAGED', 'AUDIO'),
            'acel_m': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'DAMAGED', 'ACEL'),
            'audio_new_amr': os.path.join(current_dir, 'Dataset_Bearings', 'NEW_AMR', 'DAMAGED', 'AUDIO'),
            'accel_new_amr': os.path.join(current_dir, 'Dataset_Bearings', 'NEW_AMR', 'DAMAGED', 'ACCEL'),
            'audio_1d': os.path.join(current_dir, 'Dataset_Bearings', 'LAST_TEST', '1_damaged', 'AUDIO'),
            'accel_1d': os.path.join(current_dir, 'Dataset_Bearings', 'LAST_TEST', '1_damaged', 'ACCEL'),
            'audio_2d': os.path.join(current_dir, 'Dataset_Bearings', 'LAST_TEST', '2_damaged', 'AUDIO'),
            'accel_2d': os.path.join(current_dir, 'Dataset_Bearings', 'LAST_TEST', '2_damaged', 'ACCEL'),
            'audio_1d_1g': os.path.join(current_dir, 'Dataset_Bearings', 'LAST_TEST', '1_damaged_1_good', 'AUDIO'),
            'accel_1d_1g': os.path.join(current_dir, 'Dataset_Bearings', 'LAST_TEST', '1_damaged_1_good', 'ACCEL'),
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
    
    n_samples_floor_noise = len(combined_features)
    print(f"Number of samples (Floor Noise): {n_samples_floor_noise}")

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

    n_samples_good_bearing = len(combined_features) - n_samples_floor_noise
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
    
    n_samples_damaged_bearing = len(combined_features) - n_samples_floor_noise - n_samples_good_bearing
    print(f"Number of samples (Damaged Bearing): {n_samples_damaged_bearing}")


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


def read_and_concat_data(base_path, preprocessing_options=config.preprocessing_options):
    # Initialize an empty list to store DataFrames
    dataframes_noise = []
    dataframes_bearings = []
    combined_df = []
    
    bearings_path = os.path.join(base_path, 'Bearings_Complete')
    
    noise_path = os.path.join(base_path, 'Noise')
    
    nr = preprocessing_options.get('noise_reduction', True)
    
    preprocessing_options.update({'noise_reduction': False})
        
    labels_noise_path = os.path.join(noise_path, 'labels.csv')
    labels_bearings_path = os.path.join(bearings_path, 'labels.csv')
    
    labels_noise = pd.read_csv(labels_noise_path)   
    labels_bearings = pd.read_csv(labels_bearings_path)
    labels_bearings['label'] = labels_bearings['label'] + 1

    
    labels = pd.concat([labels_noise, labels_bearings], axis=0)
    print(labels.head())
    print(labels.tail())
    
    # Iterate over the preprocessing options
    for option, include in preprocessing_options.items():
        if include:
            # Construct the file name based on the option (assuming CSV file names match keys)
            if nr:
                noise_path = os.path.join(noise_path, 'NR')
                csv_file = os.path.join(noise_path, f"{option}_nr.csv")                
            else:
                csv_file = os.path.join(noise_path, f"{option}.csv")
            
            try:
                # Read the CSV file into a DataFrame
                df = pd.read_csv(csv_file)
                # Append the DataFrame to the list
                dataframes_noise.append(df)
            except FileNotFoundError:
                print(f"File {csv_file} not found. Skipping...")

    # Iterate over the preprocessing options
    for option, include in preprocessing_options.items():
        if include:
            # Construct the file name based on the option (assuming CSV file names match keys)
            if nr:
                bearings_path = os.path.join(bearings_path, 'NR')
                csv_file = os.path.join(bearings_path, f"{option}_nr.csv")                
            else:
                csv_file = os.path.join(bearings_path, f"{option}.csv")
            
            try:
                # Read the CSV file into a DataFrame
                df = pd.read_csv(csv_file)
                # Append the DataFrame to the list
                dataframes_bearings.append(df)
            except FileNotFoundError:
                print(f"File {csv_file} not found. Skipping...")
                
    # Concatenate all DataFrames along columns (axis=1)
    if dataframes_noise:
        combined_df_noise = pd.concat(dataframes_noise, axis=1)
    else:
        raise ValueError("No valid dataframes to concatenate.")
    
    if dataframes_bearings:
        combined_df_bearing = pd.concat(dataframes_bearings, axis=1)
    else:
        raise ValueError("No valid dataframes to concatenate.")
    
    combined_df = pd.concat([combined_df_noise, combined_df_bearing], axis=0)
    

    print(combined_df.head())
    
    return combined_df, labels


def main():

    #########################################################################
    
    # Main execution
    directories = define_directories()
    combined_features, labels, pre_proc_time, pre_proc_string = load_and_extract_features(directories)
    
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(combined_features)
    
    # Convert features to a DataFrame
    features_df = pd.DataFrame(features, columns=combined_features.columns)
    #features_df.to_csv('ae_detect_stft_accel.csv', index=False)

    # Convert labels to a DataFrame
    labels_df = pd.DataFrame(labels, columns=["label"])
    #labels_df.to_csv('ae_detect_labels.csv', index=False)
    
    #########################################################################
    
    
    ##############################################################################
    
    '''features_df, labels = read_and_concat_data(os.path.join(os.getcwd(), 'Dataset_csv'))
    print(f'Features Shape: {features_df.shape}')
    features = features_df.to_numpy()
    labels = labels['label'].to_numpy()
    pre_proc_time = -1
    pre_proc_string = "_".join(method for method, value in config.preprocessing_options.items() if value)'''
    
    ##############################################################################
    
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
    
    print(f'X Train Shape: {X_train.shape}')
    
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
    only_bearings_eval = np.concatenate((bearings_val_complete[val_size:], bearings_test_complete[test_size:]))
    only_bearings_eval_labels = np.concatenate((bearings_val_labels[val_size:], bearings_test_labels[test_size:]))
    
    
    bearings_val_labels_ae = np.ones(len(bearings_val_labels))
    bearings_test_labels_ae = np.ones(len(bearings_test_labels)) 
    bearings_eval_labels_ae = np.ones(len(only_bearings_eval_labels))

    # The data shape is (number of samples x number of features)

    print(f"Preprocessing Time: {pre_proc_time:.3f} seconds")

    input_shape = X_train.shape[1]
    
    print(f"Input shape: {input_shape}")
    
    # Epochs and batch size
    epochs = 300
    batch_size = 256
    
    # Only for VDAE
    latent_dim = 32
    
    # Define directories
    current_dir = os.getcwd()    
    project_name = 'AE_TB_' + str(batch_size) + 'bs_' + pre_proc_string
    model_string = 'DAE'
    
    if model == 'VDAE':
        model_string = 'VDAE_lm_' + str(latent_dim)
        
    else:
        model_string = model
        
    # ' before batch size is used to identify models with costumized hyperparameters        
    model_name = f"{model_string}_{pre_proc_string}_bs_{batch_size}.keras"
    save_parameters_name = f"{model_string}_{pre_proc_string}_bs_{batch_size}_parameters.json"        
    metrics_file_name = f"{model_string}_{pre_proc_string}_bs_{batch_size}_results.csv"
    
    output_dir_metrics = os.path.join(current_dir, 'AE_Results', 'NEW_AMR', 'IMPROVED_DATASET')
    model_save_path = os.path.join(current_dir, 'AE_Models', 'NEW_AMR', 'IMPROVED_DATASET', 'Detection', model_name)
    os.makedirs(os.path.join(current_dir, 'AE_Models', 'NEW_AMR', 'IMPROVED_DATASET', 'Detection'), exist_ok=True)
        
    # For Bayesian tuning
    model_save_path_tuner = os.path.join(current_dir, 'AE_Models', 'NEW_AMR', 'IMPROVED_DATASET', 'Bayesian', 'Detection', model_name) 
    os.makedirs(os.path.join(current_dir, 'AE_Models', 'NEW_AMR', 'IMPROVED_DATASET', 'Bayesian', 'Detection'), exist_ok=True)
    parameters_save_path = os.path.join(current_dir, 'AE_Models', 'NEW_AMR', 'IMPROVED_DATASET', 'Bayesian', 'Detection', save_parameters_name)
    os.makedirs(os.path.join(current_dir, 'AE_Models', 'NEW_AMR', 'IMPROVED_DATASET', 'Bayesian', 'Detection'), exist_ok=True)
    output_dir_metrics_tuner = os.path.join(current_dir, 'AE_Models', 'NEW_AMR', 'Bayesian', 'IMPROVED_DATASET', 'Detection') 
    os.makedirs(os.path.join(current_dir, 'AE_Models', 'NEW_AMR', 'Bayesian', 'IMPROVED_DATASET', 'Detection'), exist_ok=True)
    tuner_dir = os.path.join(current_dir, 'Bayesian_Tuning', 'NEW_AMR', 'AE_DENSE', 'IMPROVED_DATASET', 'Detection')    
    os.makedirs(os.path.join(current_dir, 'Bayesian_Tuning', 'NEW_AMR', 'AE_DENSE', 'IMPROVED_DATASET', 'Detection'), exist_ok=True)
    
    # Early stopping setup
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True, verbose=2)
    
    ######################################################################################################################
    
    ## For Hypertuning, uncomment the following block and comment the block below it
    ## For training and testing the model with default parameters, comment the following block
    '''
    tuning = True
    
    # Define the BayesianOptimization tuner
    tuner = BayesianOptimization(
        hypermodel=DENSE_TUNING_MFCC(input_shape),
        objective='val_loss',
        max_trials=15,
        executions_per_trial=2,
        directory=tuner_dir,
        project_name=project_name
    )
    
    #tuner_hyper = Hyperband(
    #    hypermodel=DENSE_TUNING(input_shape),
    #    objective=Objective("val_loss", direction="min"),
         #max_trials=10,
    #    #executions_per_trial=2,
    #    max_epochs=epochs,
    #    factor=3,
    #    directory=tuner_dir,
    #    project_name=project_name
    #)

    try:
        # Attempt to load the best model        
        autoencoder = load_model(model_save_path_tuner)        
        
    except (IndexError, ValueError):
        # Handle the case where no best model is available
        print("No best model found. Performing search.")

        # Perform Bayesian optimization
        tuner.search(X_train, X_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], validation_split=0.2, verbose=1)
        
        # Try again to get the best model after the search
        try:
            autoencoder = tuner.get_best_models(num_models=1)[0]
            autoencoder.save(model_save_path_tuner)
            
            best_trial = tuner.oracle.get_best_trials(1)[0]
            print(f"Best trial: {best_trial.trial_id}")
            print(f"Best trial value: {best_trial.score}")

            # Best hyperparameters
            hyperparameters = best_trial.hyperparameters
            print(f"Best trial hyperparameters: {hyperparameters.values}")
            
            # Convert and write JSON object to file
            with open(parameters_save_path, "w") as outfile: 
                json.dump(hyperparameters.values, outfile)            
            
        except IndexError:
            print("Search completed, but no best model was found.")
            exit(1)'''
            
    ######################################################################################################################
    
    
    ######################################################################################################################
    
    ## For training and testing the model with default parameters, uncomment the following block and comment the block above it
    ## You can also use this block for hyperparameter tuning, but you will have to manually set the hyperparameters
    ## For automatic hyperparameter tuning, comment the following block
    tuning = False  
    
    if(config.model_load and os.path.exists(model_save_path_tuner)):
        autoencoder = load_model(model_save_path_tuner)
    
    else:
        autoencoder = DENSE(
                    # Hyperparameters from Bayesian Optimization with STFT features                    
                    input_dim=input_shape,                       # Replace with your actual input dimension
                    units_1=448,                                 # Custom number of units for first dense layer
                    units_2=256,                                 # Custom number of units for second dense layer
                    bottleneck_units=54,                         # Custom number of units for bottleneck layer
        ).build()
        
        #autoencoder = DENSE_SIMPLE(input_dim=input_shape).build()
            
        history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], validation_split=0.1, verbose=2)
        
        if(config.save_model):
            autoencoder.save(model_save_path)        
        
        # Plot training & validation loss values
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.show()
        
    ######################################################################################################################
    
    autoencoder.summary()
    
    # Reconstruction and threshold finding
    start_time = time.time()
    val_predictions = autoencoder.predict(X_val)
    inference_time = time.time() - start_time
    
    # Get the reconstruction error on the validation set
    val_errors = np.mean(np.square(X_val - val_predictions), axis=1)

    # Get the reconstruction error on the labeled validation anomalies
    bearing_val_predictions = autoencoder.predict(bearings_val)
    bearing_val_errors = np.mean(np.square(bearings_val - bearing_val_predictions), axis=1)
    
    # Tune the threshold based on precision-recall curve
    # Labels for the validation set should be all 0 (normal) and for the labeled validation anomalies should be 1 (anomalous)
    y_val_binary = np.array([0] * len(val_errors))  # Normal data
    y_bearing_binary = np.array([1] * len(bearing_val_errors))  # Anomalous data

    # Combine validation errors and bearing validation errors
    combined_val_errors = np.concatenate([val_errors, bearing_val_errors])
    combined_val_labels = np.concatenate([y_val_binary, y_bearing_binary])

    # Calculate precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(combined_val_labels, combined_val_errors)

    # Calculate the absolute difference between precision and recall
    diff = abs(precision - recall)

    # Find the index of the minimum difference
    optimal_idx = np.argmin(diff)   

    # Get the optimal threshold
    optimal_threshold = thresholds[optimal_idx]
    
    #optimal_threshold = find_optimal_threshold_f1(combined_val_errors, combined_val_labels)
    
    '''# Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(combined_val_labels, combined_val_errors)

    # Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)              # This gives you the threshold with the maximum difference between TPR and FPR
    optimal_threshold = thresholds[optimal_idx]'''

    print(f"Optimal Threshold: {optimal_threshold}")
    
    # Final evaluation on the test set
    bearing_test_predictions = autoencoder.predict(bearings_test)
    bearing_test_errors = np.mean(np.square(bearings_test - bearing_test_predictions), axis=1)

    # Predict the reconstruction error for the selected noise samples
    noise_test_predictions = autoencoder.predict(X_test)
    noise_test_errors = np.mean(np.square(X_test - noise_test_predictions), axis=1)

    # Combine the noise and bearing errors
    combined_test_errors = np.concatenate([noise_test_errors, bearing_test_errors])
    combined_test_labels = np.concatenate([y_test, bearings_test_labels_ae])
    
    # Plotting the error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(noise_test_errors, bins=50, alpha=0.7, label='Noise Test Errors', color='orange')
    plt.hist(bearing_test_errors, bins=50, alpha=0.7, label='Bearing Test Errors', color='blue')
    plt.title('Error Distribution for Noise and Bearing Test Samples')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    probs_df = pd.DataFrame({
        'True Label': combined_test_labels,  # Assuming y_test is the true binary labels array
        'Predicted Probability': combined_test_errors.flatten()
    })

    # Map the true labels to their corresponding names for better visualization
    probs_df['Label Name'] = probs_df['True Label'].map({0: 'Noise', 1: 'Bearing'})

    # Set up the figure
    plt.figure(figsize=(12, 6))

    # Define a custom color palette with greenish for Healthy and reddish for Damaged
    palette = ['#66c2a5', '#fc8d62']  # Colors: greenish for Healthy, reddish for Damaged
    label_names = ['Noise', 'Bearing']
    color_map = dict(zip(label_names, palette))

    # Plot histograms with KDE disabled for better control over KDE plots
    sns.histplot(data=probs_df, x='Predicted Probability', hue='Label Name', kde=False, bins=50, palette=color_map, alpha=0.4)

    # Plot KDE curves separately for each label to ensure correct color and labeling
    for label in label_names:
        sns.kdeplot(
            data=probs_df[probs_df['Label Name'] == label],
            x='Predicted Probability',
            color=color_map[label],
            label=f'{label} Curve',
            linewidth=2
        )
    threshold_string = str(round(optimal_threshold, 2))
    # Customize the plot
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Error Distribution for Noise and Bearing Test Samples')
    plt.axvline(optimal_threshold, color='red', linestyle='--', label='Decision Threshold (' + threshold_string + ')')
    # Place the legend inside the plot in the upper right
    plt.legend(title='True Label', loc='upper right')
    plt.show()
    # Determine which samples are anomalies based on the optimal threshold
    test_anomalies = combined_test_errors > optimal_threshold

    # Calculate and print the final detection metrics
    predicted_labels = test_anomalies.astype(int)

    print(classification_report(combined_test_labels, predicted_labels, target_names=["Noise", "Anomaly"]))
    
    rec_error_noise_avg = np.mean(val_errors)
    rec_error_bearing_avg = np.mean(bearing_val_errors)
    rec_error_noise_std = np.std(val_errors)
    rec_error_bearing_std = np.std(bearing_val_errors)
    
    print("Reconstruction Errors:")
    print("Average Reconstruction Error for Validation Data (only noise):")
    print(f'Mean Reconstruction Error: {rec_error_noise_avg}')
    print(f'Standard Deviation of Reconstruction Error: {rec_error_noise_std}')
    print("Average Reconstruction Error for Validation Data (only bearings):")
    print(f'Mean Reconstruction Error: {rec_error_bearing_avg}')
    print(f'Standard Deviation of Reconstruction Error: {rec_error_bearing_std}')
    print("\n")
    
    acc = accuracy_score(combined_test_labels, predicted_labels)
    prec = precision_score(combined_test_labels, predicted_labels)
    rec = recall_score(combined_test_labels, predicted_labels)
    f1 = f1_score(combined_test_labels, predicted_labels)
    auc = roc_auc_score(combined_test_labels, predicted_labels)
    inf_time = inference_time / len(X_val) * 1000
    proc_time = pre_proc_time / (len(y_train) + len(y_val)) * 1000
    
    # Global Evaluation
    print("Evaluation:")
    print(f"Optimal Threshold: {optimal_threshold}")  
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"AUC: {auc:.3f}")
    print(f"Average Inference Time per Sample: {inf_time:.3f} ms")
    print(f"Average Preprocessing Time per Sample: {proc_time:.3f} ms")   
    
    if config.save_metrics:
        if tuning:
            save_metrics_to_csv(output_dir_metrics_tuner, metrics_file_name, prec, rec, f1, acc, auc, optimal_threshold, rec_error_noise_avg, rec_error_bearing_avg, rec_error_noise_std, rec_error_bearing_std, inf_time, proc_time)
        else:
            save_metrics_to_csv(output_dir_metrics, metrics_file_name, prec, rec, f1, acc, auc, optimal_threshold, rec_error_noise_avg, rec_error_bearing_avg, rec_error_noise_std, rec_error_bearing_std, inf_time, proc_time)

    # Count the number of noise samples and bearings samples in the test set
    unique, counts = np.unique(combined_test_labels, return_counts=True)
    label_counts = dict(zip(unique, counts))

    print(f"Number of noise samples (0) in the test set: {label_counts.get(0, 0)}")
    print(f"Number of bearing samples (1) in the test set: {label_counts.get(1, 0)}")

    cm = confusion_matrix(combined_test_labels, predicted_labels)

    # Normalize the confusion matrix to percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create annotations with both percentage and absolute values
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm_percentage[i, j]:.1f}% ({cm[i, j]})'

    # Plot the confusion matrix with both percentages and absolute values
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=annotations, fmt='', cmap='Blues', cbar=False,
                xticklabels=['Healthy', 'Damaged'],
                yticklabels=['Healthy', 'Damaged'])

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Percentage and Count)')
    plt.show()
    
    # Final test with left over bearing samples
    print("Final Evaluation:")
    only_bearings_eval_predictions = autoencoder.predict(only_bearings_eval)
    only_bearings_eval_errors = np.mean(np.square(only_bearings_eval - only_bearings_eval_predictions), axis=1)
    detected_labels = (only_bearings_eval_errors > optimal_threshold).astype(int)
    detected_anomalies = np.count_nonzero(detected_labels)
    detected_anomalies_per = detected_anomalies / len(only_bearings_eval)
    print(f"Detected Anomalies: {detected_anomalies}")
    print(f"Detected Anomalies Percentage: {detected_anomalies_per}")
    
    
    # Thresholds vs metrics for validation data
    thresholds = np.linspace(min(combined_val_errors), max(combined_val_errors), 100)
    f1_scores_val = []
    precisions_val = []
    recalls_val = []
    accuracy_val = []
    roc_aucs_val = []

    for threshold in thresholds:
        y_val_pred = (combined_val_errors > threshold).astype(int)      # Assuming combined_val_errors is for validation
        f1_scores_val.append(f1_score(combined_val_labels, y_val_pred))
        accuracy_val.append(accuracy_score(combined_val_labels, y_val_pred))
        precisions_val.append(precision_score(combined_val_labels, y_val_pred, zero_division=0))
        recalls_val.append(recall_score(combined_val_labels, y_val_pred))
        roc_aucs_val.append(roc_auc_score(combined_val_labels, combined_val_errors))

    # Thresholds vs metrics for test data
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
    plot_metrics_vs_threshold(
        thresholds, 
        optimal_threshold, 
        f1_scores_test=f1_scores_test, 
        accuracy_test=accuracy_test, 
        precisions_test=precisions_test, 
        recalls_test=recalls_test, 
        roc_aucs_test=roc_aucs_test,
        f1_scores_val=f1_scores_val, 
        accuracy_val=accuracy_val, 
        precisions_val=precisions_val, 
        recalls_val=recalls_val, 
        roc_aucs_val=roc_aucs_val
    )

        
if __name__ == "__main__":
    main()