import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ydf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import config  # Import the config file
import PreProc_Function as ppf
from imblearn.over_sampling import SMOTE

# For Future Use
#from sklearn.feature_selection import SelectKBest, f_classif
#from sklearn.model_selection import GridSearchCV

# Helper function to sort files by the numeric part in the filename
def sort_key(file_path):
    file_name = os.path.basename(file_path)
    numeric_part = ''.join(filter(str.isdigit, file_name))
    return int(numeric_part) if numeric_part else 0


def define_directories():
    current_dir = os.getcwd()
    directories = {
        'bearing_test': {
            'audio_2g': os.path.join(current_dir, 'Dataset_Bearings', 'LAST_TEST', '1_damaged_1_good', 'AUDIO'),
            'accel_2g': os.path.join(current_dir, 'Dataset_Bearings', 'LAST_TEST', '1_damaged_1_good', 'ACCEL'),
        },
        'noise_profile': os.path.join(current_dir, 'Dataset_Piso', 'Noise.WAV')
    }
    return directories


# Load files and extract features
def load_and_extract_features(directories):
    combined_features = []
    labels = []

    methods_string = "_".join(method for method, value in config.preprocessing_options.items() if value)
    start_time_pre_proc = time.time()
    
    # Process bearing test data
    for audio_file, accel_file in zip(
        sorted([os.path.join(directories['bearing_test']['audio_2g'], file) for file in os.listdir(directories['bearing_test']['audio_2g']) if file.endswith('.WAV')], key=sort_key),
        sorted([os.path.join(directories['bearing_test']['accel_2g'], file) for file in os.listdir(directories['bearing_test']['accel_2g']) if file.endswith('.csv')], key=sort_key)
    ):
        audio_features = ppf.extract_audio_features(audio_file, directories['noise_profile'], config.preprocessing_options)
        accel_features = ppf.extract_accel_features(accel_file)
        combined = {**audio_features, **accel_features}
        combined_features.append(combined)
        labels.append(1)  # 0 for good bearing

    n_samples_good_bearing = len(combined_features)
    print(f"Number of samples (Good Bearing): {n_samples_good_bearing}")
    
    end_time_pre_proc = time.time()
    
    return pd.DataFrame(combined_features), np.array(labels), end_time_pre_proc - start_time_pre_proc, methods_string


def main():
    
    # Current directory
    current_dir = os.getcwd()
    
    ##########################################################################################################################
    
    ## Read from raw data and preprocess
    
    # Load files and extract features
    combined_features_df, y, pre_proc_time, methods_string = load_and_extract_features(define_directories())    

    # Average pre-processing time
    average_pre_proc_time = pre_proc_time / len(y)

    # Normalize features
    scaler = StandardScaler()
    combined_features_normalized = scaler.fit_transform(combined_features_df)

    # Convert features to a DataFrame
    features_df = pd.DataFrame(combined_features_normalized, columns=combined_features_df.columns)
    #features_df.to_csv('features_nr.csv', index=False)

    # Convert labels to a DataFrame
    labels_df = pd.DataFrame(y, columns=["label"])
    #labels_df.to_csv('labels.csv', index=False)
    
    # Combine features and labels into one DataFrame
    test_data = pd.concat([features_df, labels_df], axis=1)
    
    # Yggdrasil GBDT Model -> Distinguish between good and damaged bearings    
    model_string = 'RF'
    gbdt_save_path = os.path.join(current_dir, 'DTs_Models', model_string, model_string.lower() + '_stft')
    
    ygg = ydf.load_model(gbdt_save_path)
    
    y_pred_probs = ygg.predict(test_data)
    
    # Threshold for classifying a sample as a damaged bearing
    threshold = 0.5  # Based on the probability distribution plot
    
    # Convert predicted probabilities to binary class labels
    pred_bearing_faults = (y_pred_probs > threshold).astype(int).flatten()    
    
    print("Predicted bearing faults:", pred_bearing_faults)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(labels_df['label'].values, pred_bearing_faults, labels=[0, 1])

    # Plot confusion matrix
    target_names = ['HEALTHY', 'DAMAGED']
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()