import os
import time
import warnings
import numpy as np
import pandas as pd
import librosa
import noisereduce as nr
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
import tensorflow_decision_forests as tfdf
import tensorflow as tf
from keras.layers import TFSMLayer
from keras.models import load_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import config  # Import the config file
import PreProc_Function as ppf

# For Future Use
#from imblearn.over_sampling import SMOTE
#from sklearn.feature_selection import SelectKBest, f_classif
#from sklearn.model_selection import GridSearchCV

# Define directories
current_dir = os.getcwd()

good_bearing_dir_audio_m = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'GOOD', 'AUDIO')
damaged_bearing_dir_audio_m = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'DAMAGED', 'AUDIO')
good_bearing_dir_acel_m = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'GOOD', 'ACEL')
damaged_bearing_dir_acel_m = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'DAMAGED', 'ACEL')

good_bearing_dir_audio_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'GOOD', 'AUDIO')
damaged_bearing_dir_audio_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'DAMAGED', 'AUDIO')
good_bearing_dir_acel_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'GOOD', 'ACEL')
damaged_bearing_dir_acel_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'DAMAGED', 'ACEL')

good_bearing_dir_audio_new_amr = os.path.join(current_dir, 'Dataset_Bearings', 'NEW_AMR', 'GOOD', 'AUDIO')
damaged_bearing_dir_audio_new_amr = os.path.join(current_dir, 'Dataset_Bearings', 'NEW_AMR', 'DAMAGED', 'AUDIO')
good_bearing_dir_accel_new_amr  = os.path.join(current_dir, 'Dataset_Bearings', 'NEW_AMR', 'GOOD', 'ACCEL')
damaged_bearing_dir_accel_new_amr  = os.path.join(current_dir, 'Dataset_Bearings', 'NEW_AMR', 'DAMAGED', 'ACCEL')


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

# Load audio and accelerometer files for NEW_AMR
good_bearing_files_audio_new_amr = sorted(
    [os.path.join(good_bearing_dir_audio_new_amr, file) for file in os.listdir(good_bearing_dir_audio_new_amr) if file.endswith('.WAV')],
    key=sort_key
)
damaged_bearing_files_audio_new_amr = sorted(
    [os.path.join(damaged_bearing_dir_audio_new_amr, file) for file in os.listdir(damaged_bearing_dir_audio_new_amr) if file.endswith('.WAV')],
    key=sort_key
)
good_bearing_files_accel_new_amr = sorted(
    [os.path.join(good_bearing_dir_accel_new_amr, file) for file in os.listdir(good_bearing_dir_accel_new_amr) if file.endswith('.csv')],
    key=sort_key
)
damaged_bearing_files_accel_new_amr = sorted(
    [os.path.join(damaged_bearing_dir_accel_new_amr, file) for file in os.listdir(damaged_bearing_dir_accel_new_amr) if file.endswith('.csv')],
    key=sort_key
)

# Combine audio files
good_bearing_files_audio = good_bearing_files_audio_s + good_bearing_files_audio_m + good_bearing_files_audio_new_amr
damaged_bearing_files_audio = damaged_bearing_files_audio_s + damaged_bearing_files_audio_m + damaged_bearing_files_audio_new_amr

# Combine accelerometer files
good_bearing_files_acel = good_bearing_files_acel_s + good_bearing_files_acel_m + good_bearing_files_accel_new_amr
damaged_bearing_files_acel = damaged_bearing_files_acel_s + damaged_bearing_files_acel_m + damaged_bearing_files_accel_new_amr

# Ensure sort order
good_bearing_files_audio = sorted(good_bearing_files_audio, key=sort_key)
damaged_bearing_files_audio = sorted(damaged_bearing_files_audio, key=sort_key)
good_bearing_files_acel = sorted(good_bearing_files_acel, key=sort_key)
damaged_bearing_files_acel = sorted(damaged_bearing_files_acel, key=sort_key)


# Extract features for each file and combine them
combined_features = []
labels = []

# Create a string based on the methods that are on
methods_string = "_".join(method for method, value in config.preprocessing_options.items() if value)

count = 0
max_count = min(len(good_bearing_files_audio), len(damaged_bearing_files_audio))

start_time_pre_proc = time.time()

# Process good_bearing files
for audio_file, accel_file in zip(good_bearing_files_audio, good_bearing_files_acel):
    audio_features = ppf.extract_audio_features(audio_file, noise_profile_file, config.preprocessing_options)
    accel_features = ppf.extract_accel_features(accel_file)
    combined = {**audio_features, **accel_features}
    combined_features.append(combined)
    labels.append(0)  # 0 for good_bearing
    count += 1
    if config.force_balanced_dataset and count == max_count:
        break

n_samples_healthy = len(combined_features)
print(f"Number of samples (Healthy Bearing): {n_samples_healthy}")

count = 0
# Process damaged_bearing files
for audio_file, accel_file in zip(damaged_bearing_files_audio, damaged_bearing_files_acel):
    audio_features = ppf.extract_audio_features(audio_file, noise_profile_file, config.preprocessing_options)
    accel_features = ppf.extract_accel_features(accel_file)
    combined = {**audio_features, **accel_features}
    combined_features.append(combined)
    labels.append(1)  # 1 for damaged_bearing
    count += 1
    if config.force_balanced_dataset and count == max_count:
        break

end_time_pre_proc = time.time()
    
n_samples_damaged = len(combined_features) - n_samples_healthy
print(f"Number of samples (Damaged Bearing): {n_samples_damaged}")

# Create DataFrame
combined_features_df = pd.DataFrame(combined_features)

# Normalize features
scaler = StandardScaler()
combined_features_normalized = scaler.fit_transform(combined_features_df)

print(f"Combined Features Shape: {combined_features_normalized.shape}")

# Convert labels to numpy array
y = np.array(labels)

# Train classifier
X_train, X_test, y_train, y_test = train_test_split(combined_features_normalized, y, test_size=0.2, random_state=42)

if(config.model['GBDT']):

    clf = tfdf.keras.GradientBoostedTreesModel(
        task=tfdf.keras.Task.CLASSIFICATION, 
        num_trees=config.model_params_GBDT['num_trees'], 
        growing_strategy=config.model_params_GBDT['growing_strategy'], 
        max_depth=config.model_params_GBDT['max_depth'],
        early_stopping=config.model_params_GBDT['early_stopping'],
        verbose=1
    )
    clf.compile(metrics=["accuracy"])
    Folder = 'GBDT'
    model = 'gbdt'
    
    
elif (config.model['RF']):

    clf = tfdf.keras.RandomForestModel(
        task=tfdf.keras.Task.CLASSIFICATION, 
        num_trees=config.model_params_RF['num_trees'], 
        growing_strategy=config.model_params_RF['growing_strategy'], 
        max_depth=config.model_params_RF['max_depth'],
        verbose=1
    )
    #clf.compile(metrics=["accuracy"])
    Folder = 'RF'
    model = 'rf'
    
model_save_folder = os.path.join(current_dir, "DTs_Models", Folder)
model_name = model + '_' + methods_string
model_save_path = os.path.join(model_save_folder, model_name)

if(os.path.exists(model_save_path) and config.model_load):
    print("Model already exists")
    clf = tf.saved_model.load(model_save_path)

else:
    history = clf.fit(X_train, y_train, validation_split=0.1, return_dict=True)
    history_dict = history.history

    clf.summary()

    print(history_dict.keys())

    # Save the model
    if(config.save_model):
        os.makedirs(model_save_folder, exist_ok=True)
        tf.saved_model.save(clf, model_save_path)

    # The training logs
    logs = clf.make_inspector().training_logs()

    # Final training accuracy and loss
    final_training_accuracy = logs[-1].evaluation.accuracy
    final_training_loss = logs[-1].evaluation.loss
    print(f"Final Training Accuracy: {final_training_accuracy:.4f}")
    print(f"Final Training Loss: {final_training_loss:.4f}")

# Evaluate the model
start_time_test_set = time.time()
evaluation = clf.evaluate(X_test, y_test, return_dict=True)
end_time_test_set = time.time()
print("\n")

#for name, value in evaluation.items():
#  print(f"{name}: {value:.4f}")

print('Test accuracy:', evaluation)

# Add loss to the report
test_loss = evaluation['loss']
print(f'Test Loss: {test_loss:.4f}')

tfdf.model_plotter.plot_model_in_colab(clf, tree_idx=0, max_depth=3)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy (out-of-bag)")

plt.subplot(1, 2, 2)
plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Logloss (out-of-bag)")

plt.tight_layout()

# Save the combined plot
if(config.save_metrics):    
    results_plot_folder = os.path.join(current_dir, 'DT_Results', 'NEW_AMR', Folder)
    results_plot_path = os.path.join(current_dir, 'DT_Results', 'NEW_AMR', Folder, model + '_acc_loss_' + methods_string + '.svg')
    os.makedirs(results_plot_folder, exist_ok=True)
    plt.savefig(results_plot_path, format='svg')

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

# Print the inference time
pre_proc_time = end_time_pre_proc - start_time_pre_proc
average_pre_proc_time = pre_proc_time / len(combined_features)
inference_time = end_time_test_set - start_time_test_set
average_inference_time = inference_time / len(y_test)
print("\n")
print("Time Metrics:")
print(f"Pre-processing Time: {pre_proc_time:.4f} seconds")
print(f"Average Pre-processing Time: {average_pre_proc_time:.4f} seconds")
print(f"Inference Time: {inference_time:.4f} seconds")
print(f"Average Inference Time: {average_inference_time:.4f} seconds")

print("\n")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Classification report
print("Classification Report:")
target_names = ['HEALTHY', 'DAMAGED']
print(classification_report(y_test, y_pred, target_names=['DAMAGED', 'HEALTHY']))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()

# Save the residual plot
if(config.save_metrics):
    results_plot_folder = os.path.join(current_dir, 'DT_Results', 'NEW_AMR', Folder)
    results_plot_path = os.path.join(current_dir, 'DT_Results', 'NEW_AMR', Folder, model + '_conf_matrix_' + methods_string + '.svg')
    os.makedirs(results_plot_folder, exist_ok=True)
    plt.savefig(results_plot_path, format='svg')

plt.show()

# Save the classification report

# Gather all metrics
metrics_dict = {
    'Metric': ['Precision', 'Recall', 'F1 Score', 'Accuracy', 'Loss',
               'Average Pre-processing Time','Average Inference Time'],
    'Value': [precision, recall, f1, accuracy, test_loss,
              average_pre_proc_time, average_inference_time]
}
# Save Metrics
if(config.save_metrics):
    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_dict)

    # Save DataFrame to CSV
    metrics_save_folder = os.path.join(current_dir, 'DT_Results', 'NEW_AMR', Folder)
    metrics_save_path = os.path.join(current_dir, 'DT_Results', 'NEW_AMR', Folder, model + '_metrics_' + methods_string + '.csv')
    os.makedirs(metrics_save_folder, exist_ok=True)
    metrics_df.to_csv(metrics_save_path, index=False)

    print("\nMetrics saved to CSV:")
    print(metrics_df)