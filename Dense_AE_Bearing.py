import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, roc_curve, confusion_matrix, precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import config  # Import the config file
import PreProc_Function as ppf
from imblearn.over_sampling import SMOTE
from AE_Aux_Func import reduce_dimensions, plot_reduced_data, plot_metrics_vs_threshold, find_optimal_threshold_f1, save_metrics_to_csv

# For Future Use
#from sklearn.feature_selection import SelectKBest, f_classif
#from sklearn.model_selection import GridSearchCV

# Helper function to sort files by the numeric part in the filename
def sort_key(file_path):
    file_name = os.path.basename(file_path)
    numeric_part = ''.join(filter(str.isdigit, file_name))
    return int(numeric_part) if numeric_part else 0

def perform_evaluation_and_prediction(model, X_test, Folder, model_string, all_data):
    
    print("\nModel Description:")
    model.describe()
    
    evaluation = model.evaluate(X_test)
    
    print(model.benchmark(X_test))

    print("\n")
    print('Test evaluation:')
    print(evaluation)

    test_loss = evaluation.loss
    print(f'Test Loss: {test_loss:.4f}')    
    
    # Predict on the test set
    start_time_test_set = time.time()
    y_pred_probs = model.predict(X_test)
    end_time_test_set = time.time()
    
    # For late, when the dataset is labeled with the features names
    
    #print("\nPredictions Analyzed:")
    #model.analyze_prediction(X_test, sampling=0.1)
    
    print("\nVariable importances keys:")
    print(model.variable_importances().keys())
    
    print("\n10 most important features:")
    print(model.variable_importances()["SUM_SCORE"][:10])
    
    print("\n10 less important features:")
    print(model.variable_importances()["SUM_SCORE"][-10:])
    
    print("\nPredictions:")
    
    # Convert predicted probabilities to binary class labels
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Convert labels to numpy array 
    y_test = X_test['label'].values

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
        results_plot_path = os.path.join(current_dir, 'DT_Results', 'NEW_AMR', Folder, model_string + '_conf_matrix_' + methods_string + '.svg')
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
        metrics_save_path = os.path.join(current_dir, 'DT_Results', 'NEW_AMR', Folder, model_string + '_metrics_' + methods_string + '.csv')
        os.makedirs(metrics_save_folder, exist_ok=True)
        metrics_df.to_csv(metrics_save_path, index=False)

        print("\nMetrics saved to CSV:")
        print(metrics_df)
        
    # Prediction with all data    
    y_pred_probs = model.predict(all_data)
    
    # Convert predicted probabilities to binary class labels
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Convert labels to numpy array 
    y = all_data['label'].values
    
    # Classification report
    print("Classification Report with all the data (train and test):")
    target_names = ['HEALTHY', 'DAMAGED']
    print(classification_report(y, y_pred, target_names=['DAMAGED', 'HEALTHY']))

    # Confusion matrix
    conf_matrix = confusion_matrix(y, y_pred, labels=[0, 1])

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for All Data')
    plt.tight_layout()
    plt.show()

    return evaluation

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

# Convert labels to numpy array
y = np.array(labels)

# Convert features to a DataFrame
features_df = pd.DataFrame(combined_features_normalized, columns=combined_features_df.columns)

# Convert labels to a DataFrame
labels_df = pd.DataFrame(y, columns=["label"])

good_bearing_samples = features_df.loc[labels_df.index[labels_df['label'] == 0]]
good_bearing_labels = labels_df.loc[labels_df.index[labels_df['label'] == 0]]

damaged_bearing_samples = features_df.loc[labels_df.index[labels_df['label'] == 1]]
damaged_bearing_labels = labels_df.loc[labels_df.index[labels_df['label'] == 1]]

# Data Splitting
X_train_complete, X_test_good, y_train_complete, y_test_good  = train_test_split(good_bearing_samples, good_bearing_labels, test_size=0.2, random_state=47)

X_train, X_val_good, y_train, y_val_good = train_test_split(X_train_complete, y_train_complete, test_size=0.1, random_state=45)

#print(X_val_good.head(10))
#print(y_val_good.head(10))
#print(X_val_good.tail(10))
#print(y_val_good.tail(10))

X_val_damaged_complete, X_test_damaged, y_val_damaged_complete, y_test_damaged = train_test_split(damaged_bearing_samples, damaged_bearing_labels, test_size=0.5, random_state=51)

val_size = min(len(X_val_good), len(X_val_damaged_complete))

X_val_damaged = X_val_damaged_complete.iloc[:val_size]
y_val_damaged = y_val_damaged_complete.iloc[:val_size]

X_test = pd.concat([X_test_good, X_test_damaged.iloc[:len(X_test_good)], X_val_damaged.iloc[val_size:]])
y_test = pd.concat([y_test_good, y_test_damaged.iloc[:len(y_test_good)], y_val_damaged.iloc[val_size:]])

input_shape = X_train.shape[1]

epochs = 300
batch_size = 64

model_name = "DAE_stft.keras"

model_save_path = os.path.join(current_dir, 'AE_Models', 'NEW_AMR', 'Bayesian', model_name)

# Early stopping setup
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True, verbose=2)

# Load the pre-trained autoencoder
loaded_model = load_model(model_save_path)

# Freeze encoder layers
for layer in loaded_model.layers[:-6]:
    layer.trainable = False
    
# Compile the model
loaded_model.compile(optimizer='adam', loss='mse')

history = loaded_model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], validation_split=0.1, verbose=2)

#if(config.save_model):
#    loaded_model.save(model_save_path)        

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show() 


loaded_model.summary()

# Reconstruction and threshold finding
start_time = time.time()
val_predictions_good = loaded_model.predict(X_val_good)
inference_time = time.time() - start_time

# Get the reconstruction error on the validation set
val_errors_good = np.mean(np.square(X_val_good - val_predictions_good), axis=1)

# Get the reconstruction error on the labeled validation anomalies
val_predictions_damaged = loaded_model.predict(X_val_damaged)
val_errors_damaged = np.mean(np.square(X_val_damaged - val_predictions_damaged), axis=1)

print("Validation Errors lenght:")
print(len(val_errors_good))
print(len(val_errors_damaged))

# Plotting KDEs of reconstruction errors
plt.figure(figsize=(10, 6))

# Healthy errors
sns.kdeplot(val_errors_good, label="Healthy", fill=True, color='green', alpha=0.6)

# Damaged errors
sns.kdeplot(val_errors_damaged, label="Damaged", fill=True, color='red', alpha=0.6)

plt.xlabel("Reconstruction Error")
plt.ylabel("Density")
plt.title("Reconstruction Error Distribution")
plt.legend()
plt.show()

# Combine validation errors and bearing validation errors
combined_val_errors = np.concatenate([val_errors_damaged, val_errors_good])
combined_val_labels = np.concatenate([y_val_damaged, y_val_good])

'''# Calculate precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(combined_val_labels, combined_val_errors)

# Calculate the absolute difference between precision and recall
diff = abs(precision - recall)

# Find the index of the minimum difference
optimal_idx = np.argmin(diff)   

# Get the optimal threshold
optimal_threshold = thresholds[optimal_idx]'''

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(combined_val_labels, combined_val_errors)

# Find the optimal threshold
optimal_idx = np.argmax(tpr - fpr)  # This gives you the threshold with the maximum difference between TPR and FPR
optimal_threshold = thresholds[optimal_idx]

# Final evaluation on the test set
test_predictions_good = loaded_model.predict(X_test_good)
test_errors_good = np.mean(np.square(X_test_good - test_predictions_good), axis=1)

# Final evaluation on the test set
test_predictions_damaged = loaded_model.predict(X_test_damaged)
test_errors_damaged = np.mean(np.square(X_test_damaged - test_predictions_damaged), axis=1)

# Combine the noise and bearing errors
combined_test_errors = np.concatenate([test_errors_damaged, test_errors_good])
combined_test_labels = np.concatenate([y_test_damaged, y_test_good])

optimal_threshold = 0.3

# Determine which samples are anomalies based on the optimal threshold
test_anomalies = combined_test_errors > optimal_threshold

# Calculate and print the final detection metrics
predicted_labels = test_anomalies.astype(int)

print(classification_report(combined_test_labels, predicted_labels, target_names=["Noise", "Anomaly"]))

rec_error_good_avg = np.mean(test_errors_good)
rec_error_damaged_avg = np.mean(test_errors_damaged)
rec_error_good_std = np.std(test_errors_good)
rec_error_damaged_std = np.std(test_errors_damaged)

print("Reconstruction Errors:")
print("Average Reconstruction Error for Validation Data (only noise):")
print(f'Mean Reconstruction Error: {rec_error_good_avg}')
print(f'Standard Deviation of Reconstruction Error: {rec_error_good_std}')
print("Average Reconstruction Error for Validation Data (only bearings):")
print(f'Mean Reconstruction Error: {rec_error_damaged_avg}')
print(f'Standard Deviation of Reconstruction Error: {rec_error_damaged_std}')
print("\n")

acc = accuracy_score(combined_test_labels, predicted_labels)
prec = precision_score(combined_test_labels, predicted_labels)
rec = recall_score(combined_test_labels, predicted_labels)
f1 = f1_score(combined_test_labels, predicted_labels)
auc = roc_auc_score(combined_test_labels, predicted_labels)
inf_time = inference_time / len(X_val_good) * 1000
#proc_time = pre_proc_time / (len(y_train) + len(y_val)) * 1000

# Global Evaluation
print("Evaluation:")
print(f"Optimal Threshold: {optimal_threshold}")  
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"AUC: {auc:.3f}")
print(f"Average Inference Time per Sample: {inf_time:.3f} ms")
#print(f"Average Preprocessing Time per Sample: {proc_time:.3f} ms")

# Calculate the confusion matrix
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

# Thresholds vs metrics for validation data
thresholds = np.linspace(min(combined_val_errors), max(combined_val_errors), 100)
f1_scores_val = []
precisions_val = []
recalls_val = []
accuracy_val = []
roc_aucs_val = []

for threshold in thresholds:
    y_val_pred = (combined_val_errors > threshold).astype(int)  # Assuming combined_val_errors is for validation
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


