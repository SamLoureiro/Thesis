import os
import time
import numpy as np
import pandas as pd
import librosa
import noisereduce as nr
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
import scipy.signal
import tensorflow_decision_forests as tfdf
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import config  # Import the config file
#from imblearn.over_sampling import SMOTE
#from sklearn.feature_selection import SelectKBest, f_classif
#from sklearn.model_selection import GridSearchCV

# Define directories
current_dir = os.getcwd()

good_bearing_dir_audio_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'GOOD', 'AUDIO')
damaged_bearing_dir_audio_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'DAMAGED', 'AUDIO')
good_bearing_dir_acel_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'GOOD', 'ACEL')
damaged_bearing_dir_acel_s = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'DAMAGED', 'ACEL')

# Define noise profile file
noise_profile_file = os.path.join(current_dir, 'Dataset_Piso', 'Noise.WAV')

# Helper function to sort files by the numeric part in the filename
def sort_key(file_path):
    file_name = os.path.basename(file_path)
    numeric_part = ''.join(filter(str.isdigit, file_name))
    return int(numeric_part) if numeric_part else 0

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

# Combine audio files
good_bearing_files_audio = good_bearing_files_audio_s
damaged_bearing_files_audio = damaged_bearing_files_audio_s 

# Combine accelerometer files
good_bearing_files_acel = good_bearing_files_acel_s
damaged_bearing_files_acel = damaged_bearing_files_acel_s 

# Ensure sort order
good_bearing_files_audio = sorted(good_bearing_files_audio, key=sort_key)
damaged_bearing_files_audio = sorted(damaged_bearing_files_audio, key=sort_key)
good_bearing_files_acel = sorted(good_bearing_files_acel, key=sort_key)
damaged_bearing_files_acel = sorted(damaged_bearing_files_acel, key=sort_key)

# Define feature extraction functions
def extract_audio_features(file_path, noise_profile, options):
    y, sr = librosa.load(file_path, sr=192000)
    
    if options['noise_reduction']:
        y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_profile, prop_decrease=config.noise_reduction_params['prop_decrease'], 
                            n_fft=config.noise_reduction_params['n_fft'], hop_length=config.noise_reduction_params['hop_length'])

    epsilon = 1e-10
    
    features = {}
    
    if options['basics']:
        features = {
            'mean': np.mean(y),
            'std': np.std(y),
            'rms': np.sqrt(np.mean(y**2)),
            'kurtosis': kurtosis(y) if np.std(y) > epsilon else 0,
            'skew': skew(y) if np.std(y) > epsilon else 0
        }

    if options['fft']:
        fft_result = np.abs(np.fft.fft(y, n=config.fft_params['n_fft']))
        freqs = np.fft.fftfreq(config.fft_params['n_fft'], 1/sr)
        fft_range = fft_result[(freqs >= config.fft_params['fmin']) & (freqs <= config.fft_params['fmax'])]
        fft_range = fft_range[:config.fft_params['n_fft']]

        for i, value in enumerate(fft_range):
            features[f'fft_{config.fft_params["fmin"]}_{config.fft_params["fmax"]}_{i}'] = value

    if options['mfcc']:
        audio_data = np.append(y[0], y[1:] - 0.97 * y[:-1])
        # Short-Time Fourier Transform (STFT)
        stft = librosa.stft(audio_data, n_fft=config.mfcc_params['n_fft'], hop_length=config.mfcc_params['hop_length'])
        # Power spectrum
        spectrogram = np.abs(stft)**2
                
        n_mels = config.mfcc_params['n_mels']
        
        mfcc_computed = False
        
        # Find the number of Mel filter banks that can be computed without any empty filters
        # Unccomment the following code if the samples proprietaries are not known, or the pre-processing parameters were changed
        '''while not mfcc_computed:
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    # Mel-frequency filter bank
                    mel_filterbank = librosa.filters.mel(sr=sr, n_fft=config.mfcc_params['n_fft'],n_mels=n_mels, fmin=config.mfcc_params['fmin'], fmax=config.mfcc_params['fmax'])
                    
                    if len(w) > 0 and any("Empty filters detected" in str(warning.message) for warning in w):
                        raise librosa.util.exceptions.ParameterError("Empty filters detected")
                    mfcc_computed = True
            
            except librosa.util.exceptions.ParameterError as e:
                if 'Empty filters detected' in str(e):
                    n_mels -= 1  # Reduce n_mels and try again
                    print(f"Reducing n_mels to {n_mels} and retrying...")
                    if n_mels < 1:
                        raise ValueError("Unable to compute MFCCs with given parameters.")
                else:
                    raise  # Re-raise any other exceptions'''
                
        # Mel-frequency filter bank
        # Comment the following line if the samples proprietaries are not known, or the pre-processing parameters were changed
        mel_filterbank = librosa.filters.mel(sr=sr, n_fft=config.mfcc_params['n_fft'],n_mels=n_mels, fmin=config.mfcc_params['fmin'], fmax=config.mfcc_params['fmax'])
        
        # Apply Mel filterbank
        mel_spectrogram = np.dot(mel_filterbank, spectrogram)

        # Convert to dB scale
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # MFCCs
        mfccs = librosa.feature.mfcc(S=mel_spectrogram, n_mfcc=40)
        
        for i in range(mfccs.shape[0]):
            avg = np.mean(mfccs[i, :])
            std = np.std(mfccs[i, :])
            features[f'mfcc_avg_{i}'] = avg
            features[f'mfcc_std_{i}'] = std

        return features

    if options['stft']:
        stft = np.abs(librosa.stft(y, n_fft=config.stft_params['n_fft'], hop_length=config.stft_params['hop_length']))
        stft_mean = np.mean(stft, axis=1)
        stft_std = np.std(stft, axis=1)
        for i in range(len(stft_mean)):
            features[f'stft_mean_{i}'] = stft_mean[i]
            features[f'stft_std_{i}'] = stft_std[i]

    return features

def extract_accel_features(file_path):
    epsilon = 1e-10
    df = pd.read_csv(file_path)
    features = {}
    for prefix in ['accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ']:
        for side in ['l', 'r']:
            column = f'{prefix}_{side}'
            data = df[column]
            std_dev = np.std(data)
            features[f'{column}_mean'] = np.mean(data)
            features[f'{column}_std'] = std_dev
            features[f'{column}_rms'] = np.sqrt(np.mean(data**2))
            features[f'{column}_kurtosis'] = kurtosis(data) if std_dev > epsilon else 0
            features[f'{column}_skew'] = skew(data) if std_dev > epsilon else 0
    return features

# Extract features for each file and combine them
combined_features = []
labels = []

count = 0
max_count = min(len(good_bearing_files_audio), len(damaged_bearing_files_audio))

# Create a string based on the methods that are on
methods_string = "_".join(method for method, value in config.preprocessing_options.items() if value)

start_time_pre_proc = time.time()

# Process good_bearing files
for audio_file, accel_file in zip(good_bearing_files_audio, good_bearing_files_acel):
    audio_features = extract_audio_features(audio_file, noise_profile_file, config.preprocessing_options)
    accel_features = extract_accel_features(accel_file)
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
    #print(audio_file)
    #print(accel_file)
    audio_features = extract_audio_features(audio_file, noise_profile_file, config.preprocessing_options)
    accel_features = extract_accel_features(accel_file)
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

# Shuffle the data and labels
combined_features_normalized, y = shuffle(combined_features_normalized, y, random_state=42)

# Train classifier
X_train, X_test, y_train, y_test = train_test_split(combined_features_normalized, y, test_size=0.2, random_state=42)

if(config.model['GBDT']):

    clf = tfdf.keras.GradientBoostedTreesModel(
        task=tfdf.keras.Task.CLASSIFICATION, 
        num_trees=config.model_params_GBDT['num_trees'], 
        growing_strategy=config.model_params_GBDT['growing_strategy'], 
        max_depth=config.model_params_GBDT['max_depth'],
        early_stopping=config.model_params_GBDT['early_stopping']
    )
    
    Folder = 'GBDT'
    model = 'gbdt'

elif (config.model['RF']):

    clf = tfdf.keras.RandomForestModel(
        task=tfdf.keras.Task.CLASSIFICATION, 
        num_trees=config.model_params_RF['num_trees'], 
        growing_strategy=config.model_params_RF['growing_strategy'], 
        max_depth=config.model_params_RF['max_depth']
    )
    
    Folder = 'RF'
    model = 'rf'


#adamw_optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)

clf.fit(X_train, y_train)

clf.compile(loss='binary_crossentropy', metrics=["accuracy"])

#clf.summary()

# Save the model
if(config.save_model):
    model_save_path = os.path.join(current_dir, "saved_model")
    clf.save(model_save_path)
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

# Save the residual plot
if(config.save_metrics):
    results_plot_path = os.path.join(current_dir, 'Results', 'AMR_STOPPED', Folder, model + '_acc_loss_' + methods_string + '_2048.svg')
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

#print(f"Number of good bearings (0) in the test set: {label_counts.get(0, 0)}")
#print(f"Number of damaged bearings (1) in the test set: {label_counts.get(1, 0)}")

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
    results_plot_path = os.path.join(current_dir, 'Results', 'AMR_STOPPED', Folder, model + '_conf_matrix_' + methods_string + '_2048.svg')
    plt.savefig(results_plot_path, format='svg')
    
plt.show()

# Save the classification report

# Gather all metrics
metrics_dict = {
    'Metric': ['Precision', 'Recall', 'F1 Score', 'Accuracy', 'Loss', 'Training Accuracy', 'Training Loss',
               'Average Pre-processing Time','Average Inference Time'],
    'Value': [precision, recall, f1, accuracy, test_loss, final_training_accuracy, final_training_loss,
              average_pre_proc_time, average_inference_time]
}

if(config.save_metrics):
    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_dict)

    # Save DataFrame to CSV
    metrics_save_path = os.path.join(current_dir, 'Results', 'AMR_STOPPED', Folder, model + '_metrics_' + methods_string + '_2048.csv')
    metrics_df.to_csv(metrics_save_path, index=False)

    print("\nMetrics saved to CSV:")
    print(metrics_df)
