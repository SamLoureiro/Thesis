import os
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

good_bearing_dir_audio_m = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'GOOD', 'AUDIO')
damaged_bearing_dir_audio_m = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'DAMAGED', 'AUDIO')
good_bearing_dir_acel_m = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'GOOD', 'ACEL')
damaged_bearing_dir_acel_m = os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'DAMAGED', 'ACEL')

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

# Combine audio files
good_bearing_files_audio = good_bearing_files_audio_m + good_bearing_files_audio_s
damaged_bearing_files_audio = damaged_bearing_files_audio_m + damaged_bearing_files_audio_s

# Combine accelerometer files
good_bearing_files_acel = good_bearing_files_acel_m + good_bearing_files_acel_s
damaged_bearing_files_acel = damaged_bearing_files_acel_m + damaged_bearing_files_acel_s

# Ensure sort order
good_bearing_files_audio = sorted(good_bearing_files_audio, key=sort_key)
damaged_bearing_files_audio = sorted(damaged_bearing_files_audio, key=sort_key)
good_bearing_files_acel = sorted(good_bearing_files_acel, key=sort_key)
damaged_bearing_files_acel = sorted(damaged_bearing_files_acel, key=sort_key)

# Define feature extraction functions
def extract_audio_features(file_path, noise_profile, options):
    y, sr = librosa.load(file_path, sr=192000)
    
    if options['noise_reduction']:
        y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_profile, n_fft=config.noise_reduction_params['n_fft'], hop_length=config.noise_reduction_params['hop_length'])

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
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_fft=config.mfcc_params['n_fft'], hop_length=config.mfcc_params['hop_length'], 
                                     window=scipy.signal.get_window('hamming', config.mfcc_params['n_fft']), htk=False, 
                                     center=True, pad_mode='reflect', power=2.0, 
                                     n_mels=config.mfcc_params['n_mels'], fmin=config.mfcc_params['fmin'], fmax=config.mfcc_params['fmax'], n_mfcc=config.mfcc_params['n_mfcc'])
        for i in range(mfccs.shape[0]):
            avg = np.mean(mfccs[i, :])
            std = np.std(mfccs[i, :])
            features[f'mfcc_avg_{i}'] = avg
            features[f'mfcc_std_{i}'] = std

    if options['stft']:
        stft = np.abs(librosa.stft(y, n_fft=config.stft_params['n_fft'], hop_length=config.stft_params['hop_length'], win_length=config.stft_params['win_length']))
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

clf = tfdf.keras.GradientBoostedTreesModel(
    task=tfdf.keras.Task.CLASSIFICATION, 
    num_trees=config.model_params['num_trees'], 
    growing_strategy=config.model_params['growing_strategy'], 
    max_depth=config.model_params['max_depth']
)

#adamw_optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)

clf.fit(X_train, y_train)

clf.compile(loss='binary_crossentropy', metrics=["accuracy"])

clf.summary()

# Save the model
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
evaluation = clf.evaluate(X_test, y_test, return_dict=True)

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
results_plot_path = os.path.join(current_dir, 'Results', 'FULL_DATASET', 'acc_loss_fft_noise_basics_stft_mfcc.svg')
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
results_plot_path = os.path.join(current_dir, 'Results', 'FULL_DATASET', 'conf_matrix_fft_noise_basics_stft_mfcc.svg')
plt.savefig(results_plot_path, format='svg')
plt.show()
