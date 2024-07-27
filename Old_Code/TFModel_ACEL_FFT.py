import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.utils import class_weight, shuffle
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pywt

plt.style.use('ggplot')

def read_data(file_path):
    data = pd.read_csv(file_path, header=0)
    return data

def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma

def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    plot_axis(ax0, data['timestamp'], data['accX_l'], 'accX_l')
    plot_axis(ax1, data['timestamp'], data['accY_l'], 'accY_l')
    plot_axis(ax2, data['timestamp'], data['accZ_l'], 'accZ_l')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

def compute_dwt_fixed_length(data, wavelet='db1', level=4, fixed_length=360):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    dwt_vector = np.hstack(coeffs)
    if len(dwt_vector) > fixed_length:
        return dwt_vector[:fixed_length]
    else:
        return np.pad(dwt_vector, (0, fixed_length - len(dwt_vector)), 'constant')


def segment_and_compute_dwt_fixed_length(data, window_size=1000, wavelet='db1', level=4, fixed_length=500):
    segments = np.empty((0, window_size, 12))
    dwt_segments = np.empty((0, 12 * fixed_length))
    labels = np.empty((0))
    for (start, end) in windows(data["timestamp"], window_size):
        if len(dataset["timestamp"][start:end]) == window_size:
            segment = np.zeros((window_size, 12))
            for i, col in enumerate(['accX_l', 'accY_l', 'accZ_l', 'gyrX_l', 'gyrY_l', 'gyrZ_l', 'accX_r', 'accY_r', 'accZ_r', 'gyrX_r', 'gyrY_r', 'gyrZ_r']):
                segment[:, i] = data[col][start:end]
            segments = np.vstack([segments, [segment]])

            # Compute fixed-length DWT for each axis and concatenate them
            dwt_feature_vector = np.concatenate([compute_dwt_fixed_length(data[col][start:end], wavelet, level, fixed_length) for col in ['accX_l', 'accY_l', 'accZ_l', 'gyrX_l', 'gyrY_l', 'gyrZ_l', 'accX_r', 'accY_r', 'accZ_r', 'gyrX_r', 'gyrY_r', 'gyrZ_r']])
            dwt_segments = np.vstack([dwt_segments, dwt_feature_vector])

            # Majority vote for the label in this window
            unique, counts = np.unique(data["piso"][start:end], return_counts=True)
            labels = np.append(labels, unique[np.argmax(counts)])
    return segments, dwt_segments, labels


def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size / 2)


# Load the dataset
current_dir = os.getcwd()
data_file = os.path.join(current_dir, 'Dataset_Piso', 'MERGED', 'ALL_merged.csv')
dataset = read_data(data_file)

# Normalize dataset
for col in ['accX_l', 'accY_l', 'accZ_l', 'gyrX_l', 'gyrY_l', 'gyrZ_l', 'accX_r', 'accY_r', 'accZ_r', 'gyrX_r', 'gyrY_r', 'gyrZ_r']:
    dataset[col] = feature_normalize(dataset[col])


fixed_length = 360  # This should be chosen based on experimentation

segments, dwt_segments, labels = segment_and_compute_dwt_fixed_length(dataset, fixed_length=fixed_length)

# Print shapes to verify
print(f'Segments shape: {segments.shape}')
print(f'DWT Segments shape: {dwt_segments.shape}')
print(f'Labels shape: {labels.shape}')

# Combine time-domain and frequency-domain features
combined_features = np.hstack((segments.reshape(len(segments), -1), dwt_segments))


# Ensure the reshape dimensions match the number of elements
num_segments = len(segments)
reshaped_combined_features = combined_features.reshape(num_segments, 1, combined_features.shape[1], 1)

print(f'Combined features shape after reshaping: {reshaped_combined_features.shape}')

input_height = 1
input_width = reshaped_combined_features.shape[2]
num_channels = 1  # Since combined features are flattened into a single dimension
num_labels = len(np.unique(labels))
num_channels = 1

batch_size = 32
kernel_size = 60
depth = 60
num_hidden = 1000

learning_rate = 0.0001
training_epochs = 40

total_batches = reshaped_combined_features.shape[0] // batch_size

# Shuffle the data
reshaped_segments, labels = shuffle(reshaped_combined_features, labels, random_state=42)

# Compute class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))

# Convert labels to categorical
labels_categorical = np.asarray(pd.get_dummies(labels), dtype=np.int8)


# Model definition
model = Sequential([
    Conv2D(filters=depth, kernel_size=(1, kernel_size), activation='relu', input_shape=(input_height, input_width, num_channels)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(1, 20), strides=(1, 2)),
    Dropout(0.5),
    Conv2D(filters=depth//10, kernel_size=(1, 6), activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dropout(0.5),
    Dense(num_hidden, activation='tanh', kernel_regularizer=l2(0.01)),
    Dense(num_labels, activation='softmax')
])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
history = model.fit(reshaped_combined_features, labels_categorical, epochs=training_epochs, batch_size=batch_size, validation_split=0.2, class_weight=class_weights, callbacks=[reduce_lr, early_stopping])

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate the model
predictions = model.predict(reshaped_segments)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(labels_categorical, axis=1)

# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Classification report
class_report = classification_report(true_labels, predicted_labels, target_names=[str(i) for i in np.unique(labels)])
print("Classification Report:\n", class_report)

# Calculate misclassifications
misclassifications = conf_matrix.sum(axis=1) - np.diag(conf_matrix)

# Plot misclassifications
plt.figure(figsize=(10, 6))
plt.bar(np.unique(labels), misclassifications)
plt.title('Misclassifications per Class')
plt.xlabel('Class')
plt.ylabel('Number of Misclassifications')
plt.show()