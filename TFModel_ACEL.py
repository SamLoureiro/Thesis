import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import os
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.utils import class_weight, shuffle
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

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

current_dir = os.getcwd()
data_file = os.path.join(current_dir, 'Dataset_Piso', 'MERGED', 'ALL_merged.csv')
dataset = read_data(data_file)

# Normalize dataset
for col in ['accX_l', 'accY_l', 'accZ_l', 'gyrX_l', 'gyrY_l', 'gyrZ_l', 'accX_r', 'accY_r', 'accZ_r', 'gyrX_r', 'gyrY_r', 'gyrZ_r']:
    dataset[col] = feature_normalize(dataset[col])

def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size / 2)

def segment_signal(data, window_size=100):
    segments = np.empty((0, window_size, 12))
    labels = np.empty((0))
    for (start, end) in windows(data["timestamp"], window_size):
        ax_l = data["accX_l"][start:end]
        ay_l = data["accY_l"][start:end]
        az_l = data["accZ_l"][start:end]
        gx_l = data["gyrX_l"][start:end]
        gy_l = data["gyrY_l"][start:end]
        gz_l = data["gyrZ_l"][start:end]
        ax_r = data["accX_r"][start:end]
        ay_r = data["accY_r"][start:end]
        az_r = data["accZ_r"][start:end]
        gx_r = data["gyrX_r"][start:end]
        gy_r = data["gyrY_r"][start:end]
        gz_r = data["gyrZ_r"][start:end]
        if len(dataset["timestamp"][start:end]) == window_size:
            segments = np.vstack([segments, np.dstack([ax_l, ay_l, az_l, gx_l, gy_l, gz_l, ax_r, ay_r, az_r, gx_r, gy_r, gz_r])])
            unique, counts = np.unique(data["piso"][start:end], return_counts=True)
            labels = np.append(labels, unique[np.argmax(counts)])
    return segments, labels

segments, labels = segment_signal(dataset)

# Print shape of segments
print(f'Segments shape before reshaping: {segments.shape}')

# Ensure the reshape dimensions match the number of elements
num_segments = len(segments)
reshaped_segments = segments.reshape(num_segments, 1, segments.shape[1], segments.shape[2])

print(f'Segments shape after reshaping: {reshaped_segments.shape}')
print(f'Labels shape: {labels.shape}')

# Segments shape before reshaping: (951, 100, 12)
# Segments shape after reshaping: (951, 1, 100, 12)
# Labels shape: (951,)

# 951: This represents the number of segments obtained from the original signal. There are 951 windows (segments) created from the entire time series data.
# 1: A new dimension of size 1 is introduced at the beginning. This dimension typically represents the number of samples or sequences within a segment. In 
# this case, each segment is considered a single sample for the CNN model.
# 100: This indicates the size of each segment. It likely represents the number of data points per segment (window size) extracted from the original signal.
# 12: This signifies the number of features or channels in each data point. There are 12 features extracted from each data point, likely corresponding to 
# different sensor readings (e.g., accelerometer axes, gyroscope axes for x, y, and z directions).

input_height = 1
input_width = 100
num_labels = len(np.unique(labels))
num_channels = 12

batch_size = 32
kernel_size = 60
depth = 60
num_hidden = 1000

learning_rate = 0.0001
training_epochs = 40

total_batches = reshaped_segments.shape[0] // batch_size

# Shuffle the data
reshaped_segments, labels = shuffle(reshaped_segments, labels, random_state=42)

# Compute class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))

# Convert labels to categorical
labels_categorical = np.asarray(pd.get_dummies(labels), dtype=np.int8)

# Model definition
model = Sequential([
    Conv2D(filters=depth, kernel_size=(1, kernel_size), activation='relu', input_shape=(input_height, input_width, num_channels)), #kernel_regularizer=l2(0.01)
    BatchNormalization(),
    MaxPooling2D(pool_size=(1, 20), strides=(1, 2)),
    Dropout(0.5),
    Conv2D(filters=depth//10, kernel_size=(1, 6), activation='relu'), #kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Flatten(),
    Dropout(0.5),
    Dense(num_hidden, activation='tanh', kernel_regularizer=l2(0.01)),
    Dense(num_labels, activation='softmax')
])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
history = model.fit(reshaped_segments, labels_categorical, epochs=training_epochs, batch_size=batch_size, validation_split=0.2, class_weight=class_weights, callbacks=[reduce_lr, early_stopping])

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