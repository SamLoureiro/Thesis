import tensorflow as tf

from keras.layers import (Input, Dense, BatchNormalization, Dropout, LSTM, 
                          RepeatVector, TimeDistributed, Bidirectional, Input, Conv1D, MaxPooling1D, UpSampling1D, Cropping1D, ZeroPadding1D, Add, GlobalMaxPooling1D, 
                          Flatten, Reshape, ConvLSTM1D, MultiHeadAttention, LayerNormalization, Lambda)
from keras_tuner import HyperModel
from keras.regularizers import L2, l1_l2
from keras.models import Model, Sequential
from keras.losses import mean_squared_error, mean_squared_logarithmic_error, mean_absolute_error
import numpy as np

mse = mean_squared_error
msle = mean_squared_logarithmic_error
mae = mean_absolute_error


'''

Models to test in the main file

# Input Shape of each sample: (50, 2143) - where:

# 50 - timestamps of each sample
# 2143 - features per timestamp

CNN Simple gave the best results so far, even though they are not good enough compared to the Dense Model with the pre-processing from Supervised Learning

'''

def RNN_Complex(input_shape):
    """Build and compile an RNN autoencoder model."""
    inputs = Input(shape=input_shape)
    
    # Encoder
    x = Bidirectional(LSTM(128, activation='relu', return_sequences=True))(inputs)
    # Bidirectional LSTM layer to capture dependencies in both forward and backward directions
    x = Dropout(0.2)(x)  # Dropout to prevent overfitting
    x = BatchNormalization()(x)  # Normalizes the activations of the previous layer
    
    x = Bidirectional(LSTM(64, activation='relu', return_sequences=True))(x)
    # Another Bidirectional LSTM layer with fewer units
    x = Dropout(0.2)(x)  # Dropout to prevent overfitting
    x = BatchNormalization()(x)  # Normalizes the activations of the previous layer
    
    encoded = LSTM(32, activation='relu', return_sequences=False)(x)
    # Final LSTM layer in the encoder, reducing the sequence to a single vector
    encoded = Dropout(0.2)(encoded)  # Dropout to prevent overfitting
    encoded = BatchNormalization()(encoded)  # Normalizes the activations of the previous layer
    
    # Bottleneck
    bottleneck = Dense(16, activation='relu')(encoded)  # Dense layer to reduce dimensionality further
    bottleneck = RepeatVector(input_shape[0])(bottleneck)
    # Repeats the bottleneck vector across the time dimension to prepare for decoding
    
    # Decoder
    x = LSTM(32, activation='relu', return_sequences=True)(bottleneck)
    # LSTM layer to start reconstructing the sequence
    x = Dropout(0.2)(x)  # Dropout to prevent overfitting
    x = BatchNormalization()(x)  # Normalizes the activations of the previous layer
    
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    # Another LSTM layer to further decode the sequence
    x = Dropout(0.2)(x)  # Dropout to prevent overfitting
    x = BatchNormalization()(x)  # Normalizes the activations of the previous layer
    
    decoded = LSTM(128, activation='relu', return_sequences=True)(x)
    # Final LSTM layer to reconstruct the original sequence shape
    outputs = TimeDistributed(Dense(input_shape[1]))(decoded)
    # TimeDistributed layer applies Dense to each time step individually, reconstructing the original feature space
    
    autoencoder = Model(inputs, outputs)  # Model initialization
    autoencoder.compile(optimizer='adam', loss='mse')  # Compile model with MSE loss
    
    return autoencoder

def RNN_Simple(input_shape):
    """Build and compile a simpler RNN autoencoder model."""
    inputs = Input(shape=input_shape)
    
    # Encoder
    x = Bidirectional(LSTM(64, activation='relu', return_sequences=True))(inputs)
    # Bidirectional LSTM layer to capture dependencies in both forward and backward directions
    x = Dropout(0.2)(x)  # Dropout to prevent overfitting
    x = BatchNormalization()(x)  # Normalizes the activations of the previous layer
    
    encoded = LSTM(32, activation='relu', return_sequences=False)(x)
    # Single LSTM layer in the encoder, reducing the sequence to a single vector
    encoded = Dropout(0.2)(encoded)  # Dropout to prevent overfitting
    encoded = BatchNormalization()(encoded)  # Normalizes the activations of the previous layer
    
    # Bottleneck
    bottleneck = Dense(16, activation='relu')(encoded)  # Dense layer to reduce dimensionality further
    bottleneck = RepeatVector(input_shape[0])(bottleneck)
    # Repeats the bottleneck vector across the time dimension to prepare for decoding
    
    # Decoder
    x = LSTM(32, activation='relu', return_sequences=True)(bottleneck)
    # LSTM layer to start reconstructing the sequence
    x = Dropout(0.2)(x)  # Dropout to prevent overfitting
    x = BatchNormalization()(x)  # Normalizes the activations of the previous layer
    
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    # Another LSTM layer to further decode the sequence
    x = Dropout(0.2)(x)  # Dropout to prevent overfitting
    x = BatchNormalization()(x)  # Normalizes the activations of the previous layer
    
    decoded = LSTM(128, activation='relu', return_sequences=True)(x)
    # Final LSTM layer to reconstruct the original sequence shape
    outputs = TimeDistributed(Dense(input_shape[1]))(decoded)
    # TimeDistributed layer applies Dense to each time step individually, reconstructing the original feature space
    
    autoencoder = Model(inputs, outputs)  # Model initialization
    autoencoder.compile(optimizer='adam', loss='mse')  # Compile model with MSE loss
    
    return autoencoder

def DEEP_CNN(input_shape):
    inputs = Input(shape=input_shape)
    
    # Encoder
    x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=L2(0.001))(inputs)
    # First convolutional layer with L2 regularization
    x = Dropout(0.2)(x)  # Dropout to prevent overfitting
    x = MaxPooling1D(2, padding='same')(x)  # Max pooling to reduce the sequence length by half
    x = BatchNormalization()(x)  # Normalizes the activations of the previous layer
    
    x = Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=L2(0.001))(x)
    # Second convolutional layer with more filters
    x = Dropout(0.2)(x)  # Dropout to prevent overfitting
    x = MaxPooling1D(2, padding='same')(x)  # Max pooling to reduce the sequence length by half
    x = BatchNormalization()(x)  # Normalizes the activations of the previous layer

    x = Conv1D(256, 3, activation='relu', padding='same', kernel_regularizer=L2(0.001))(x)
    # Third convolutional layer with even more filters
    x = Dropout(0.2)(x)  # Dropout to prevent overfitting
    x = MaxPooling1D(2, padding='same')(x)  # Max pooling to reduce the sequence length by half
    x = BatchNormalization()(x)  # Normalizes the activations of the previous layer
    
    # Residual connection
    residual = Conv1D(512, 1, padding='same')(x)  # 1x1 convolution to adjust dimensions for residual connection
    x = Conv1D(512, 3, activation='relu', padding='same', kernel_regularizer=L2(0.001))(x)
    # Fourth convolutional layer with 512 filters
    x = Dropout(0.2)(x)  # Dropout to prevent overfitting
    x = Add()([x, residual])  # Add residual connection (skip connection) to improve gradient flow
    
    # Bottleneck
    x = GlobalMaxPooling1D()(x)  # Global max pooling to reduce each feature map to a single value
    encoded = Dense(256, activation='relu')(x)  # Dense layer to further reduce dimensionality
    encoded = Dropout(0.3)(encoded)  # Dropout to prevent overfitting
    bottleneck = Dense(64, activation='relu')(encoded)  # Final dense layer to create the bottleneck
    
    # Decoder
    x = Dense(256, activation='relu')(bottleneck)  # Dense layer to expand dimensionality back
    x = Dropout(0.2)(x)  # Dropout to prevent overfitting
    x = RepeatVector(input_shape[0])(x)  # Repeat the vector across the time dimension
    
    x = Conv1D(512, 3, activation='relu', padding='same')(x)  # Deconvolution to expand back to original dimensions
    x = BatchNormalization()(x)  # Normalizes the activations of the previous layer
    x = Dropout(0.2)(x)  # Dropout to prevent overfitting
    
    x = Conv1D(256, 3, activation='relu', padding='same')(x)  # Deconvolution with fewer filters
    x = BatchNormalization()(x)  # Normalizes the activations of the previous layer
    x = Dropout(0.2)(x)  # Dropout to prevent overfitting
    
    x = Conv1D(128, 3, activation='relu', padding='same')(x)  # Deconvolution with fewer filters
    x = BatchNormalization()(x)  # Normalizes the activations of the previous layer
    x = Dropout(0.2)(x)  # Dropout to prevent overfitting
    
    x = Conv1D(64, 3, activation='relu', padding='same')(x)  # Deconvolution with fewer filters
    x = BatchNormalization()(x)  # Normalizes the activations of the previous layer
    x = Dropout(0.2)(x)  # Dropout to prevent overfitting
    
    # Adjust for mismatch in temporal dimension
    crop_amount = (x.shape[1] - input_shape[0])
    if crop_amount > 0:
        x = Cropping1D(cropping=(crop_amount // 2, crop_amount - crop_amount // 2))(x)  # Crop symmetrically
    elif crop_amount < 0:
        x = ZeroPadding1D(padding=(-crop_amount // 2, -crop_amount + (-crop_amount // 2)))(x)  # Pad symmetrically
    
    decoded = Conv1D(input_shape[-1], 3, activation='sigmoid', padding='same')(x)  # Output layer
    
    autoencoder = Model(inputs, decoded)  # Model initialization
    autoencoder.compile(optimizer='adam', loss='mse')  # Compile model with MSE loss
    
    return autoencoder

def CNN_Simple(input_shape):
    inputs = Input(shape=input_shape)
    
    # Encoder
    x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    # First convolutional layer
    x = MaxPooling1D(2, padding='same')(x)  # Max pooling to reduce the sequence length by half
    x = BatchNormalization()(x)  # Normalizes the activations of the previous layer
    x = Dropout(0.2)(x)  # Dropout to prevent overfitting
    
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    # Second convolutional layer with fewer filters
    x = MaxPooling1D(2, padding='same')(x)  # Max pooling to reduce the sequence length by half
    x = BatchNormalization()(x)  # Normalizes the activations of the previous layer
    x = Dropout(0.2)(x)  # Dropout to prevent overfitting
    
    encoded = Conv1D(16, 3, activation='relu', padding='same')(x)  # Encoded representation
    
    # Decoder
    x = Conv1D(16, 3, activation='relu', padding='same')(encoded)
    # First deconvolutional layer to start expanding dimensions
    x = UpSampling1D(2)(x)  # Increases the sequence length
    x = BatchNormalization()(x)  # Normalizes the activations of the previous layer
    x = Dropout(0.2)(x)  # Dropout to prevent overfitting
    
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    # Second deconvolutional layer to continue expanding dimensions
    x = UpSampling1D(2)(x)  # Increases the sequence length
    x = BatchNormalization()(x)  # Normalizes the activations of the previous layer
    x = Dropout(0.2)(x)  # Dropout to prevent overfitting
    
    # Adjust for mismatch in temporal dimension
    crop_amount = (x.shape[1] - input_shape[0])
    if crop_amount > 0:
        x = Cropping1D(cropping=(crop_amount // 2, crop_amount - crop_amount // 2))(x)  # Crop symmetrically
    elif crop_amount < 0:
        x = ZeroPadding1D(padding=(-crop_amount // 2, -crop_amount + (-crop_amount // 2)))(x)  # Pad symmetrically
    
    decoded = Conv1D(input_shape[-1], 3, activation='sigmoid', padding='same')(x)  # Output should match the original input
    
    autoencoder = Model(inputs, decoded)  # Model initialization
    autoencoder.compile(optimizer='adam', loss='mse')  # Compile model with MSE loss
    
    return autoencoder


def Attention_AE(input_shape):
    inputs = Input(shape=input_shape)
    
    # Encoder
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=L2(1e-4)))(inputs)
    # Bidirectional LSTM to capture dependencies in both forward and backward directions
    x = BatchNormalization()(x)  # Normalizes the activations of the previous layer
    x = Dropout(0.3)(x)  # Dropout to prevent overfitting
    
    # Multi-Head Self-Attention
    attention = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    # Multi-Head Attention to focus on different parts of the input sequence
    x = Add()([x, attention])  # Residual Connection to improve gradient flow
    x = LayerNormalization()(x)  # Layer normalization to stabilize the training
    
    x = Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=L2(1e-4)))(x)
    # Bidirectional LSTM to reduce the sequence to a single vector
    x = Dropout(0.3)(x)  # Dropout to prevent overfitting
    
    # Bottleneck
    bottleneck = Dense(16, activation='relu', kernel_regularizer=L2(1e-4))(x)  # Dense layer to create the bottleneck
    bottleneck = RepeatVector(input_shape[0])(bottleneck)  # Repeat the bottleneck vector across the time dimension
    
    # Decoder
    x = LSTM(32, return_sequences=True, kernel_regularizer=L2(1e-4))(bottleneck)
    # LSTM layer to start reconstructing the sequence
    x = BatchNormalization()(x)  # Normalizes the activations of the previous layer
    x = Dropout(0.3)(x)  # Dropout to prevent overfitting
    
    x = LSTM(64, return_sequences=True, kernel_regularizer=L2(1e-4))(x)
    # Another LSTM layer to further decode the sequence
    x = BatchNormalization()(x)  # Normalizes the activations of the previous layer
    x = Dropout(0.3)(x)  # Dropout to prevent overfitting
    
    # Multi-Head Self-Attention
    attention = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    # Multi-Head Attention to focus on different parts of the output sequence
    x = Add()([x, attention])  # Residual Connection to improve gradient flow
    x = LayerNormalization()(x)  # Layer normalization to stabilize the training
    
    decoded = TimeDistributed(Dense(input_shape[1], activation='sigmoid'))(x)
    # TimeDistributed Dense layer to reconstruct the original feature space
    
    autoencoder = Model(inputs, decoded)  # Model initialization
    autoencoder.compile(optimizer='adam', loss='mse')  # Compile model with MSE loss
    
    return autoencoder

'''
The following AE model can be applied to any of the above models to add noise to the input data
'''
def Denoising_AE(input_shape, noise_factor=0.5):
    inputs = Input(shape=input_shape)
    noisy_inputs = Lambda(lambda x: x + noise_factor * tf.random.normal(tf.shape(x)))(inputs)
    
    # Encoder
    x = Bidirectional(LSTM(64, activation='relu', return_sequences=True))(noisy_inputs)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    
    encoded = LSTM(32, activation='relu', return_sequences=False)(x)
    encoded = Dropout(0.2)(encoded)
    encoded = BatchNormalization()(encoded)
    
    # Bottleneck
    bottleneck = Dense(16, activation='relu')(encoded)
    bottleneck = RepeatVector(input_shape[0])(bottleneck)
    
    # Decoder
    x = LSTM(32, activation='relu', return_sequences=True)(bottleneck)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    
    decoded = LSTM(128, activation='relu', return_sequences=True)(x)
    outputs = TimeDistributed(Dense(input_shape[1]))(decoded)
    
    autoencoder = Model(inputs, outputs)
    autoencoder.compile(optimizer='adam', loss='msle')
    
    return autoencoder


# I don't know why, but the model below is not working

#def ConvLSTM_AE(input_shape):
    inputs = Input(shape=(input_shape[0], input_shape[1], 1))
    
    # Encoder
    x = ConvLSTM1D(64, 3, activation='relu', padding='same', return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = ConvLSTM1D(32, 3, activation='relu', padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    
    # Bottleneck
    encoded = ConvLSTM1D(16, 3, activation='relu', padding='same', return_sequences=False)(x)    
    bottleneck = Dense(16, activation='relu')(encoded)
    
    # Decoder
    x = RepeatVector(input_shape[0])(bottleneck)
    x = ConvLSTM1D(32, 3, activation='relu', padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM1D(64, 3, activation='relu', padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    
    # Adjust for mismatch in temporal dimension
    crop_amount = (x.shape[1] - input_shape[0])
    if crop_amount > 0:
        x = Cropping1D(cropping=(crop_amount // 2, crop_amount - crop_amount // 2))(x)  # Crop symmetrically
    elif crop_amount < 0:
        x = ZeroPadding1D(padding=(-crop_amount // 2, -crop_amount + (-crop_amount // 2)))(x)  # Pad symmetrically
    
    decoded = ConvLSTM1D(1, 3, activation='sigmoid', padding='same', return_sequences=True)(x)
    
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='msle')
    
    return autoencoder