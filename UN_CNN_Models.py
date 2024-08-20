import tensorflow as tf

from keras.layers import (Input, Dense, BatchNormalization, Dropout, LSTM, 
                          RepeatVector, TimeDistributed, Bidirectional, Input, Conv1D, MaxPooling1D, UpSampling1D, Cropping1D, ZeroPadding1D, Add, GlobalMaxPooling1D, 
                          Flatten, Reshape, ConvLSTM1D, MultiHeadAttention, LayerNormalization, Lambda)
from keras.regularizers import L2, l1_l2
from keras.models import Model, Sequential
from keras.losses import mean_squared_error, mean_squared_logarithmic_error, mean_absolute_error
import numpy as np

mse = mean_squared_error
msle = mean_squared_logarithmic_error
mae = mean_absolute_error


'''

Models to test in the main file

# Input Shape of each saple: (50, 2143) - where:

# 50 - timestamps of each sample
# 2143 - features per timestamp


'''

def RNN_Complex(input_shape):
    """Build and compile an RNN autoencoder model."""
    inputs = Input(shape=input_shape)
    
    # Encoder
    x = Bidirectional(LSTM(128, activation='relu', return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    
    x = Bidirectional(LSTM(64, activation='relu', return_sequences=True))(x)
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

def RNN_Simple(input_shape):
    """Build and compile a simpler RNN autoencoder model."""
    inputs = Input(shape=input_shape)
    
    # Encoder
    x = Bidirectional(LSTM(64, activation='relu', return_sequences=True))(inputs)
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


def DEEP_CNN(input_shape):
    inputs = Input(shape=input_shape)
    
    # Encoder
    x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=L2(0.001))(inputs)
    x = Dropout(0.2)(x)
    x = MaxPooling1D(2, padding='same')(x)  # Output shape: (25, 64)
    x = BatchNormalization()(x)
    
    x = Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=L2(0.001))(x)
    x = Dropout(0.2)(x)
    x = MaxPooling1D(2, padding='same')(x)  # Output shape: (13, 128)
    x = BatchNormalization()(x)

    x = Conv1D(256, 3, activation='relu', padding='same', kernel_regularizer=L2(0.001))(x)
    x = Dropout(0.2)(x)
    x = MaxPooling1D(2, padding='same')(x)  # Output shape: (7, 256)
    x = BatchNormalization()(x)
    
    # Residual connection
    residual = Conv1D(512, 1, padding='same')(x)  # Adjusting to 512 filters to match the next layer
    x = Conv1D(512, 3, activation='relu', padding='same', kernel_regularizer=L2(0.001))(x)
    x = Dropout(0.2)(x)
    x = Add()([x, residual])  # Residual connection
    
    # Bottleneck
    x = GlobalMaxPooling1D()(x)
    encoded = Dense(256, activation='relu')(x)
    encoded = Dropout(0.3)(encoded)
    bottleneck = Dense(64, activation='relu')(encoded)
    
    # Decoder
    x = Dense(256, activation='relu')(bottleneck)
    x = Dropout(0.2)(x)
    x = RepeatVector(input_shape[0])(x)
    
    x = Conv1D(512, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Conv1D(256, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Adjust for mismatch in temporal dimension
    crop_amount = (x.shape[1] - input_shape[0])
    if crop_amount > 0:
        x = Cropping1D(cropping=(crop_amount // 2, crop_amount - crop_amount // 2))(x)  # Crop symmetrically
    elif crop_amount < 0:
        x = ZeroPadding1D(padding=(-crop_amount // 2, -crop_amount + (-crop_amount // 2)))(x)  # Pad symmetrically
    
    decoded = Conv1D(input_shape[-1], 3, activation='sigmoid', padding='same')(x)  # Output layer
    
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='msle')
    
    return autoencoder

def CNN_Simple(input_shape):
    inputs = Input(shape=input_shape)
    
    # Encoder
    x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2, padding='same')(x)  # Reduces to 25
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)  # Reduces to 13
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    encoded = Conv1D(16, 3, activation='relu', padding='same')(x)  # Encoded representation
    
    # Decoder
    x = Conv1D(16, 3, activation='relu', padding='same')(encoded)
    x = UpSampling1D(2)(x)  # Increases to 26
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)  # Increases to 52
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Adjust for mismatch in temporal dimension
    crop_amount = (x.shape[1] - input_shape[0])
    if crop_amount > 0:
        x = Cropping1D(cropping=(crop_amount // 2, crop_amount - crop_amount // 2))(x)  # Crop symmetrically
    elif crop_amount < 0:
        x = ZeroPadding1D(padding=(-crop_amount // 2, -crop_amount + (-crop_amount // 2)))(x)  # Pad symmetrically
    
    decoded = Conv1D(input_shape[-1], 3, activation='sigmoid', padding='same')(x)  # Output should match the original input
    
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='msle')
    
    return autoencoder


def Attention_AE(input_shape):
    inputs = Input(shape=input_shape)
    
    # Encoder
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=L2(1e-4)))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Multi-Head Self-Attention
    attention = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = Add()([x, attention])  # Residual Connection
    x = LayerNormalization()(x)
    
    x = Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=L2(1e-4)))(x)
    x = Dropout(0.3)(x)
    
    # Bottleneck
    bottleneck = Dense(16, activation='relu', kernel_regularizer=L2(1e-4))(x)
    bottleneck = RepeatVector(input_shape[0])(bottleneck)
    
    # Decoder
    x = LSTM(32, return_sequences=True, kernel_regularizer=L2(1e-4))(bottleneck)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = LSTM(64, return_sequences=True, kernel_regularizer=L2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Multi-Head Self-Attention
    attention = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = Add()([x, attention])  # Residual Connection
    x = LayerNormalization()(x)
    
    decoded = TimeDistributed(Dense(input_shape[1], activation='sigmoid'))(x)
    
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='msle')
    
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