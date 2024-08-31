import tensorflow as tf

from keras.layers import (Input, Dense, BatchNormalization, Dropout, LSTM, 
                          RepeatVector, TimeDistributed, Bidirectional, Input, Conv1D, MaxPooling1D, UpSampling1D, Cropping1D, ZeroPadding1D, Add, GlobalMaxPooling1D, 
                          Flatten, Reshape, ConvLSTM1D, MultiHeadAttention, LayerNormalization, Lambda, GaussianNoise, Layer, Activation, Conv1DTranspose)
from keras_tuner import HyperModel
from keras.regularizers import L2, l1_l2
from keras.models import Model, Sequential
from keras.losses import mean_squared_error, mean_squared_logarithmic_error, mean_absolute_error
from keras.optimizers import Adam
from keras import metrics, random
from keras_tuner import HyperModel, HyperParameters, BayesianOptimization
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

class RNN_DEEP(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        # Declare hyperparameters
        units1 = hp.Int('units1', min_value=32, max_value=256, step=32)
        dropout1 = hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)
        units2 = hp.Int('units2', min_value=32, max_value=128, step=32)
        dropout2 = hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1)
        units3 = hp.Int('units3', min_value=16, max_value=64, step=16)
        dropout3 = hp.Float('dropout3', min_value=0.1, max_value=0.5, step=0.1)
        bottleneck_units = hp.Int('bottleneck_units', min_value=8, max_value=32, step=8)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        noise_ratio = hp.Float('noise_ratio', min_value=0.05, max_value=0.5, step=0.05)

        inputs = Input(shape=self.input_shape)
        noisy_inputs = GaussianNoise(noise_ratio)(inputs)  # Add Gaussian noise to the 
        
        # Encoder
        x = Bidirectional(LSTM(units1, activation='relu', return_sequences=True))(noisy_inputs)
        x = Dropout(dropout1)(x)
        x = BatchNormalization()(x)
        
        x = Bidirectional(LSTM(units2, activation='relu', return_sequences=True))(x)
        x = Dropout(dropout2)(x)
        x = BatchNormalization()(x)
        
        encoded = LSTM(units3, activation='relu', return_sequences=False)(x)
        encoded = Dropout(dropout3)(encoded)
        encoded = BatchNormalization()(encoded)
        
        # Bottleneck
        bottleneck = Dense(bottleneck_units, activation='relu')(encoded)
        bottleneck = RepeatVector(self.input_shape[0])(bottleneck)
        
        # Decoder
        x = LSTM(units3, activation='relu', return_sequences=True)(bottleneck)
        x = Dropout(dropout3)(x)
        x = BatchNormalization()(x)
        
        x = LSTM(units2, activation='relu', return_sequences=True)(x)
        x = Dropout(dropout2)(x)
        x = BatchNormalization()(x)
        
        decoded = LSTM(units1, activation='relu', return_sequences=True)(x)
        outputs = TimeDistributed(Dense(self.input_shape[1]))(decoded)
        
        autoencoder = Model(inputs, outputs)
        autoencoder.compile(optimizer=Adam(learning_rate=learning_rate),
                            loss='mse')
        
        return autoencoder

class RNN_SIMPLE(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        # Declare hyperparameters
        units1 = hp.Int('units1', min_value=32, max_value=256, step=32)
        dropout1 = hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)
        units2 = hp.Int('units2', min_value=32, max_value=128, step=32)
        dropout2 = hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1)
        units3 = hp.Int('units3', min_value=16, max_value=64, step=16)
        dropout3 = hp.Float('dropout3', min_value=0.1, max_value=0.5, step=0.1)
        bottleneck_units = hp.Int('bottleneck_units', min_value=8, max_value=32, step=8)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        noise_ratio = hp.Float('noise_ratio', min_value=0.1, max_value=0.5, step=0.1)

        inputs = Input(shape=self.input_shape)
        noisy_inputs = GaussianNoise(noise_ratio)(inputs)  # Add Gaussian noise to the 
        
        # Encoder
        x = Bidirectional(LSTM(units1, activation='relu', return_sequences=True))(noisy_inputs)
        x = Dropout(dropout1)(x)
        x = BatchNormalization()(x)
        
        x = Bidirectional(LSTM(units2, activation='relu', return_sequences=True))(x)
        x = Dropout(dropout2)(x)
        x = BatchNormalization()(x)
        
        encoded = LSTM(units3, activation='relu', return_sequences=False)(x)
        encoded = Dropout(dropout3)(encoded)
        encoded = BatchNormalization()(encoded)
        
        # Bottleneck
        bottleneck = Dense(bottleneck_units, activation='relu')(encoded)
        bottleneck = RepeatVector(self.input_shape[0])(bottleneck)
        
        # Decoder
        x = LSTM(units3, activation='relu', return_sequences=True)(bottleneck)
        x = Dropout(dropout3)(x)
        x = BatchNormalization()(x)
        
        x = LSTM(units2, activation='relu', return_sequences=True)(x)
        x = Dropout(dropout2)(x)
        x = BatchNormalization()(x)
        
        decoded = LSTM(units1, activation='relu', return_sequences=True)(x)
        outputs = TimeDistributed(Dense(self.input_shape[1]))(decoded)
        
        autoencoder = Model(inputs, outputs)
        autoencoder.compile(optimizer=Adam(learning_rate=learning_rate),
                            loss='mse')
        
        return autoencoder
    
    
class CNN_DEEP(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        # Declare hyperparameters
        conv1_filters = hp.Int('conv1_filters', min_value=32, max_value=128, step=32)
        conv2_filters = hp.Int('conv2_filters', min_value=64, max_value=256, step=64)
        conv3_filters = hp.Int('conv3_filters', min_value=128, max_value=512, step=128)
        conv4_filters = hp.Int('conv4_filters', min_value=256, max_value=1024, step=256)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        bottleneck_units = hp.Int('bottleneck_units', min_value=32, max_value=128, step=32)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        noise_ratio = hp.Float('noise_ratio', min_value=0.1, max_value=0.5, step=0.1)

        inputs = Input(shape=self.input_shape)
        noisy_inputs = GaussianNoise(noise_ratio)(inputs)  # Add Gaussian noise to the 
        
        # Encoder
        x = Conv1D(conv1_filters, 3, activation='relu', padding='same', kernel_regularizer=L2(0.001))(noisy_inputs)
        x = Dropout(dropout_rate)(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = BatchNormalization()(x)
        
        x = Conv1D(conv2_filters, 3, activation='relu', padding='same', kernel_regularizer=L2(0.001))(x)
        x = Dropout(dropout_rate)(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = BatchNormalization()(x)

        x = Conv1D(conv3_filters, 3, activation='relu', padding='same', kernel_regularizer=L2(0.001))(x)
        x = Dropout(dropout_rate)(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = BatchNormalization()(x)
        
        # Residual connection
        residual = Conv1D(conv4_filters, 1, padding='same')(x)
        x = Conv1D(conv4_filters, 3, activation='relu', padding='same', kernel_regularizer=L2(0.001))(x)
        x = Dropout(dropout_rate)(x)
        x = Add()([x, residual])
        
        # Bottleneck
        x = GlobalMaxPooling1D()(x)
        encoded = Dense(256, activation='relu')(x)
        encoded = Dropout(0.3)(encoded)
        bottleneck = Dense(bottleneck_units, activation='relu')(encoded)
        
        # Decoder
        x = Dense(256, activation='relu')(bottleneck)
        x = Dropout(dropout_rate)(x)
        x = RepeatVector(self.input_shape[0])(x)
        
        x = Conv1D(conv4_filters, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Conv1D(conv3_filters, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Conv1D(conv2_filters, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Conv1D(conv1_filters, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Adjust for mismatch in temporal dimension
        crop_amount = (x.shape[1] - self.input_shape[0])
        if crop_amount > 0:
            x = Cropping1D(cropping=(crop_amount // 2, crop_amount - crop_amount // 2))(x)
        elif crop_amount < 0:
            x = ZeroPadding1D(padding=(-crop_amount // 2, -crop_amount + (-crop_amount // 2)))(x)
        
        decoded = Conv1D(self.input_shape[-1], 3, activation='sigmoid', padding='same')(x)
        
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        
        return autoencoder

class CNN_SIMPLE(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        # Declare hyperparameters
        conv1_filters = hp.Int('conv1_filters', min_value=32, max_value=128, step=32)
        conv2_filters = hp.Int('conv2_filters', min_value=16, max_value=64, step=16)
        conv3_filters = hp.Int('conv3_filters', min_value=8, max_value=32, step=8)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        noise_ratio = hp.Float('noise_ratio', min_value=0.1, max_value=0.5, step=0.1)

        inputs = Input(shape=self.input_shape)
        noisy_inputs = GaussianNoise(noise_ratio)(inputs)  # Add Gaussian noise to the 
        
        # Encoder
        x = Conv1D(conv1_filters, 3, activation='relu', padding='same')(noisy_inputs)
        x = MaxPooling1D(2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Conv1D(conv2_filters, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        encoded = Conv1D(conv3_filters, 3, activation='relu', padding='same')(x)
        
        # Decoder
        x = Conv1D(conv3_filters, 3, activation='relu', padding='same')(encoded)
        x = UpSampling1D(2)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Conv1D(conv2_filters, 3, activation='relu', padding='same')(x)
        x = UpSampling1D(2)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Adjust for mismatch in temporal dimension
        crop_amount = (x.shape[1] - self.input_shape[0])
        if crop_amount > 0:
            x = Cropping1D(cropping=(crop_amount // 2, crop_amount - crop_amount // 2))(x)
        elif crop_amount < 0:
            x = ZeroPadding1D(padding=(-crop_amount // 2, -crop_amount + (-crop_amount // 2)))(x)
        
        decoded = Conv1D(self.input_shape[-1], 3, activation='sigmoid', padding='same')(x)
        
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        
        return autoencoder


class Attention_AE(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        # Declare hyperparameters
        lstm_units1 = hp.Int('lstm_units1', min_value=32, max_value=128, step=32)
        lstm_units2 = hp.Int('lstm_units2', min_value=16, max_value=64, step=16)
        attention_heads = hp.Int('attention_heads', min_value=2, max_value=8, step=2)
        attention_key_dim = hp.Int('attention_key_dim', min_value=32, max_value=128, step=32)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        bottleneck_units = hp.Int('bottleneck_units', min_value=8, max_value=32, step=8)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        noise_ratio = hp.Float('noise_ratio', min_value=0.1, max_value=0.5, step=0.1)

        inputs = Input(shape=self.input_shape)
        noisy_inputs = GaussianNoise(noise_ratio)(inputs)  # Add Gaussian noise to the 
        
        # Encoder
        x = Bidirectional(LSTM(lstm_units1, return_sequences=True, kernel_regularizer=L2(1e-4)))(noisy_inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Multi-Head Self-Attention
        attention = MultiHeadAttention(num_heads=attention_heads, key_dim=attention_key_dim)(x, x)
        x = Add()([x, attention])
        x = LayerNormalization()(x)
        
        x = Bidirectional(LSTM(lstm_units2, return_sequences=False, kernel_regularizer=L2(1e-4)))(x)
        x = Dropout(dropout_rate)(x)
        
        # Bottleneck
        bottleneck = Dense(bottleneck_units, activation='relu', kernel_regularizer=L2(1e-4))(x)
        bottleneck = RepeatVector(self.input_shape[0])(bottleneck)
        
        # Decoder
        x = LSTM(lstm_units2, return_sequences=True, kernel_regularizer=L2(1e-4))(bottleneck)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = LSTM(lstm_units1, return_sequences=True, kernel_regularizer=L2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Multi-Head Self-Attention
        attention = MultiHeadAttention(num_heads=attention_heads, key_dim=attention_key_dim)(x, x)
        x = Add()([x, attention])
        x = LayerNormalization()(x)
        
        decoded = TimeDistributed(Dense(self.input_shape[1], activation='sigmoid'))(x)
        
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        
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



# Define the Sampling layer
class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder(input_shape, latent_dim):
    encoder_inputs = Input(shape=input_shape)

    x = Conv1D(32, 3, padding="same")(encoder_inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(64, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(128, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # Adding LSTM layers
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(32)(x)
    
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.2)(x)  # Dropout for regularization
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

def build_decoder(input_shape, latent_dim):
    timestamps, features_per_timestamp = input_shape
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(timestamps // 4 * 64, activation="relu")(latent_inputs)
    x = Reshape((timestamps // 4, 64))(x)
    x = LSTM(64, return_sequences=True, activation="relu")(x)
    x = Dropout(0.2)(x)  # Dropout for regularization
    x = LSTM(32, return_sequences=True, activation="relu")(x)
    x = Dropout(0.2)(x)  # Dropout for regularization
    x = Conv1DTranspose(128, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1DTranspose(64, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1DTranspose(32, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Adjust for mismatch in temporal dimension
    crop_amount = (x.shape[1] - input_shape[0])
    if crop_amount > 0:
        x = Cropping1D(cropping=(crop_amount // 2, crop_amount - crop_amount // 2))(x)
    elif crop_amount < 0:
        x = ZeroPadding1D(padding=(-crop_amount // 2, -crop_amount - (-crop_amount // 2)))(x)

    decoder_outputs = Conv1DTranspose(features_per_timestamp, 3, activation="sigmoid", padding="same")(x)
    return Model(latent_inputs, decoder_outputs, name="decoder")

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        # Forward pass through encoder
        z_mean, z_log_var, z = self.encoder(inputs)
        # Forward pass through decoder
        reconstruction = self.decoder(z)
        return reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction = tf.expand_dims(reconstruction, axis=-1)  # Match target shape
            reconstruction_loss = tf.reduce_mean(tf.square(data - reconstruction), axis=(1, 2))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# Save a function to build and return the VAE model
def build_vae(input_shape, latent_dim):
    encoder = build_encoder(input_shape, latent_dim)
    decoder = build_decoder(input_shape, latent_dim)
    return VAE(encoder, decoder)




class SEQ_VAE(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        # Hyperparameters to tune
        conv_filters_1 = hp.Int('conv_filters_1', min_value=16, max_value=64, step=16)
        conv_filters_2 = hp.Int('conv_filters_2', min_value=32, max_value=128, step=32)
        kernel_size_1 = hp.Int('kernel_size_1', min_value=2, max_value=5, step=1)
        kernel_size_2 = hp.Int('kernel_size_2', min_value=2, max_value=5, step=1)
        lstm_units_1 = hp.Int('lstm_units_1', min_value=32, max_value=128, step=32)
        dropout_rate_1 = hp.Float('dropout_rate_1', min_value=0.1, max_value=0.5, step=0.1)
        lstm_units_2 = hp.Int('lstm_units_2', min_value=16, max_value=64, step=16)
        latent_dim = hp.Int('latent_dim', min_value=8, max_value=32, step=8)
        lstm_units_3 = hp.Int('lstm_units_3', min_value=16, max_value=64, step=16)
        dropout_rate_2 = hp.Float('dropout_rate_2', min_value=0.1, max_value=0.5, step=0.1)
        conv_filters_3 = hp.Int('conv_filters_3', min_value=32, max_value=128, step=32)
        conv_filters_4 = hp.Int('conv_filters_4', min_value=16, max_value=64, step=16)
        kernel_size_3 = hp.Int('kernel_size_3', min_value=2, max_value=5, step=1)
        kernel_size_4 = hp.Int('kernel_size_4', min_value=2, max_value=5, step=1)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

        timestamps = self.input_shape[0]
        features_per_timestamp = self.input_shape[1]
        encoder_inputs = Input(shape=self.input_shape)

        # Encoder
        x = Conv1D(conv_filters_1, kernel_size=kernel_size_1, activation='relu', padding='same')(encoder_inputs)
        x = Conv1D(conv_filters_2, kernel_size=kernel_size_2, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)

        # LSTM layers
        x = Reshape((timestamps, conv_filters_2))(x)
        x = Bidirectional(LSTM(lstm_units_1, return_sequences=True))(x)
        x = Dropout(dropout_rate_1)(x)
        x = LSTM(lstm_units_2)(x)

        # Latent space
        z_mean = Dense(latent_dim)(x)
        z_log_var = Dense(latent_dim)(x)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.random.normal(shape=tf.shape(z_mean), mean=0., stddev=1.)
            return z_mean + tf.exp(z_log_var / 2) * epsilon

        z = Lambda(sampling)([z_mean, z_log_var])
        encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

        # Decoder
        latent_inputs = Input(shape=(latent_dim,))
        x = Dense(timestamps * conv_filters_2)(latent_inputs)
        x = Reshape((timestamps, conv_filters_2))(x)

        # LSTM layers
        x = LSTM(lstm_units_2, return_sequences=True)(x)
        x = Dropout(dropout_rate_2)(x)
        x = Bidirectional(LSTM(lstm_units_3, return_sequences=True))(x)

        # Convolutional layers
        x = Reshape((timestamps, lstm_units_3*2))(x)
        x = Conv1D(conv_filters_3, kernel_size=kernel_size_3, activation='relu', padding='same')(x)
        x = Conv1D(conv_filters_4, kernel_size=kernel_size_4, activation='relu', padding='same')(x)
        decoder_outputs = TimeDistributed(Dense(features_per_timestamp))(x)
        decoder = Model(latent_inputs, decoder_outputs, name='decoder')

        # VAE Outputs
        vae_outputs = decoder(encoder(encoder_inputs)[2])
        # Custom loss layer
        class VAELossLayer(Layer):
            def call(self, inputs):
                encoder_inputs, vae_outputs, z_mean, z_log_var = inputs
                reconstruction_loss = mse(encoder_inputs, vae_outputs)
                #reconstruction_loss = tf.reduce_mean(tf.square(encoder_inputs - vae_outputs), axis=(1,2))
                reconstruction_loss *= timestamps * features_per_timestamp
                kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                kl_loss = tf.reduce_mean(kl_loss)
                kl_loss *= -0.5
                vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
                self.add_loss(vae_loss)
                return vae_outputs

        vae_loss_layer = VAELossLayer()([encoder_inputs, vae_outputs, z_mean, z_log_var])
        vae = Model(encoder_inputs, vae_loss_layer, name='vae')
        vae.compile(optimizer=Adam(learning_rate=learning_rate))

        return vae