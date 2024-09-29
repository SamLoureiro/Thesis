import tensorflow as tf

from keras.layers import (Input, Dense, BatchNormalization, Dropout, LSTM, 
                          RepeatVector, TimeDistributed, Bidirectional, Input, Conv1D, MaxPooling1D, UpSampling1D, Cropping1D, ZeroPadding1D, Add, GlobalMaxPooling1D, 
                          Flatten, Reshape, ConvLSTM1D, MultiHeadAttention, LayerNormalization, Lambda, GaussianNoise, Layer, Activation, Conv1DTranspose, LeakyReLU)
from keras_tuner import HyperModel
from tensorflow.keras import regularizers
from keras.models import Model, Sequential
from keras.losses import mean_squared_error, mean_squared_logarithmic_error, mean_absolute_error
from keras.optimizers import Adam, RMSprop, SGD
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

# Time based models

class RNN_DEEP:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self):
        # Default hyperparameters
        units1 = 128
        dropout1 = 0.3
        units2 = 64
        dropout2 = 0.3
        units3 = 32
        dropout3 = 0.3
        bottleneck_units = 16
        noise_ratio = 0.2

        inputs = Input(shape=self.input_shape)
        noisy_inputs = GaussianNoise(noise_ratio)(inputs)  # Add Gaussian noise to the inputs
        
        # Encoder
        x = Bidirectional(LSTM(units1, activation='relu', return_sequences=True, regularizers=regularizers.L1L2(l1=1e-4, l2=1e-4)))(noisy_inputs)
        x = Dropout(dropout1)(x)
        x = BatchNormalization()(x)
        
        x = Bidirectional(LSTM(units2, activation='relu', return_sequences=True, regularizers=regularizers.L1L2(l1=1e-4, l2=1e-4)))(x)
        x = Dropout(dropout2)(x)
        x = BatchNormalization()(x)
        
        encoded = LSTM(units3, activation='relu', return_sequences=False, regularizers=regularizers.L1L2(l1=1e-4, l2=1e-4))(x)
        encoded = Dropout(dropout3)(encoded)
        encoded = BatchNormalization()(encoded)
        
        # Bottleneck
        bottleneck = Dense(bottleneck_units, activation='relu', regularizers=regularizers.L1L2(l1=1e-4, l2=1e-4))(encoded)
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
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder

class RNN_SIMPLE:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self):
        # Default hyperparameters
        units1 = 64
        dropout1 = 0.3
        units2 = 32
        dropout2 = 0.3
        units3 = 16
        dropout3 = 0.3
        bottleneck_units = 16
        noise_ratio = 0.2

        inputs = Input(shape=self.input_shape)
        noisy_inputs = GaussianNoise(noise_ratio)(inputs)  # Add Gaussian noise to the inputs
        
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
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    
class CNN_DEEP:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self):
        # Default hyperparameters
        conv1_filters = 64
        conv2_filters = 128
        conv3_filters = 256
        conv4_filters = 512
        dropout_rate = 0.3
        bottleneck_units = 128
        noise_ratio = 0.2
        kernel_size = 5

        inputs = Input(shape=self.input_shape)
        noisy_inputs = GaussianNoise(noise_ratio)(inputs)  # Add Gaussian noise to the inputs
        
        # Encoder
        x = Conv1D(conv1_filters, kernel_size, activation='relu', pad='same', regularizers=regularizers.L1L2(l1=1e-4, l2=1e-4))(noisy_inputs)
        x = Dropout(dropout_rate)(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = BatchNormalization()(x)
        
        x = Conv1D(conv2_filters, kernel_size, activation='relu', pad='same', regularizers=regularizers.L1L2(l1=1e-4, l2=1e-4))(x)
        x = Dropout(dropout_rate)(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = BatchNormalization()(x)

        x = Conv1D(conv3_filters, kernel_size, activation='relu', pad='same', regularizers=regularizers.L1L2(l1=1e-4, l2=1e-4))(x)
        x = Dropout(dropout_rate)(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = BatchNormalization()(x)
        
        # Residual connection
        residual = Conv1D(conv4_filters, 1, padding='same')(x)
        x = Conv1D(conv4_filters, kernel_size, activation='relu', padding='same',
                   kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4))(x)
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
        
        x = Conv1D(conv4_filters, kernel_size, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Conv1D(conv3_filters, kernel_size, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Conv1D(conv2_filters, kernel_size, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Conv1D(conv1_filters, kernel_size, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Adjust for mismatch in temporal dimension
        crop_amount = (x.shape[1] - self.input_shape[0])
        if crop_amount > 0:
            x = Cropping1D(cropping=(crop_amount // 2, crop_amount - crop_amount // 2))(x)
        elif crop_amount < 0:
            x = ZeroPadding1D(padding=(-crop_amount // 2, -crop_amount - (-crop_amount // 2)))(x)
        
        decoded = Conv1D(self.input_shape[-1], kernel_size, activation='sigmoid', padding='same')(x)
        
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder

class CNN_SIMPLE:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self):
        # Default hyperparameters
        conv1_filters = 16
        conv2_filters = 32
        conv3_filters = 64
        kernel_size = 5
        dropout_rate = 0.3
        noise_ratio = 0.2

        inputs = Input(shape=self.input_shape)
        noisy_inputs = GaussianNoise(noise_ratio)(inputs)  # Add Gaussian noise to the inputs
        
        # Encoder
        x = Conv1D(conv1_filters, kernel_size, activation='relu', padding='same')(noisy_inputs)
        x = MaxPooling1D(2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Conv1D(conv2_filters, kernel_size, activation='relu', padding='same')(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        encoded = Conv1D(conv3_filters, kernel_size, activation='relu', padding='same')(x)
        
        # Decoder
        x = Conv1D(conv3_filters, kernel_size, activation='relu', padding='same')(encoded)
        x = UpSampling1D(2)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Conv1D(conv2_filters, kernel_size, activation='relu', padding='same')(x)
        x = UpSampling1D(2)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Adjust for mismatch in temporal dimension
        crop_amount = (x.shape[1] - self.input_shape[0])
        if crop_amount > 0:
            x = Cropping1D(cropping=(crop_amount // 2, crop_amount - crop_amount // 2))(x)
        elif crop_amount < 0:
            x = ZeroPadding1D(padding=(-crop_amount // 2, -crop_amount - (-crop_amount // 2)))(x)
        
        decoded = Conv1D(self.input_shape[-1], kernel_size, activation='sigmoid', padding='same')(x)
        
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder



class Attention_AE:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self):
        # Default hyperparameters
        lstm_units1 = 64
        lstm_units2 = 32
        attention_heads = 4
        attention_key_dim = 64
        dropout_rate = 0.3
        bottleneck_units = 16
        learning_rate = 1e-3
        noise_ratio = 0.2

        inputs = Input(shape=self.input_shape)
        noisy_inputs = GaussianNoise(noise_ratio)(inputs)  # Add Gaussian noise to the inputs
        
        # Encoder
        x = Bidirectional(LSTM(lstm_units1, return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4)))(noisy_inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Multi-Head Self-Attention
        attention = MultiHeadAttention(num_heads=attention_heads, key_dim=attention_key_dim)(x, x)
        x = Add()([x, attention])
        x = LayerNormalization()(x)
        
        x = Bidirectional(LSTM(lstm_units2, return_sequences=False, kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4)))(x)
        x = Dropout(dropout_rate)(x)
        
        # Bottleneck
        bottleneck = Dense(bottleneck_units, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4))(x)
        bottleneck = RepeatVector(self.input_shape[0])(bottleneck)
        
        # Decoder
        x = LSTM(lstm_units2, return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4))(bottleneck)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = LSTM(lstm_units1, return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4))(x)
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
    
    # Default hyperparameters
    conv1_filters = 128
    conv2_filters = 64
    conv3_filters = 32
    kernel_size = 5

    x = Conv1D(conv1_filters, kernel_size, padding="same")(encoder_inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(conv2_filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(conv3_filters, kernel_size, padding="same")(x)
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
    
    # Default hyperparameters
    conv1_filters = 32
    conv2_filters = 64
    conv3_filters = 128
    kernel_size = 5

    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(timestamps // 4 * 64, activation="relu")(latent_inputs)
    x = Reshape((timestamps // 4, 64))(x)
    x = LSTM(64, return_sequences=True, activation="relu")(x)
    x = Dropout(0.2)(x)  # Dropout for regularization
    x = LSTM(32, return_sequences=True, activation="relu")(x)
    x = Dropout(0.2)(x)  # Dropout for regularization
    
    x = Conv1DTranspose(conv1_filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1DTranspose(conv2_filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1DTranspose(conv3_filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Adjust for mismatch in temporal dimension
    crop_amount = (x.shape[1] - input_shape[0])
    if crop_amount > 0:
        x = Cropping1D(cropping=(crop_amount // 2, crop_amount - crop_amount // 2))(x)
    elif crop_amount < 0:
        x = ZeroPadding1D(padding=(-crop_amount // 2, -crop_amount - (-crop_amount // 2)))(x)

    decoder_outputs = Conv1DTranspose(features_per_timestamp, kernel_size, activation="sigmoid", padding="same")(x)
    return Model(latent_inputs, decoder_outputs, name="decoder")

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
        self.val_total_loss_tracker = metrics.Mean(name="val_total_loss")
        self.val_reconstruction_loss_tracker = metrics.Mean(name="val_reconstruction_loss")
        self.val_kl_loss_tracker = metrics.Mean(name="val_kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_kl_loss_tracker,
        ]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def train_step(self, data):
        if isinstance(data, tuple):
            x, __ = data
        else:
            x = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction = tf.expand_dims(reconstruction, axis=-1)

            reconstruction_loss = tf.reduce_mean(tf.square(x - reconstruction), axis=(1, 2))
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

    def test_step(self, data):
        if isinstance(data, tuple):
            x, __ = data
        else:
            x = data

        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        reconstruction = tf.expand_dims(reconstruction, axis=-1)

        reconstruction_loss = tf.reduce_mean(tf.square(x - reconstruction), axis=(1, 2))
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        total_loss = reconstruction_loss + kl_loss

        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.val_total_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "kl_loss": self.val_kl_loss_tracker.result(),
        }

    # Adding the get_config method for serialization
    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            'encoder': self.encoder.get_config(),
            'decoder': self.decoder.get_config(),
        })
        return config

    # Adding the from_config method for deserialization
    @classmethod
    def from_config(cls, config):
        encoder_config = config.pop('encoder')
        decoder_config = config.pop('decoder')
        encoder = Model.from_config(encoder_config)
        decoder = Model.from_config(decoder_config)
        return cls(encoder, decoder, **config)


def build_vae(input_shape, latent_dim):
    encoder = build_encoder(input_shape, latent_dim)
    decoder = build_decoder(input_shape, latent_dim)
    return VAE(encoder, decoder)

# Example of how to compile the VAE model
def vae_model_builder(input_shape, latent_dim=16):
    model = build_vae(input_shape, latent_dim)
    
    model.compile(optimizer='adam')
    return model



# Straight features based models


class VDAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
        self.val_total_loss_tracker = metrics.Mean(name="val_loss")
        self.val_reconstruction_loss_tracker = metrics.Mean(name="val_reconstruction_loss")
        self.val_kl_loss_tracker = metrics.Mean(name="val_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_kl_loss_tracker,
        ]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def train_step(self, data):
        if isinstance(data, tuple):
            x, __ = data
        else:
            x = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(tf.square(x - reconstruction), axis=1)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var + 1e-7))
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

    def test_step(self, data):
        if isinstance(data, tuple):
            x, __ = data
        else:
            x = data

        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)

        reconstruction_loss = tf.reduce_mean(tf.square(x - reconstruction), axis=1)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var + 1e-7))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        total_loss = reconstruction_loss + kl_loss

        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.val_total_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "kl_loss": self.val_kl_loss_tracker.result(),
        }
        
    # Adding the get_config method for serialization
    def get_config(self):
        config = super(VDAE, self).get_config()
        config.update({
            'encoder': self.encoder.get_config(),
            'decoder': self.decoder.get_config(),
        })
        return config

    # Adding the from_config method for deserialization
    @classmethod
    def from_config(cls, config):
        encoder_config = config.pop('encoder')
        decoder_config = config.pop('decoder')
        encoder = Model.from_config(encoder_config)
        decoder = Model.from_config(decoder_config)
        return cls(encoder, decoder, **config)

def build_dense_encoder(input_shape, latent_dim):
    inputs = Input(shape=(input_shape,))    
    
    x = Dense(64, activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x) 
    
    x = Dense(128, activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x) 
    
    x = Dense(256, activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4))(inputs)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x) 
    
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling_Dense()([z_mean, z_log_var])
    return Model(inputs, [z_mean, z_log_var, z], name="encoder")

def build_dense_decoder(input_shape, latent_dim):
    latent_inputs = Input(shape=(latent_dim,))
    
    x = Dense(256, activation='relu')(latent_inputs)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x) 
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x) 
    
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x) 
    
    outputs = Dense(input_shape, activation='sigmoid')(x)
    return Model(latent_inputs, outputs, name="decoder")

class Sampling_Dense(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vdae(input_shape, latent_dim):
    encoder = build_dense_encoder(input_shape, latent_dim)
    decoder = build_dense_decoder(input_shape, latent_dim)
    return VDAE(encoder, decoder)

def vdae_model_builder(input_shape, latent_dim):
    model = build_vdae(input_shape, latent_dim)
    model.compile(optimizer='adam')
    return model

class DENSE:
    def __init__(
        self, 
        input_dim, 
        noise_factor=0.1,
        units_1=256, 
        dropout_1=0.2, 
        units_2=128, 
        dropout_2=0.2, 
        units_3=64, 
        bottleneck_units=16, 
        dropout_3=0.2, 
        dropout_4=0.2, 
        l1=1e-4, 
        l2=1e-4,
        learning_rate=0.001
    ):
        self.input_dim = input_dim
        self.noise_factor = noise_factor
        self.units_1 = units_1
        self.dropout_1 = dropout_1
        self.units_2 = units_2
        self.dropout_2 = dropout_2
        self.units_3 = units_3
        self.bottleneck_units = bottleneck_units
        self.dropout_3 = dropout_3
        self.dropout_4 = dropout_4
        self.l1 = l1
        self.l2 = l2
        self.learning_rate = learning_rate

    def build(self):

        input_layer = Input(shape=(self.input_dim,))
        noisy_inputs = GaussianNoise(self.noise_factor)(input_layer)

        
        # Encoder
        encoded = Dense(self.units_1, activation='relu', kernel_regularizer=regularizers.L1L2(l1=self.l1, l2=self.l2))(noisy_inputs)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(self.dropout_1)(encoded)
        encoded = Dense(self.units_2, activation='relu', kernel_regularizer=regularizers.L1L2(l1=self.l1, l2=self.l2))(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(self.dropout_2)(encoded)
        encoded = Dense(self.units_3, activation='relu', kernel_regularizer=regularizers.L1L2(l1=self.l1, l2=self.l2))(encoded)
        
        # Bottleneck
        bottleneck = Dense(self.bottleneck_units, activation='relu')(encoded)

        # Decoder
        decoded = Dense(self.units_3, activation='relu')(bottleneck)
        decoded = BatchNormalization()(decoded)
        decoded = Dropout(self.dropout_3)(decoded)
        decoded = Dense(self.units_2, activation='relu')(decoded)
        decoded = BatchNormalization()(decoded)
        decoded = Dropout(self.dropout_4)(decoded)
        decoded = Dense(self.units_1, activation='relu')(decoded)
        decoded = Dense(self.input_dim, activation='sigmoid')(decoded)
        
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')        
        return autoencoder
