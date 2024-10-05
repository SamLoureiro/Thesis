import tensorflow as tf

from keras.layers import (Input, Dense, BatchNormalization, Dropout, LSTM, 
                          RepeatVector, TimeDistributed, Bidirectional, Input, Conv1D, MaxPooling1D, UpSampling1D, Cropping1D, ZeroPadding1D, Add, GlobalMaxPooling1D, 
                          Flatten, Reshape, ConvLSTM1D, MultiHeadAttention, LayerNormalization, Lambda, GaussianNoise, Layer, Activation, Conv1DTranspose)
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

# Time based sequencial models

class RNN_DEEP_Tuning(HyperModel):
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

class RNN_SIMPLE_Tuning(HyperModel):
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
    
    
class CNN_DEEP_Tuning(HyperModel):
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

class CNN_SIMPLE_Tuning(HyperModel):
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


class Attention_AE_Tuning(HyperModel):
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

def build_encoder(hp, input_shape, latent_dim):
    encoder_inputs = Input(shape=input_shape)
    
    # Declare hyperparameters
    conv1_filters = hp.Int('conv1_filters', min_value=32, max_value=128, step=32)
    conv2_filters = hp.Int('conv2_filters', min_value=32, max_value=128, step=32)
    conv3_filters = hp.Int('conv3_filters', min_value=32, max_value=128, step=32)
    kernel_size = hp.Int('kernel_size', min_value=3, max_value=7, step=2)

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

def build_decoder(hp, input_shape, latent_dim):
    timestamps, features_per_timestamp = input_shape
    
    # Declare hyperparameters
    conv1_filters = hp.Int('conv1_filters', min_value=32, max_value=128, step=32)
    conv2_filters = hp.Int('conv2_filters', min_value=32, max_value=128, step=32)
    conv3_filters = hp.Int('conv3_filters', min_value=32, max_value=128, step=32)
    kernel_size = hp.Int('kernel_size', min_value=3, max_value=7, step=2)

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

class VAE_Tuning(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
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
            "val_loss": self.val_total_loss_tracker.result(),
            "val_reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "val_kl_loss": self.val_kl_loss_tracker.result(),
        }


def build_vae(hp, input_shape, latent_dim):
    encoder = build_encoder(hp, input_shape, latent_dim)
    decoder = build_decoder(hp, input_shape, latent_dim)
    return VAE_Tuning(encoder, decoder)


# Example of how to use Keras Tuner for hypertuning
def vae_model_builder_Tuning(hp, input_shape):
    input_shape = input_shape  # Replace with your actual input shape
    latent_dim = hp.Int('latent_dim', min_value=2, max_value=64, step=2)
    model = build_vae(hp, input_shape, latent_dim)
    
    optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
    
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        opt = SGD(learning_rate=learning_rate)
    
    model.compile(optimizer=opt)
    return model

# Straight features based models


class VDAE_Tuning(Model):
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

def build_dense_encoder(hp, input_shape, latent_dim):
    inputs = Input(shape=(input_shape,))
    
    x = Dense(
        hp.Int('units1', min_value=32, max_value=512, step=32), 
        activation='relu',
        kernel_regularizer=regularizers.L1L2(
            l1=hp.Float('l1_1', min_value=1e-5, max_value=1e-2, sampling='log'),
            l2=hp.Float('l2_1', min_value=1e-5, max_value=1e-2, sampling='log')
        )
    )(inputs)
    x = BatchNormalization()(x)
    x = Dropout(rate=hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1))(x)
    
    x = Dense(
        hp.Int('units2', min_value=32, max_value=512, step=32), 
        activation='relu',
        kernel_regularizer=regularizers.L1L2(
            l1=hp.Float('l1_2', min_value=1e-5, max_value=1e-2, sampling='log'),
            l2=hp.Float('l2_2', min_value=1e-5, max_value=1e-2, sampling='log')
        )
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1))(x)
    
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling_Dense()([z_mean, z_log_var])
    return Model(inputs, [z_mean, z_log_var, z], name="encoder")

def build_dense_decoder(hp, input_shape, latent_dim):
    latent_inputs = Input(shape=(latent_dim,))
    
    x = Dense(
        hp.Int('units2', min_value=32, max_value=512, step=32), 
        activation='relu',
        kernel_regularizer=regularizers.L1L2(
            l1=hp.Float('l1_2', min_value=1e-5, max_value=1e-2, sampling='log'),
            l2=hp.Float('l2_2', min_value=1e-5, max_value=1e-2, sampling='log')
        )
    )(latent_inputs)
    x = BatchNormalization()(x)
    x = Dropout(rate=hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1))(x)
    
    x = Dense(
        hp.Int('units1', min_value=32, max_value=512, step=32), 
        activation='relu',
        kernel_regularizer=regularizers.L1L2(
            l1=hp.Float('l1_1', min_value=1e-5, max_value=1e-2, sampling='log'),
            l2=hp.Float('l2_1', min_value=1e-5, max_value=1e-2, sampling='log')
        )
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1))(x)
    
    outputs = Dense(input_shape, activation='sigmoid')(x)
    return Model(latent_inputs, outputs, name="decoder")

class Sampling_Dense(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vdae(hp, input_shape, latent_dim):
    encoder = build_dense_encoder(hp, input_shape, latent_dim)
    decoder = build_dense_decoder(hp, input_shape, latent_dim)
    return VDAE_Tuning(encoder, decoder)

def vdae_model_builder_Tuning(hp, input_shape):
    input_shape = input_shape  # Replace with your actual input shape
    latent_dim = hp.Int('latent_dim', min_value=2, max_value=64, step=2)
    model = build_vdae(hp, input_shape, latent_dim)
    
    optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop'])
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
    
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    else:
        opt = RMSprop(learning_rate=learning_rate)
    
    model.compile(optimizer=opt)
    return model


class DENSE_TUNING(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        if not isinstance(hp, HyperParameters):
            raise ValueError("Expected 'hp' to be an instance of 'HyperParameters'.")

        units_1 = hp.Int('units_1', min_value=256, max_value=512, step=64)
        units_2 = hp.Int('units_2', min_value=128, max_value=256, step=32)
        bottleneck_units = hp.Int('units_3', min_value=32, max_value=128, step=16)
        #units_3 = hp.Int('units_3', min_value=32, max_value=128, step=16)
        #bottleneck_units = hp.Int('bottleneck_units', min_value=8, max_value=12, step=8)
        
        noise_factor = 0.1   
        dropout_1 = 0.1       
        dropout_2 = 0.2      
        dropout_3 = 0.2
        dropout_4 = 0.1
        learning_rate = 0.01        
        l1_l2_reg = regularizers.L1L2(l1=0.01, l2=0.01)

        input_layer = Input(shape=(self.input_shape,))
        noisy_inputs = Lambda(lambda x: x + noise_factor * tf.random.normal(tf.shape(x)))(input_layer)
        noisy_inputs = GaussianNoise(noise_factor)(input_layer)
        
        # Encoder
        encoded = Dense(units_1, activation='relu', kernel_regularizer=l1_l2_reg)(noisy_inputs)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(dropout_1)(encoded)
        encoded = Dense(units_2, activation='relu', kernel_regularizer=l1_l2_reg)(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(dropout_2)(encoded)
        #encoded = Dense(units_3, activation='relu', kernel_regularizer=l1_l2_reg)(encoded)
        
        # Bottleneck
        bottleneck = Dense(bottleneck_units, activation='relu', kernel_regularizer=l1_l2_reg)(encoded)
        
        # Decoder
        #decoded = Dense(units_3, activation='relu', kernel_regularizer=l1_l2_reg)(bottleneck)
        decoded = BatchNormalization()(bottleneck)
        decoded = Dropout(dropout_3)(decoded)        
        decoded = Dense(units_2, activation='relu', kernel_regularizer=l1_l2_reg)(decoded)
        decoded = BatchNormalization()(decoded)
        decoded = Dropout(dropout_4)(decoded)
        decoded = Dense(units_1, activation='relu', kernel_regularizer=l1_l2_reg)(decoded)
        decoded = Dense(self.input_shape, activation='relu')(decoded)
        
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=learning_rate),
                            loss='mse')        
        return autoencoder
    
    
class DENSE_TUNING_MFCC(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        if not isinstance(hp, HyperParameters):
            raise ValueError("Expected 'hp' to be an instance of 'HyperParameters'.")

        units_1 = hp.Int('units_1', min_value=16, max_value=32, step=4)
        bottleneck_units = hp.Int('units_3', min_value=8, max_value=16, step=2)
        
        noise_factor = 0.1   
        dropout_1 = 0.1       
        dropout_4 = 0.1
        learning_rate = 0.01        
        l1_l2_reg = regularizers.L1L2(l1=0.01, l2=0.01)

        input_layer = Input(shape=(self.input_shape,))
        noisy_inputs = Lambda(lambda x: x + noise_factor * tf.random.normal(tf.shape(x)))(input_layer)
        noisy_inputs = GaussianNoise(noise_factor)(input_layer)
        
        # Encoder
        encoded = Dense(units_1, activation='relu', kernel_regularizer=l1_l2_reg)(noisy_inputs)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(dropout_1)(encoded)
        
        # Bottleneck
        bottleneck = Dense(bottleneck_units, activation='relu', kernel_regularizer=l1_l2_reg)(encoded)
        
        # Decoder
        decoded = BatchNormalization()(bottleneck)
        decoded = Dropout(dropout_4)(decoded)
        decoded = Dense(units_1, activation='relu', kernel_regularizer=l1_l2_reg)(decoded)
        decoded = Dense(self.input_shape, activation='relu')(decoded)
        
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=learning_rate),
                            loss='mse')        
        return autoencoder