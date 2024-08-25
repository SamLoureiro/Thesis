import numpy as np
import pandas as pd
import librosa
import noisereduce as nr
from scipy.stats import kurtosis, skew
import config  # Import the config file


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
        # Pre-emphasis filter
        audio_data = np.append(y[0], y[1:] - 0.97 * y[:-1])
        # Short-Time Fourier Transform (STFT)
        stft = librosa.stft(audio_data, n_fft=config.mfcc_params['n_fft'], hop_length=config.mfcc_params['hop_length'])
        # Power spectrum
        spectrogram = np.abs(stft)**2
                
        n_mels = config.mfcc_params['n_mels']
        
        # Find the number of Mel filter banks that can be computed without any empty filters
        # Unccomment the following code if the samples proprietaries are not known, or the pre-processing parameters were changed
        '''mfcc_computed = False
        while not mfcc_computed:
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
        stft_rms = np.sqrt(np.mean(stft**2, axis=1))
        for i in range(len(stft_mean)):
            features[f'stft_mean_{i}'] = stft_mean[i]
            features[f'stft_std_{i}'] = stft_std[i]
            features[f'stft_rms_{i}'] = stft_rms[i]

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