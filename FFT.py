import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr

# Define directories
current_dir = os.getcwd()

# Directories for DAMAGED and GOOD audio files
audio_dir_damaged = os.path.join(current_dir, 'Dataset_Bearings', 'BEARING_ONLY', 'V3')
audio_filename_damaged = "DAMAGED.WAV"
audio_path_damaged = os.path.join(audio_dir_damaged, audio_filename_damaged)

audio_dir_good = os.path.join(current_dir, 'Dataset_Bearings', 'BEARING_ONLY', 'V3')
audio_filename_good = "HEAVILY_DAMAGED.WAV"
audio_path_good = os.path.join(audio_dir_good, audio_filename_good)

# Noise profile path
noise_profile_file = os.path.join(current_dir, 'Dataset_Bearings', 'BEARING_ONLY', 'V2', 'NOISE', 'RECORD_3.WAV')

# Create a directory to save plots
plot_dir = "Plots"
os.makedirs(plot_dir, exist_ok=True)

# Load the DAMAGED audio file with a sample rate of 192 kHz
damaged_y, sr = librosa.load(audio_path_damaged, offset=1, duration=15, sr=192000)

# Load the GOOD audio file with a sample rate of 192 kHz
good_y, sr = librosa.load(audio_path_good, offset=1, duration=15, sr=192000)

# Load the noise profile audio file with a sample rate of 192 kHz
noise_y, sr = librosa.load(noise_profile_file, offset=1, duration=15, sr=192000)

# Noise reduction using noise profile for DAMAGED signal
damaged_y_denoised = nr.reduce_noise(y=damaged_y, sr=192000, y_noise=noise_profile_file, device="cuda", n_jobs=32)

# Noise reduction using noise profile for GOOD signal
good_y_denoised = nr.reduce_noise(y=good_y, sr=192000, y_noise=noise_profile_file, device="cuda", n_jobs=32)

# Compute the DFT of the original and denoised signals for DAMAGED and GOOD
dft_damaged_original = np.fft.fft(damaged_y)
dft_damaged_denoised = np.fft.fft(damaged_y_denoised)

dft_good_original = np.fft.fft(good_y)
dft_good_denoised = np.fft.fft(good_y_denoised)

# Frequency axis for the DFT
frequencies = np.fft.fftfreq(len(damaged_y), 1/sr)

# Plot the FFTs side by side for comparison
plt.figure(figsize=(16, 12))

# FFT of DAMAGED original audio signal
plt.subplot(2, 2, 1)
plt.plot(frequencies[:len(frequencies)//2], np.abs(dft_damaged_original)[:len(frequencies)//2], label='Original Audio Signal')
plt.title('FFT of Slightly Damaged Bearing Original Audio')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

# FFT of GOOD original audio signal
plt.subplot(2, 2, 2)
plt.plot(frequencies[:len(frequencies)//2], np.abs(dft_good_original)[:len(frequencies)//2], label='Original Audio Signal')
plt.title('FFT of HEAVILY_DAMAGED Bearing Original Audio')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

# FFT of DAMAGED denoised audio signal
plt.subplot(2, 2, 3)
plt.plot(frequencies[:len(frequencies)//2], np.abs(dft_damaged_denoised)[:len(frequencies)//2], label='Denoised Audio Signal')
plt.title('FFT of HEAVILY_DAMAGED Damaged Bearing Denoised Audio')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

# FFT of GOOD denoised audio signal
plt.subplot(2, 2, 4)
plt.plot(frequencies[:len(frequencies)//2], np.abs(dft_good_denoised)[:len(frequencies)//2], label='Denoised Audio Signal')
plt.title('FFT of HEAVILY_DAMAGED Bearing Denoised Audio')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.tight_layout()

# Save the plot
fft_plot_path = os.path.join(plot_dir, 'ONLY_BEARING', 'FFT_Slightly_DAMAGED_vs_HEAVILY_DAMAGED_Original_and_Denoised.svg')
plt.savefig(fft_plot_path, format='svg')
plt.show()

# Compute the residuals
residual_original = np.abs(dft_damaged_original) - np.abs(dft_good_original)
residual_denoised = np.abs(dft_damaged_denoised) - np.abs(dft_good_denoised)

# Plot the residuals in a separate figure
plt.figure(figsize=(16, 8))

# Residual of original audio signals
plt.subplot(2, 1, 1)
plt.plot(frequencies[:len(frequencies)//2], residual_original[:len(frequencies)//2], label='Residual (Original)')
plt.title('Residual FFT (Slightly Bearing - HEAVILY_DAMAGED Bearing) - Original Audio Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

# Residual of denoised audio signals
plt.subplot(2, 1, 2)
plt.plot(frequencies[:len(frequencies)//2], residual_denoised[:len(frequencies)//2], label='Residual (Denoised)')
plt.title('Residual FFT (Slightly Bearing - HEAVILY_DAMAGED Bearing) - Denoised Audio Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.tight_layout()

# Save the residual plot
residual_plot_path = os.path.join(plot_dir, 'ONLY_BEARING', 'FFT_Residuals_HEAVILY_DAMAGED_vs_GOOD_Original_and_Denoised.svg')
plt.savefig(residual_plot_path, format='svg')
plt.show()

print(f'Plots saved to {fft_plot_path} and {residual_plot_path}')
