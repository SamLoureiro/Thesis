import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr

# Define directories
current_dir = os.getcwd()

# Directory where the audio file is located
audio_dir = os.path.join(current_dir, 'Dataset_Bearings', 'BEARING_ONLY', 'V2', 'DAMAGED', 'RAW')
audio_filename = "RECORD_2.WAV"
audio_path = os.path.join(audio_dir, audio_filename)
noise_profile_file = os.path.join(current_dir, 'Dataset_Bearings', 'BEARING_ONLY', 'V2', 'NOISE', 'RECORD_3.WAV')

# Create a directory to save plots
plot_dir = "Plots"
os.makedirs(plot_dir, exist_ok=True)

# Load the audio file with a sample rate of 192 kHz
y, sr = librosa.load(noise_profile_file, sr = 192000)

#y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_profile_file, device="cuda", n_jobs=32)

# Compute the Short-Time Fourier Transform (STFT)
D = librosa.stft(y)

# Convert the complex-valued STFT matrix to magnitude and phase
magnitude, phase = librosa.magphase(D)

# Compute the Discrete Fourier Transform (DFT)
dft = np.fft.fft(y)

# Frequency axis for the DFT
frequencies = np.fft.fftfreq(len(y), 1/sr)

# Time axis for the original audio signal
time = np.linspace(0, len(y) / sr, len(y))

# Plot the original audio signal and DFT magnitude
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, y)
plt.title('Original Audio Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot the magnitude of the DFT
plt.subplot(2, 1, 2)
plt.plot(frequencies[:len(frequencies)//2], np.abs(dft)[:len(frequencies)//2])
plt.title('Fast Fourier Transform (Magnitude)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.tight_layout()

# Save the plot as an SVG file
plot_path = os.path.join(plot_dir, 'FFT_NOISE.svg')
plt.savefig(plot_path, format='svg')
plt.show()

print(f'Plot saved to {plot_path}')
