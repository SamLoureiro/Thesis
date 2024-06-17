import wave
import math
import os

def split_wav_file(input_file, output_directory, input_name, sample_duration=1):
    input_wav = wave.open(input_file, 'rb')
    framerate = input_wav.getframerate()
    sample_width = input_wav.getsampwidth()
    channels = input_wav.getnchannels()
    total_frames = input_wav.getnframes()
    total_duration = total_frames / framerate
    sample_frames = int(framerate * sample_duration)
    total_samples = math.ceil(total_duration / sample_duration)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i in range(total_samples - 1):  # Exclude the last sample
        print(i)
        start_frame = i * sample_frames
        end_frame = min((i + 1) * sample_frames, total_frames)
        output_file = os.path.join(output_directory, f'{input_name}_sample_{i}.WAV')
        output_wav = wave.open(output_file, 'wb')
        output_wav.setparams((channels, sample_width, framerate, end_frame - start_frame, 'NONE', 'not compressed'))

        input_wav.setpos(start_frame)
        output_wav.writeframes(input_wav.readframes(end_frame - start_frame))

        output_wav.close()

    input_wav.close()
    # Check if the last sample has the intended duration, if not, delete it
    last_sample_file = os.path.join(output_directory, f'{input_name}_sample_{total_samples - 2}.WAV')
    last_sample_duration = os.path.getsize(last_sample_file) / (framerate * sample_width * channels)

    if last_sample_duration < sample_duration:
        os.remove(last_sample_file)
        print(f"Last sample {last_sample_file} discarded because it does not have the intended duration ({last_sample_duration}).")

# Example usage
current_dir = os.getcwd()
input_name = 'LISO_4'
raw_data = os.path.join(current_dir, 'RAW')

# Define the input and output file paths
input_file = os.path.join(raw_data, input_name + ".WAV")
output_directory = os.path.join(current_dir, 'SAMPLES_1s', 'AUDIO')

# Check if output_directory exists, create it if not
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

split_wav_file(input_file, output_directory, input_name, sample_duration=1)

print("Sample files have been created successfully.")
