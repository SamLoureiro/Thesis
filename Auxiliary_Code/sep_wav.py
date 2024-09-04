import wave
import math
import os

def split_wav_file(input_file, output_directory, input_name, sample_duration=1, overlap_factor=0.5):
    input_wav = wave.open(input_file, 'rb')
    framerate = input_wav.getframerate()
    sample_width = input_wav.getsampwidth()
    channels = input_wav.getnchannels()
    total_frames = input_wav.getnframes()
    sample_frames = int(framerate * sample_duration)
    overlap_frames = int(sample_frames * overlap_factor)
    step_frames = sample_frames - overlap_frames

    # Calculate the number of samples needed, considering the overlap
    total_samples = math.ceil((total_frames - overlap_frames) / step_frames)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i in range(total_samples):
        start_frame = i * step_frames
        end_frame = min(start_frame + sample_frames, total_frames)
        output_file = os.path.join(output_directory, f'{input_name}_sample_{i}.WAV')
        output_wav = wave.open(output_file, 'wb')
        output_wav.setparams((channels, sample_width, framerate, end_frame - start_frame, 'NONE', 'not compressed'))

        input_wav.setpos(start_frame)
        output_wav.writeframes(input_wav.readframes(end_frame - start_frame))

        output_wav.close()
        print(f"Sample {i} created from frame {start_frame} to {end_frame}.")

    input_wav.close()

    # Check if the last sample has the intended duration, if not, delete it
    last_sample_file = os.path.join(output_directory, f'{input_name}_sample_{total_samples - 1}.WAV')
    last_sample_duration = os.path.getsize(last_sample_file) / (framerate * sample_width * channels)

    if last_sample_duration < sample_duration:
        os.remove(last_sample_file)
        print(f"Last sample {last_sample_file} discarded because it does not have the intended duration ({last_sample_duration} seconds).")

# Get the current directory and define the input name
current_dir = os.getcwd()
input_name = 'LISO_4'
raw_data = os.path.join(current_dir, 'Dataset_Piso', 'LISO', 'RAW')

# Define the input and output file paths
input_file = os.path.join(raw_data, input_name + ".WAV")
output_directory = os.path.join(current_dir, 'Dataset_Augmented', 'smooth_floor', 'AUDIO')

# Check if output_directory exists, create it if not
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Split the WAV file with a sample duration of 1 second and 50% overlap
split_wav_file(input_file, output_directory, input_name, sample_duration=1, overlap_factor=0.5)

print("Sample files have been created successfully.")
