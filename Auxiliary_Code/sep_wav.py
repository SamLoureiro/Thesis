import wave
import math
import os

def split_wav_file(input_file, sample_duration=1):
    input_wav = wave.open(input_file, 'rb')
    framerate = input_wav.getframerate()
    sample_width = input_wav.getsampwidth()
    channels = input_wav.getnchannels()
    total_frames = input_wav.getnframes()
    total_duration = total_frames / framerate
    sample_frames = int(framerate * sample_duration)
    total_samples = math.ceil(total_duration / sample_duration)    

    # 192000: This is the framerate of the input WAV file. Framerate typically indicates how many frames (or samples) of audio are processed per second.

    # 2: This is the sample_width of the input WAV file, indicating the number of bytes per sample. In this case, each sample is 2 bytes (16-bit audio).

    # 1: This is the number of channels in the input WAV file. A value of 1 indicates mono audio (single channel).

    # 63172864 (i.e.): This is the total number of frames (total_frames) in the WAV file. Frames represent individual audio samples across all channels.

    # 329.0253333333333: This is the total_duration of the WAV file in seconds, calculated as total_frames / framerate. It indicates that the total duration of the WAV file is approximately 329.03 seconds.

    # 192000: This is again the framerate, indicating the same framerate as before.

    # 330: This is total_samples, which represents the number of 1-second samples (sample_duration=1) that will be created from the input WAV file. It's calculated as math.ceil(total_duration / sample_duration).

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i in range(total_samples):
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
    last_sample_file = os.path.join(output_directory, f'{input_name}_sample_{total_samples - 1}.WAV')
    last_sample_duration = os.path.getsize(last_sample_file) / (framerate * sample_width * channels)
    
    if last_sample_duration < sample_duration:
        os.remove(last_sample_file)
        print(f"Last sample {last_sample_file} discarded because it does not have the intended duration.")

current_dir = os.getcwd()
input_name = 'LISO_4'
raw_data = os.path.join(current_dir, 'RAW')

# Define the input and output file paths
input_file = os.path.join(raw_data, input_name + ".WAV")
output_directory = os.path.join(current_dir, 'SAMPLES_1s', 'AUDIO')

# Check if output_directory exists, create it if not
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

split_wav_file(input_file, sample_duration=1)

print("Sample files have been created successfully.")
