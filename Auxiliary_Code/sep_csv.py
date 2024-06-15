import csv
import os

# Get the current directory and define the input name
current_dir = os.getcwd()
input_name = 'acel_liso_4'
raw_data = os.path.join(current_dir, 'RAW')

# Define the input and output file paths
input_file = os.path.join(raw_data, input_name + ".csv")
output_directory = os.path.join(current_dir, 'SAMPLES_1s', 'ACCEL')

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Define the time interval in milliseconds
time_interval = 1000  # 1 second

# Minimum number of lines (rows) required in each sample
min_lines_per_sample = 50

i = 0

# Read the input .csv file
with open(input_file, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip the header row

    # Initialize variables
    current_sample = []
    current_sample_start_time = None

    # Iterate through each row in the .csv file
    for row in reader:
        timestamp = int(row[0])  # Assuming the timestamp is in the first column

        # Check if the current row belongs to the current sample or a new sample
        if current_sample_start_time is None:
            current_sample_start_time = timestamp

        # Add the current row to the current sample
        current_sample.append(row)

        # Check if the current sample duration is at least time_interval
        if timestamp - current_sample_start_time >= time_interval:
            # Check if the current sample has at least min_lines_per_sample lines
            if len(current_sample) >= min_lines_per_sample:
                # Save the current sample to a new file
                output_file = os.path.join(output_directory, f'{input_name}_sample_{i}.csv')
                with open(output_file, 'w', newline='') as output:
                    writer = csv.writer(output)
                    writer.writerow(header)
                    writer.writerows(current_sample)
                i += 1
            else:
                print(f"Discarding sample {input_name}_sample_{i}.csv because it does not have at least {min_lines_per_sample} lines.")

            # Reset variables for the new sample
            current_sample = []
            current_sample_start_time = timestamp

    # Check if the last sample meets the intended criteria
    if len(current_sample) >= min_lines_per_sample:
        output_file = os.path.join(output_directory, f'{input_name}_sample_{i}.csv')
        with open(output_file, 'w', newline='') as output:
            writer = csv.writer(output)
            writer.writerow(header)
            writer.writerows(current_sample)
    elif current_sample:
        print(f"Discarding last sample {input_name}_sample_{i}.csv because it does not have at least {min_lines_per_sample} lines.")

print("Sample files have been created successfully.")
