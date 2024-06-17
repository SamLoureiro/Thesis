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

i = 0

# Function to calculate the average of the past 3 lines
def average_of_last_three(rows):
    if len(rows) < 3:
        return None
    avg_row = []
    for col in range(len(rows[0])):
        avg = sum(float(rows[-j][col]) for j in range(1, 4)) / 3
        avg_row.append(f'{avg:.3f}')  # Formatting average to 3 decimal places
    return avg_row

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
        elif timestamp - current_sample_start_time >= time_interval:
            # Save the current sample to a new file
            print("Sample lenght before adjust: ", len(current_sample))
            if len(current_sample) > 50:
                current_sample = current_sample[:50]
            elif len(current_sample) < 50:
                avg_row = average_of_last_three(current_sample)
                if avg_row:
                    while len(current_sample) < 50:
                        current_sample.append(avg_row)
            print("Sample lenght after adjust: ", len(current_sample))
            output_file = os.path.join(output_directory, f'{input_name}_sample_{i}.csv')
            i += 1
            with open(output_file, 'w', newline='') as output:
                writer = csv.writer(output)
                writer.writerow(header)
                writer.writerows(current_sample)

            # Reset variables for the new sample
            current_sample = []
            current_sample_start_time = timestamp

        # Add the current row to the current sample
        current_sample.append(row)        

    # Save the last sample to a new file if it contains any rows
    '''if current_sample:
        if len(current_sample) > 50:
            current_sample = current_sample[:50]
        elif len(current_sample) < 50:
            avg_row = average_of_last_three(current_sample)
            if avg_row:
                while len(current_sample) < 50:
                    current_sample.append(avg_row)

        output_file = os.path.join(output_directory, f'{input_name}_sample_{i}.csv')
        with open(output_file, 'w', newline='') as output:
            writer = csv.writer(output)
            writer.writerow(header)
            writer.writerows(current_sample)'''

print("Sample files have been created successfully.")
