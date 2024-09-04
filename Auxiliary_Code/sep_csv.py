import csv
import os

# Get the current directory and define the input name
current_dir = os.getcwd()
input_name = 'acel_liso_4'
raw_data = os.path.join(current_dir, 'Dataset_Piso', 'LISO', 'RAW')

# Define the input and output file paths
input_file = os.path.join(raw_data, input_name + ".csv")
output_directory = os.path.join(current_dir, 'Dataset_Augmented', 'smooth_floor', 'ACCEL')

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Define the time interval in milliseconds and overlap factor
time_interval = 1000  # 1 second
overlap_factor = 0.5  # 50% overlap

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

    # Calculate the step size based on the overlap factor
    step_size = int(50 * (1 - overlap_factor))

    # Iterate through each row in the .csv file
    for row in reader:
        timestamp = int(row[0])  # Assuming the timestamp is in the first column

        # If this is the first row in the current sample
        if current_sample_start_time is None:
            current_sample_start_time = timestamp

        # Add the current row to the current sample
        current_sample.append(row)

        # Check if the current sample has reached 50 rows
        if len(current_sample) == 50:
            # Save the current sample to a new file
            output_file = os.path.join(output_directory, f'{input_name}_sample_{i}.csv')
            with open(output_file, 'w', newline='') as output:
                writer = csv.writer(output)
                writer.writerow(header)
                writer.writerows(current_sample)
            print(f"Sample {i} created with {len(current_sample)} rows.")

            # Increment the sample counter
            i += 1

            # Determine the overlap portion of the current sample to retain
            current_sample = current_sample[-step_size:]

            # Reset the start time for the next sample
            current_sample_start_time = timestamp

    '''# If there are any remaining rows after the last full sample
    if len(current_sample) > 0 and len(current_sample) < 50:
        # Fill the remaining rows to make it 50
        avg_row = average_of_last_three(current_sample)
        if avg_row:
            while len(current_sample) < 50:
                current_sample.append(avg_row)

        # Save the last sample to a new file
        output_file = os.path.join(output_directory, f'{input_name}_sample_{i}.csv')
        with open(output_file, 'w', newline='') as output:
            writer = csv.writer(output)
            writer.writerow(header)
            writer.writerows(current_sample)
        print(f"Sample {i} created with {len(current_sample)} rows.")'''

print("Sample files have been created successfully.")