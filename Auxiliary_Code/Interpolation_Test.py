import pandas as pd
import numpy as np
from scipy import interpolate
import os

def interpolation(csv_file_path, target_frames, output_dir=None):
    """Extract features from accelerometer data CSV files, resample to match target frames, round values, and save to a new CSV file."""
    columns = ['timestamp', 'accX_l', 'accY_l', 'accZ_l', 'gyrX_l', 'gyrY_l', 'gyrZ_l', 
               'accX_r', 'accY_r', 'accZ_r', 'gyrX_r', 'gyrY_r', 'gyrZ_r']
    df = pd.read_csv(csv_file_path, usecols=columns)
    accel_data = df.to_numpy()
    
    # Extract the features
    num_samples, num_features = accel_data.shape

    # Determine the resampling ratio
    original_frames = num_samples
    if original_frames == target_frames:
        resampled_data = accel_data  # No need to resample if lengths are already equal
    else:
        # Create time axis for original and target data
        original_time = np.linspace(0, 1, original_frames)
        target_time = np.linspace(0, 1, target_frames)

        # Initialize the resampled data array
        resampled_data = np.zeros((target_frames, num_features))

        # Resample each feature individually
        for feature_idx in range(num_features):
            feature_data = accel_data[:, feature_idx]
            interpolator = interpolate.interp1d(original_time, feature_data, kind='linear', fill_value='extrapolate')
            resampled_data[:, feature_idx] = interpolator(target_time)
    
    # Round the resampled data to two decimal places
    resampled_data = np.round(resampled_data, 2)
    
    # Convert the resampled data back to a DataFrame
    resampled_df = pd.DataFrame(resampled_data, columns=columns)

    # Generate output file path
    if output_dir is None:
        output_dir = os.path.dirname(csv_file_path)
    
    output_file_name = f"resampled_{os.path.basename(csv_file_path)}"
    output_file_path = os.path.join(output_dir, output_file_name)
    
    # Save the resampled data to a new CSV file
    resampled_df.to_csv(output_file_path, index=False)
    
    print(f"Resampled data saved to {output_file_path}")
    return output_file_path

# Example usage
current_dir = os.getcwd()
csv_file_path = os.path.join(current_dir, 'Dataset_Piso', 'LISO', 'SAMPLES_1s', 'ACCEL', 'acel_liso_0_sample_0.csv')
target_frames = 100  # Replace with the desired target frame count
output_dir = os.path.join(current_dir, 'resampled_data')  # Directory where resampled CSVs will be saved

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

interpolation(csv_file_path, target_frames, output_dir)
