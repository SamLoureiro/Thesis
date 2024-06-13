import pandas as pd
import glob
import os

# Merge multiple CSV files into one
def merge_csv_files(folder_path, current_dir):
    """
    Merges all CSV files in the specified folder into a single DataFrame.

    Parameters:
    folder_path (str): The path to the folder containing the CSV files.

    Returns:
    pd.DataFrame: A DataFrame containing the merged data from all CSV files.
    """
    # List to store individual DataFrames
    data_frames = []
    data_frames_merged = []
    last_file = ""
    n_merged = 0
    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):        
        if 'merged' in file_name.lower() and file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            print(f"Loading CSV file: {file_path}")
            df = pd.read_csv(file_path)
            data_frames_merged.append(df)
            n_merged += 1
            if n_merged == 2:
                merged_df = pd.concat(data_frames_merged, ignore_index=True)
                output_file = os.path.join(current_dir, 'Dataset_Piso', 'MERGED', 'ALL_merged.csv')
                merged_df.to_csv(output_file, index=False)
                print(f"Merged CSV file saved to: {output_file}")
                return merged_df
            continue
        elif file_name.endswith('.csv') and not file_name.startswith('error'):
            last_file = file_name
            file_path = os.path.join(folder_path, file_name)
            print(f"Loading CSV file: {file_path}")
            df = pd.read_csv(file_path)
            data_frames.append(df)

    # Concatenate all DataFrames into a single DataFrame
    merged_df = pd.concat(data_frames, ignore_index=True)
    # Add a column to the merged DataFrame
    merged_df['piso'] = 'LISO' if extract_label_from_filename(last_file) == 0 else 'TIJOLEIRA'
    if extract_label_from_filename(last_file) == 0:
        output_file = os.path.join(current_dir, 'Dataset_Piso', 'MERGED', 'LISO_merged_data.csv')
    else:
        output_file = os.path.join(current_dir, 'Dataset_Piso', 'MERGED', 'TIJOLEIRA_merged_data.csv')
    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV file saved to: {output_file}")

    return merged_df


# Function to extract label from filename
def extract_label_from_filename(filename):
    if 'LISO' in filename.upper():
        return 0  # Label 0 for LISO
    elif 'TIJOLEIRA' in filename.upper():
        return 1  # Label 1 for TIJOLEIRA
    else:
        raise ValueError("Filename does not contain a valid label")


# Example usage
current_dir = os.getcwd()
folder_path = os.path.join(current_dir, 'Dataset_Piso', 'TIJOLEIRA', 'RAW')
merged_df = merge_csv_files(folder_path, current_dir)
print(f"Merged DataFrame shape: {merged_df.shape}")
print(merged_df.head())

folder_path = os.path.join(current_dir, 'Dataset_Piso', 'LISO', 'RAW')
merged_df = merge_csv_files(folder_path, current_dir)
print(f"Merged DataFrame shape: {merged_df.shape}")
print(merged_df.head())

merged_folder = os.path.join(current_dir, 'Dataset_Piso', 'MERGED')
merged_df = merge_csv_files(merged_folder, current_dir)
