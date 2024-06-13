import os
import csv

def remove_column(folder_path, column_index):
    #for filename in os.listdir(folder_path):
    filename = os.path.join(folder_path, 'acel_liso_4.csv')
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            rows = list(reader)
        
        for row in rows:
            del row[column_index]
        
        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(rows)

# Usage example
current_dir = os.getcwd()
folder_path = os.path.join(current_dir, 'Dataset_Piso', 'LISO', 'RAW')
column_index = 13  # Index of the column to remove (0-based)
remove_column(folder_path, column_index)