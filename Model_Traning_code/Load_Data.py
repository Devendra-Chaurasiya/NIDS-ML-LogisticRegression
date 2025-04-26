import pandas as pd
import numpy as np
import os

# Define paths to the CSV files (only the filenames, relative to your directory)
file_paths = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv"
]

# Define the corresponding attack labels for each file
file_attack_labels = {
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": "DDoS",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": "PortScan",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": "BENIGN",
    "Monday-WorkingHours.pcap_ISCX.csv": "BENIGN",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": "Infiltration",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": "Web Attacks",
    "Tuesday-WorkingHours.pcap_ISCX.csv": "BENIGN",
    "Wednesday-workingHours.pcap_ISCX.csv": "BENIGN"
}

# Base directory where the files are located
base_dir = "C:/Users/Lenovo/Desktop/Dev_Linear_Traning/cicids2017/"

# Read and combine the data with the appropriate labels
combined_data = []

# Load data from each file
for file_name in file_paths:
    file_path = os.path.join(base_dir, file_name)
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Clean column names to remove any extra spaces
    df.columns = df.columns.str.strip()
    
    # Add a new column for the attack type based on the file
    if file_name in file_attack_labels:
        attack_label = file_attack_labels[file_name]
    else:
        print(f"Error: No attack label found for {file_name}")
        continue
    
    df['AttackType'] = attack_label
    
    # Filter BENIGN entries to 50% for BENIGN class files
    if attack_label == 'BENIGN':
        df = df.sample(frac=0.5, random_state=42)
    
    # Append the data from this file to the combined list
    combined_data.append(df)

# Concatenate all the data into a single DataFrame
final_data = pd.concat(combined_data, ignore_index=True)

# Save the final combined dataset to a CSV file
final_data.to_csv("C:/Users/Lenovo/Desktop/Dev_Linear_Traning/combined_data_50percent.csv", index=False)

# Output the final dataset shape
print(f"Final dataset shape: {final_data.shape}")
print(f"Total attack types in the final dataset: {final_data['AttackType'].value_counts()}")
