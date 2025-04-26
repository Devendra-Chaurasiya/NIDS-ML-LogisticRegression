import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load the trained model and scaler
model = joblib.load('code\model.pkl')  # Update path if necessary
scaler = joblib.load('code\scaler.pkl')  # Update path if necessary

# Function to create an attack sample
def create_attack_sample():
    # Creating a sample for 'DoS Hulk' attack (with example values)
    sample_doS_hulk = [
        80,  # Destination Port
        9000,  # Flow Duration
        200,  # Total Fwd Packets
        100,  # Total Backward Packets
        1000,  # Total Length of Fwd Packets
        500,  # Total Length of Bwd Packets
        200,  # Fwd Packet Length Max
        50,  # Fwd Packet Length Min
        150,  # Fwd Packet Length Mean
        30,  # Fwd Packet Length Std
        100,  # Bwd Packet Length Max
        20,  # Bwd Packet Length Min
        60,  # Bwd Packet Length Mean
        15,  # Bwd Packet Length Std
        1000,  # Flow Bytes/s
        50,  # Flow Packets/s
        10,  # Flow IAT Mean
        5,  # Flow IAT Std
        15,  # Flow IAT Max
        2,  # Flow IAT Min
        50,  # Fwd IAT Total
        10,  # Fwd IAT Mean
        20,  # Fwd IAT Std
        40,  # Fwd IAT Max
        5,  # Fwd IAT Min
        20,  # Bwd IAT Total
        10,  # Bwd IAT Mean
        5,  # Bwd IAT Std
        30,  # Bwd IAT Max
        2,  # Bwd IAT Min
        1,  # Fwd PSH Flags
        0,  # Bwd PSH Flags
        0,  # Fwd URG Flags
        0,  # Bwd URG Flags
        20,  # Fwd Header Length
        15,  # Bwd Header Length
        30,  # Fwd Packets/s
        25,  # Bwd Packets/s
        50,  # Min Packet Length
        200,  # Max Packet Length
        100,  # Packet Length Mean
        40,  # Packet Length Std
        30,  # Packet Length Variance
        10,  # FIN Flag Count
        5,  # SYN Flag Count
        2,  # RST Flag Count
        0,  # PSH Flag Count
        30,  # ACK Flag Count
        0,  # URG Flag Count
        0,  # CWE Flag Count
        0,  # ECE Flag Count
        0.5,  # Down/Up Ratio
        100,  # Average Packet Size
        20,  # Avg Fwd Segment Size
        30,  # Avg Bwd Segment Size
        200,  # Fwd Header Length.1
        5,  # Fwd Avg Bytes/Bulk
        10,  # Fwd Avg Packets/Bulk
        100,  # Fwd Avg Bulk Rate
        20,  # Bwd Avg Bytes/Bulk
        30,  # Bwd Avg Packets/Bulk
        5,  # Bwd Avg Bulk Rate
        10,  # Subflow Fwd Packets
        20,  # Subflow Fwd Bytes
        50,  # Subflow Bwd Packets
        30,  # Subflow Bwd Bytes
        100,  # Init_Win_bytes_forward
        50,  # Init_Win_bytes_backward
        200,  # act_data_pkt_fwd
        10,  # min_seg_size_forward
        15,  # Active Mean
        10,  # Active Std
        20,  # Active Max
        5,  # Active Min
        20,  # Idle Mean
        10,  # Idle Std
        15,  # Idle Max
        5,  # Idle Min
    ]
    
    # You can create more attack samples like this and customize their values
    return sample_doS_hulk

# Add more attack samples or change features accordingly for different attacks
attack_samples = [create_attack_sample()]

# Transform and predict the attack samples
for sample in attack_samples:
    sample = np.array(sample).reshape(1, -1)
    scaled_sample = scaler.transform(sample)  # Scale the sample
    prediction = model.predict(scaled_sample)  # Get the prediction from the model
    print(f"Prediction for attack sample: {prediction[0]}")
