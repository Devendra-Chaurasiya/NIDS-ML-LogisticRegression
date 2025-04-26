import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Load the combined dataset (50% data as per previous step)
data = pd.read_csv("C:/Users/Lenovo/Desktop/Dev_Linear_Traning/combined_data_50percent.csv")

# Check the columns in the dataset to ensure the correct column names
print("Columns in dataset:", data.columns)

# Ensure that columns are correctly referenced, including spaces if they exist
# Dropping non-feature columns (if Label' or other columns have leading/trailing spaces, ensure they are correctly referenced)
X = data.drop(['Label', 'AttackType'], axis=1)  # Dropping Label' and 'AttackType' columns for features
y = data['Label']  # The Label' column for target (with leading space, as observed in the data)

# Check for infinity or NaN values
print("Checking for NaN or infinite values in the dataset...")
print("NaN values in X:", X.isna().sum())
print("Infinite values in X:", (X == float('inf')).sum())

# Replace infinite values with NaN
X.replace([float('inf'), -float('inf')], float('nan'), inplace=True)

# Replace NaN values with the column mean (or another strategy)
X.fillna(X.mean(), inplace=True)

# Shuffle the dataset to ensure randomness
X, y = shuffle(X, y, random_state=42)

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Check the shapes of the processed data
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# Save the preprocessed data into new CSV files for further steps
pd.DataFrame(X_train, columns=X.columns).to_csv("X_train_preprocessed.csv", index=False)
pd.DataFrame(X_test, columns=X.columns).to_csv("X_test_preprocessed.csv", index=False)
pd.DataFrame(y_train, columns=['Label']).to_csv("y_train_preprocessed.csv", index=False)
pd.DataFrame(y_test, columns=['Label']).to_csv("y_test_preprocessed.csv", index=False)

# Proceed with the next step for training the model (logistic regression, etc.)
