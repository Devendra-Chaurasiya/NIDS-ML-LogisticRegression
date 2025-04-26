import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ========== Load Dataset ==========
file_path = r'C:\Users\Lenovo\Desktop\Dev_Linear_Traning\combined_data.csv'
df = pd.read_csv(file_path)

# ========== Drop Unwanted Columns ==========
# Drop non-numeric or identifier columns like 'Filename' or ' Label' from features
drop_columns = [' Label', 'Filename']  # Add any extra non-numeric columns if needed
drop_columns = [col for col in drop_columns if col in df.columns]
y = df[' Label']
X = df.drop(columns=drop_columns, errors='ignore')

# ========== Keep Only Numeric Columns ==========
X = X.select_dtypes(include=[np.number])

# ========== Remove Outliers ==========
threshold = 1e12  # You can adjust this threshold
X = X.applymap(lambda x: np.nan if abs(x) > threshold else x)

# ========== Fill Missing Values ==========
X = X.fillna(X.median())

# ========== Train-Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Pipeline Setup ==========
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# ========== Train Model ==========
pipeline.fit(X_train, y_train)
                                                     
# ========== Evaluate (Optional) ==========
accuracy = pipeline.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}")
   