import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load preprocessed training and test data
X_train = pd.read_csv("C:/Users/Lenovo/Desktop/Dev_Linear_Traning/X_train_preprocessed.csv")
X_test = pd.read_csv("C:/Users/Lenovo/Desktop/Dev_Linear_Traning/X_test_preprocessed.csv")
y_train = pd.read_csv("C:/Users/Lenovo/Desktop/Dev_Linear_Traning/y_train_preprocessed.csv").values.ravel()
y_test = pd.read_csv("C:/Users/Lenovo/Desktop/Dev_Linear_Traning/y_test_preprocessed.csv").values.ravel()

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for future use
joblib.dump(scaler, "C:/Users/Lenovo/Desktop/Dev_Linear_Traning/code/scaler.pkl")

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Save the trained model
joblib.dump(model, "C:/Users/Lenovo/Desktop/Dev_Linear_Traning/code/model.pkl")

# Evaluate the model
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
