# NIDS-ML-LogisticClassifier

A machine learning pipeline for **Network Intrusion Detection System (NIDS)** using **Logistic Regression**.  
This project uses the **CICIDS2017 dataset** to classify network traffic into benign or malicious attacks through preprocessing, scaling, SMOTE balancing, and model evaluation.

---

## 📂 Project Structure

| File | Purpose |
| :--- | :------ |
| `Load_Data.py` | Combines multiple CSV files and assigns appropriate attack labels. |
| `Load-step2.PY` | Cleans the dataset: drops unwanted columns, handles outliers, splits into train-test. |
| `preprocessing .py` | Handles missing/infinite values, shuffles, scales, and prepares final preprocessed datasets. |
| `train_model.py` | Trains a **Logistic Regression** model, evaluates accuracy, saves model and scaler. |

---

## 🛠️ Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- joblib

Install dependencies using:

```bash
pip install pandas numpy scikit-learn imbalanced-learn joblib
```

---

## 🚀 How to Run

1. **Combine and label the data**  
   ```bash
   python Load_Data.py
   ```
   → Generates: `combined_data_50percent.csv`

2. **Clean and prepare the data**  
   ```bash
   python preprocessing\ .py
   ```
   → Generates:
   - `X_train_preprocessed.csv`
   - `X_test_preprocessed.csv`
   - `y_train_preprocessed.csv`
   - `y_test_preprocessed.csv`

3. **Train the Logistic Regression model**  
   ```bash
   python train_model.py
   ```
   → Saves:
   - `model.pkl` (the trained Logistic Regression model)
   - `scaler.pkl` (the fitted StandardScaler)

4. **Evaluate the results**  
   - Accuracy Score
   - Classification Report (Precision, Recall, F1-Score)
   - Confusion Matrix

---

## 📊 Dataset

- **Source:** [CICIDS 2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Attacks included:** DDoS, PortScan, Infiltration, Web Attacks, and Benign traffic.

---

## ⚙️ Machine Learning Details

- **Algorithm:** Logistic Regression
- **Preprocessing:**
  - Handling missing values and outliers
  - Feature Scaling using StandardScaler
  - SMOTE oversampling for class imbalance
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score

---

## 📈 Example Results

- **Accuracy:** ~85–90% (varies slightly by run)
- **Strong precision and recall** for major attack classes.

---

## 📚 Future Enhancements

- Hyperparameter tuning (C, penalty terms)
- Cross-validation
- Testing other models like Random Forests, XGBoost, etc.
- Visualization of confusion matrix and ROC curves

---

## 🙌 Contributions

Feel free to fork this repo, suggest improvements, and submit pull requests!

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🏷️ Tags

`NIDS` `Machine Learning` `Logistic Regression` `CICIDS2017` `Cybersecurity` `Intrusion Detection`

---

# 🚀 Let's Detect Intrusions with Machine Learning!

---
