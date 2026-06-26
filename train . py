import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("--- AI Fraud Detection: Model Training Shuru ---")

# 1. Fake Fraud Dataset banana
np.random.seed(42)
data_size = 1000

data = {
    'transaction_amount': np.random.uniform(10, 5000, data_size),
    'is_international': np.random.choice([0, 1], size=data_size, p=[0.9, 0.1]),
    'is_midnight': np.random.choice([0, 1], size=data_size, p=[0.85, 0.15]),
}

df = pd.DataFrame(data)
# Ek simple rule se fraud logic decide karna (e.g., bada amount aur midnight = fraud ka zyada chance)
df['is_fraud'] = ((df['transaction_amount'] > 3000) & (df['is_midnight'] == 1) | 
                  (df['is_international'] == 1) & (df['transaction_amount'] > 2000)).astype(int)

# Dataset ko 'dataset' folder me save karna
os.makedirs('dataset', exist_ok=True)
df.to_csv('dataset/fraud_data.csv', index=False)
print("[✓] Fake Dataset ban gaya aur 'dataset/fraud_data.csv' me save ho gaya.")

# 2. Features aur Target alag karna
X = df[['transaction_amount', 'is_international', 'is_midnight']]
y = df['is_fraud']

# Data ko Train aur Test sets me baantna
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Train karna (Random Forest AI Model)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy check karna
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[✓] Model ki Accuracy: {acc * 100:.2f}%")

# 4. Train hue Model ko 'model' folder me save karna
os.makedirs('model', exist_ok=True)
with open('model/fraud_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("[✓] AI Model train ho gaya aur 'model/fraud_model.pkl' me save ho gaya!")
print("--- Training Khatam ---")