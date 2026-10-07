#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# --- Load ---
df = pd.read_csv('dataset/darknet_dataset_processed_encoded.csv')
print("Original shape:", df.shape)

# --- Separate features and labels ---
y = (df['Label'] == 'Darknet').astype(int)   # 1 = Darknet, 0 = Benign
X = df.drop(columns=['Label', 'Label.1'])

# Keep only numeric features
X = X.select_dtypes(include=[np.number])
print("Feature matrix:", X.shape)
print("Class balance:", y.value_counts().to_dict())

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train:", X_train.shape, "Test:", X_test.shape)

# --- Scale for MLP only (trees don't need it) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# --- Define models ---
models = {
    'Decision Tree': (DecisionTreeClassifier(random_state=42), False),
    'Random Forest': (RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42), False),
    'XGBoost':       (XGBClassifier(n_estimators=100, n_jobs=-1, random_state=42,
                                     eval_metric='logloss'), False),
    'LightGBM':      (LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=42, verbose=-1), False),
    'MLP':           (MLPClassifier(hidden_layer_sizes=(100,), max_iter=200,
                                     early_stopping=True, random_state=42), True),
}

# --- Train + evaluate ---
results = []
for name, (model, needs_scaling) in models.items():
    print(f"\n=== {name} ===")
    Xtr = X_train_scaled if needs_scaling else X_train
    Xte = X_test_scaled  if needs_scaling else X_test

    t0 = time.time()
    model.fit(Xtr, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)

    row = {
        'Model': name,
        'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall':    round(recall_score(y_test, y_pred), 4),
        'F1':        round(f1_score(y_test, y_pred), 4),
        'FPR':       round(fpr, 4),
        'AUC':       round(roc_auc_score(y_test, y_prob), 4),
        'TrainTime_s': round(train_time, 1),
    }
    results.append(row)
    print(row)
    print(classification_report(y_test, y_pred, target_names=['Benign','Darknet']))

# --- Summary table ---
results_df = pd.DataFrame(results)
print("\n===== SUMMARY =====")
print(results_df.to_string(index=False))
results_df.to_csv('results_summary.csv', index=False)
print("\nSaved to results_summary.csv")


# In[ ]:




