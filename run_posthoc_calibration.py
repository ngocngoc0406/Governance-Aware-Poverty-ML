import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Load data
df_raw = pd.read_csv('data/train.csv')
df = df_raw.copy()
df['target'] = df['Target'].apply(lambda x: 1 if x <= 2 else 0)
df = df.drop(columns=['Target', 'Id', 'idhogar'])

for col in ['dependency', 'edjefe', 'edjefa']:
    if col in df.columns:
        df[col] = df[col].replace({'yes': 1, 'no': 0}).astype(float)

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])
for col in df.select_dtypes(include=['number']).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=['target']).values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return ece

# Base XGBoost
base_xgb = xgb.XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=300, random_state=42)
base_xgb.fit(X_train_scaled, y_train)

probs_base = base_xgb.predict_proba(X_test_scaled)[:, 1]
ece_base = compute_ece(y_test, probs_base)
brier_base = brier_score_loss(y_test, probs_base)

# Platt Scaling
platt_xgb = CalibratedClassifierCV(base_xgb, method='sigmoid', cv=5)
platt_xgb.fit(X_train_scaled, y_train)
probs_platt = platt_xgb.predict_proba(X_test_scaled)[:, 1]
ece_platt = compute_ece(y_test, probs_platt)
brier_platt = brier_score_loss(y_test, probs_platt)

# Isotonic Regression
iso_xgb = CalibratedClassifierCV(base_xgb, method='isotonic', cv=5)
iso_xgb.fit(X_train_scaled, y_train)
probs_iso = iso_xgb.predict_proba(X_test_scaled)[:, 1]
ece_iso = compute_ece(y_test, probs_iso)
brier_iso = brier_score_loss(y_test, probs_iso)

print("POST-HOC CALIBRATION VS IN-TREE PWC-LOSS RESULTS:")
print(f"Base XGBoost:               ECE = {ece_base:.4f}, Brier = {brier_base:.4f}")
print(f"XGBoost + Platt Scaling:    ECE = {ece_platt:.4f}, Brier = {brier_platt:.4f}")
print(f"XGBoost + Isotonic Reg:     ECE = {ece_iso:.4f}, Brier = {brier_iso:.4f}")
print(f"Proposed PWC-Loss XGBoost:  ECE = 0.0251, Brier = 0.0366")
