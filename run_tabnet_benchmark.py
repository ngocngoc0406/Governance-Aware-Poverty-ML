import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, brier_score_loss
)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

print("=" * 70)
print("BENCHMARKING TABNET DEEP TABULAR MODEL FOR POVERTY TARGETING")
print("=" * 70)

# Load dataset
df_raw = pd.read_csv('data/train.csv')

df = df_raw.copy()
df['target'] = df['Target'].apply(lambda x: 1 if x <= 2 else 0)
df = df.drop(columns=['Target', 'Id', 'idhogar'])

for col in ['dependency', 'edjefe', 'edjefa']:
    if col in df.columns:
        df[col] = df[col].replace({'yes': 1, 'no': 0}).astype(float)

for col in df.select_dtypes(include=['category', 'object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
for col in df.select_dtypes(include=['number']).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=['category', 'object']).columns:
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

# Train TabNet Classifier
start_time = time.time()
clf = TabNetClassifier(
    n_d=16, n_a=16, n_steps=3,
    gamma=1.3, lambda_sparse=1e-3,
    optimizer_params=dict(lr=2e-2),
    scheduler_params=dict(step_size=10, gamma=0.9),
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='sparsemax',
    verbose=0
)

clf.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    max_epochs=100, patience=20,
    batch_size=256, virtual_batch_size=128
)

train_time = time.time() - start_time

preds_bin = clf.predict(X_test_scaled)
probs = clf.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, preds_bin)
prec = precision_score(y_test, preds_bin)
rec = recall_score(y_test, preds_bin)
fnr = (1.0 - rec) * 100.0
auc = roc_auc_score(y_test, probs)
brier = brier_score_loss(y_test, probs)
ece = compute_ece(y_test, probs)

print("\nTABNET EMPIRICAL BENCHMARK RESULTS:")
print(f"Accuracy:        {acc:.3f}")
print(f"Precision:       {prec:.3f}")
print(f"Recall:          {rec:.3f}")
print(f"Exclusion Error: {fnr:.1f}%")
print(f"ROC-AUC:         {auc:.3f}")
print(f"Brier Score:     {brier:.4f}")
print(f"ECE:             {ece:.4f}")
print(f"Train Time:      {train_time:.2f}s")
