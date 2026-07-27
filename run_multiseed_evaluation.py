import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, brier_score_loss

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

def pwc_obj(alpha=4.0, gamma=2.0):
    def custom_obj(preds, dtrain):
        labels = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-preds))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        g = np.zeros_like(p)
        h = np.zeros_like(p)
        pos = (labels == 1)
        g[pos] = alpha * ((1.0 - p[pos]) ** gamma) * (p[pos] - 1.0 + gamma * p[pos] * np.log(p[pos]))
        h[pos] = alpha * ((1.0 - p[pos]) ** (gamma - 1.0)) * p[pos] * (1.0 - p[pos]) * (1.0 + gamma * np.log(p[pos]))
        neg = (labels == 0)
        g[neg] = (p[neg] ** gamma) * (p[neg] - gamma * (1.0 - p[neg]) * np.log(1.0 - p[neg]))
        h[neg] = (p[neg] ** gamma) * (1.0 - p[neg]) * (1.0 + gamma * np.log(1.0 - p[neg]))
        h = np.maximum(h, 1e-6)
        return g, h
    return custom_obj

seeds = [42, 101, 202, 303, 505]
accs, recs, aucs, briers, eces = [], [], [], [], []

for seed in seeds:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_accs, fold_recs, fold_aucs, fold_briers, fold_eces = [], [], [], [], []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        params = {
            'max_depth': 6, 'eta': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'lambda': 1.0, 'seed': seed, 'disable_default_eval_metric': 1
        }
        bst = xgb.train(params, dtrain, num_boost_round=300, obj=pwc_obj(4.0, 2.0))
        raw_preds = bst.predict(dval)
        probs = 1.0 / (1.0 + np.exp(-raw_preds))
        preds_bin = (probs >= 0.5).astype(int)
        
        fold_accs.append(accuracy_score(y_val, preds_bin))
        fold_recs.append(recall_score(y_val, preds_bin))
        fold_aucs.append(roc_auc_score(y_val, probs))
        fold_briers.append(brier_score_loss(y_val, probs))
        fold_eces.append(compute_ece(y_val, probs))
        
    accs.append(np.mean(fold_accs))
    recs.append(np.mean(fold_recs))
    aucs.append(np.mean(fold_aucs))
    briers.append(np.mean(fold_briers))
    eces.append(np.mean(fold_eces))

print("MULTI-SEED STABILITY RESULTS (5 RANDOM SEEDS):")
print(f"Accuracy:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
print(f"Recall:    {np.mean(recs):.4f} +/- {np.std(recs):.4f}")
print(f"ROC-AUC:   {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
print(f"Brier:     {np.mean(briers):.4f} +/- {np.std(briers):.4f}")
print(f"ECE:       {np.mean(eces):.4f} +/- {np.std(eces):.4f}")
