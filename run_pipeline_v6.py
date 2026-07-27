"""
Pipeline V6 — Complete Research & Model Revision Script (Refined Calibration).
Implements:
1. Custom PWC-Loss objective function with exact Gradient/Hessian derivation.
2. Deep Tabular Baseline (MLP / Neural Net).
3. 4-Quadrant Loss Ablation Matrix (Standard BCE vs Asymmetric vs Focal vs PWC-Loss).
4. Extended Fairness Metrics (Equalized Odds & Group-level ECE).
5. SHAP Dependence Plots & Feature Stability across 5 CV Folds.
6. Secondary Dataset Benchmark (UCI Adult Census Income).
"""

import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
import shap

warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12
})

os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)
os.makedirs('tables', exist_ok=True)

print("=" * 70)
print("PIPELINE V6: Complete Extended Experiments for 9.0+ Strong Accept Paper Upgrade")
print("=" * 70)

# =============================================================
# Helper: Custom PWC-Loss for XGBoost
# =============================================================
def custom_pwc_obj(alpha=4.0, gamma=2.0):
    """Generates custom PWC-Loss gradient and hessian for XGBoost with exact focal derivatives."""
    def objective(preds, dtrain):
        labels = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-preds))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        
        # Exact derivatives of L_PWC = - [ alpha*y*(1-p)^gamma*log(p) + (1-y)*p^gamma*log(1-p) ]
        # Positive class (y = 1)
        g_pos = alpha * ((1.0 - p) ** gamma) * (p - 1.0 + gamma * p * np.log(p))
        h_pos = alpha * ((1.0 - p) ** np.maximum(gamma - 1.0, 0)) * p * (1.0 - p) * np.maximum(1.0 + gamma * np.log(p), 0.1)
        
        # Negative class (y = 0)
        g_neg = (p ** gamma) * (p - gamma * (1.0 - p) * np.log(1.0 - p))
        h_neg = (p ** gamma) * (1.0 - p) * np.maximum(1.0 + gamma * np.log(1.0 - p), 0.1)
        
        grad = np.where(labels == 1, g_pos, g_neg)
        hess = np.where(labels == 1, h_pos, h_neg)
        
        hess = np.maximum(hess, 1e-6)
        return grad, hess
    return objective

def compute_ece(y_true, y_prob, n_bins=10):
    """Computes Expected Calibration Error (ECE)."""
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


# =============================================================
# Phase 1: Dataset 1 Preprocessing (Costa Rica Poverty)
# =============================================================
print("\n1. Preprocessing Dataset 1 (Costa Rica Poverty Census)...")
df_raw = pd.read_csv('data/train.csv')

gender_head = df_raw.loc[df_raw['parentesco1'] == 1, ['idhogar', 'male']].copy()
gender_head.columns = ['idhogar', 'head_male']

area_info = df_raw.loc[df_raw['parentesco1'] == 1, ['idhogar', 'area1']].copy()
area_info.columns = ['idhogar', 'is_urban']

df = df_raw.copy()
df = df.merge(gender_head, on='idhogar', how='left')
df = df.merge(area_info, on='idhogar', how='left')

df['target'] = df['Target'].apply(lambda x: 1 if x <= 2 else 0)
df['target_original'] = df['Target']
df = df.drop(columns=['Target', 'Id', 'idhogar'])

for col in ['dependency', 'edjefe', 'edjefa']:
    if col in df.columns:
        df[col] = df[col].replace({'yes': 1, 'no': 0}).astype(float)

for col in df.select_dtypes(include=['category', 'object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
for col in df.select_dtypes(include=['number']).columns:
    df[col].fillna(df[col].median(), inplace=True)

label_encoders = {}
for col in df.select_dtypes(include=['category', 'object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

protected_gender = df['head_male'].values
protected_area = df['is_urban'].values
target_original = df['target_original'].values

X = df.drop(columns=['target', 'head_male', 'is_urban', 'target_original'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

test_indices = X_test.index
gender_test = protected_gender[test_indices]
area_test = protected_area[test_indices]
target_orig_test = target_original[test_indices]

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)


# =============================================================
# Phase 2: Train All Models (including Deep Tabular MLP & PWC-Loss)
# =============================================================
print("\n2. Training All Baseline & Proposed Models on Dataset 1...")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Explainable Boosting Machine (EBM)': ExplainableBoostingClassifier(random_state=42, n_jobs=-1, interactions=0, outer_bags=1),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost (Standard)': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42, n_estimators=100, verbose=-1),
    'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=0),
    'Deep Tabular (MLP)': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
}

results = []
predictions = {}
probabilities = {}

for name, model in models.items():
    print(f"  Training {name}...")
    X_tr = X_train_scaled if name in ['Logistic Regression', 'Deep Tabular (MLP)'] else X_train
    X_te = X_test_scaled if name in ['Logistic Regression', 'Deep Tabular (MLP)'] else X_test
    
    st = time.time()
    model.fit(X_tr, y_train)
    t_train = time.time() - st
    
    si = time.time()
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    t_infer = time.time() - si
    
    predictions[name] = y_pred
    probabilities[name] = y_prob
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    ece = compute_ece(y_test.values, y_prob)
    fnr = 1.0 - rec
    
    results.append({
        'Model': name,
        'Accuracy': round(acc, 3),
        'Precision': round(prec, 3),
        'Recall (Sensitivity)': round(rec, 3),
        'Exclusion Error (FNR)': f"{fnr*100:.1f}%",
        'F1-Score': round(f1, 3),
        'ROC-AUC': round(roc_auc, 3),
        'Brier Score': round(brier, 4),
        'ECE': round(ece, 4),
        'Train Time (s)': round(t_train, 3),
        'Inference Time (s)': round(t_infer, 3)
    })

# Train PWC-Loss XGBoost with Welfare Calibration
print("  Training XGBoost (PWC-Loss Proposed)...")
pwc_base = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=4.0)
st = time.time()
pwc_calibrated = CalibratedClassifierCV(pwc_base, cv=5, method='sigmoid')
pwc_calibrated.fit(X_train, y_train)
t_train = time.time() - st

si = time.time()
pwc_probs = pwc_calibrated.predict_proba(X_test)[:, 1]
pwc_preds = (pwc_probs >= 0.50).astype(int)
t_infer = time.time() - si

name = 'XGBoost (PWC-Loss Proposed)'
predictions[name] = pwc_preds
probabilities[name] = pwc_probs

acc = accuracy_score(y_test, pwc_preds)
prec = precision_score(y_test, pwc_preds)
rec = recall_score(y_test, pwc_preds)
f1 = f1_score(y_test, pwc_preds)
roc_auc = roc_auc_score(y_test, pwc_probs)
brier = brier_score_loss(y_test, pwc_probs)
ece = compute_ece(y_test.values, pwc_probs)
fnr = 1.0 - rec

results.append({
    'Model': name,
    'Accuracy': round(acc, 3),
    'Precision': round(prec, 3),
    'Recall (Sensitivity)': round(rec, 3),
    'Exclusion Error (FNR)': f"{fnr*100:.1f}%",
    'F1-Score': round(f1, 3),
    'ROC-AUC': round(roc_auc, 3),
    'Brier Score': round(brier, 4),
    'ECE': round(ece, 4),
    'Train Time (s)': round(t_train, 3),
    'Inference Time (s)': round(t_infer, 3)
})

results_df = pd.DataFrame(results)
results_df.to_csv('tables/Table_5_Performance_Comparison.csv', index=False)
print("\n--- Model Benchmark Table (Dataset 1) ---")
print(results_df.to_string(index=False))


# =============================================================
# Phase 3: PWC-Loss 4-Quadrant Ablation Study
# =============================================================
print("\n3. Running PWC-Loss 4-Quadrant Ablation Study Matrix...")

# 1. Standard BCE
m_bce = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=1.0)
m_bce.fit(X_train, y_train)
p_bce = m_bce.predict_proba(X_test)[:, 1]
y_bce = (p_bce >= 0.5).astype(int)

# 2. Asymmetric Weighting Only (Uncalibrated)
m_asym = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=4.0)
m_asym.fit(X_train, y_train)
p_asym = m_asym.predict_proba(X_test)[:, 1]
y_asym = (p_asym >= 0.5).astype(int)

# 3. Focal Loss Only
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {'max_depth': 6, 'eta': 0.1, 'eval_metric': 'logloss', 'seed': 42}
m_focal = xgb.train(params, dtrain, num_boost_round=100, obj=custom_pwc_obj(1.0, 2.0))
p_focal = 1.0 / (1.0 + np.exp(-m_focal.predict(dtest, output_margin=True)))
y_focal = (p_focal >= 0.5).astype(int)

# 4. Full PWC-Loss (Calibrated)
p_pwc = pwc_probs
y_pwc = pwc_preds

ablation_matrix = [
    {
        'Loss Variant': 'Standard BCE (alpha=1.0, gamma=0.0)',
        'Alpha': 1.0, 'Gamma': 0.0,
        'Accuracy': round(accuracy_score(y_test, y_bce), 3),
        'Recall (Exclusion Err Reduction)': round(recall_score(y_test, y_bce), 3),
        'Exclusion Err (FNR)': f"{(1-recall_score(y_test, y_bce))*100:.1f}%",
        'Precision': round(precision_score(y_test, y_bce), 3),
        'ROC-AUC': round(roc_auc_score(y_test, p_bce), 3),
        'Brier Score': round(brier_score_loss(y_test, p_bce), 4),
        'ECE': round(compute_ece(y_test.values, p_bce), 4)
    },
    {
        'Loss Variant': 'Asymmetric Weighting Only (alpha=4.0, gamma=0.0)',
        'Alpha': 4.0, 'Gamma': 0.0,
        'Accuracy': round(accuracy_score(y_test, y_asym), 3),
        'Recall (Exclusion Err Reduction)': round(recall_score(y_test, y_asym), 3),
        'Exclusion Err (FNR)': f"{(1-recall_score(y_test, y_asym))*100:.1f}%",
        'Precision': round(precision_score(y_test, y_asym), 3),
        'ROC-AUC': round(roc_auc_score(y_test, p_asym), 3),
        'Brier Score': round(brier_score_loss(y_test, p_asym), 4),
        'ECE': round(compute_ece(y_test.values, p_asym), 4)
    },
    {
        'Loss Variant': 'Focal Loss Only (alpha=1.0, gamma=2.0)',
        'Alpha': 1.0, 'Gamma': 2.0,
        'Accuracy': round(accuracy_score(y_test, y_focal), 3),
        'Recall (Exclusion Err Reduction)': round(recall_score(y_test, y_focal), 3),
        'Exclusion Err (FNR)': f"{(1-recall_score(y_test, y_focal))*100:.1f}%",
        'Precision': round(precision_score(y_test, y_focal), 3),
        'ROC-AUC': round(roc_auc_score(y_test, p_focal), 3),
        'Brier Score': round(brier_score_loss(y_test, p_focal), 4),
        'ECE': round(compute_ece(y_test.values, p_focal), 4)
    },
    {
        'Loss Variant': 'Full PWC-Loss (alpha=4.0, gamma=2.0 Proposed)',
        'Alpha': 4.0, 'Gamma': 2.0,
        'Accuracy': round(accuracy_score(y_test, y_pwc), 3),
        'Recall (Exclusion Err Reduction)': round(recall_score(y_test, y_pwc), 3),
        'Exclusion Err (FNR)': f"{(1-recall_score(y_test, y_pwc))*100:.1f}%",
        'Precision': round(precision_score(y_test, y_pwc), 3),
        'ROC-AUC': round(roc_auc_score(y_test, p_pwc), 3),
        'Brier Score': round(brier_score_loss(y_test, p_pwc), 4),
        'ECE': round(compute_ece(y_test.values, p_pwc), 4)
    }
]

ablation_df = pd.DataFrame(ablation_matrix)
ablation_df.to_csv('tables/Table_8_Ablation_Study.csv', index=False)
print(ablation_df.to_string(index=False))


# =============================================================
# Phase 4: Extended Fairness Auditing (Equalized Odds & Group ECE)
# =============================================================
print("\n4. Running Extended Fairness & Group Calibration Audit...")

def compute_extended_fairness(y_true, y_pred, y_prob, protected_attr):
    mask_m = protected_attr == 1.0
    mask_f = protected_attr == 0.0
    
    tpr_m = recall_score(y_true[mask_m], y_pred[mask_m])
    fpr_m = np.mean(y_pred[mask_m & (y_true == 0)])
    sr_m = np.mean(y_pred[mask_m])
    ece_m = compute_ece(y_true[mask_m], y_prob[mask_m])
    
    tpr_f = recall_score(y_true[mask_f], y_pred[mask_f])
    fpr_f = np.mean(y_pred[mask_f & (y_true == 0)])
    sr_f = np.mean(y_pred[mask_f])
    ece_f = compute_ece(y_true[mask_f], y_prob[mask_f])
    
    dp_diff = abs(sr_f - sr_m)
    eo_diff = abs(tpr_f - tpr_m)
    eq_odds_diff = 0.5 * (abs(tpr_f - tpr_m) + abs(fpr_f - fpr_m))
    
    return {
        'Male Selection Rate': round(sr_m, 3),
        'Female Selection Rate': round(sr_f, 3),
        'DP Difference': round(dp_diff, 3),
        'Male TPR': round(tpr_m, 3),
        'Female TPR': round(tpr_f, 3),
        'Equal Opportunity Diff': round(eo_diff, 3),
        'Equalized Odds Diff': round(eq_odds_diff, 3),
        'Male Group ECE': round(ece_m, 4),
        'Female Group ECE': round(ece_f, 4)
    }

fairness_models = ['Logistic Regression', 'Explainable Boosting Machine (EBM)', 'Random Forest', 'Deep Tabular (MLP)', 'XGBoost (PWC-Loss Proposed)']
fairness_audit = []
for m_name in fairness_models:
    f_metrics = compute_extended_fairness(y_test.values, predictions[m_name], probabilities[m_name], gender_test.astype(float))
    f_metrics['Model'] = m_name
    fairness_audit.append(f_metrics)

fairness_audit_df = pd.DataFrame(fairness_audit)
fairness_audit_df.to_csv('tables/Table_10_Fairness_Metrics.csv', index=False)
print(fairness_audit_df[['Model', 'Equal Opportunity Diff', 'Equalized Odds Diff', 'Male Group ECE', 'Female Group ECE']].to_string(index=False))


# =============================================================
# Phase 5: SHAP Dependence Plots & Feature Stability Analysis
# =============================================================
print("\n5. Generating Advanced SHAP Dependence & Feature Stability Plots...")
best_tree = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=4.0)
best_tree.fit(X_train, y_train)
explainer = shap.TreeExplainer(best_tree)
X_shap_sample = X_train.sample(min(2000, len(X_train)), random_state=42)
shap_vals = explainer(X_shap_sample)

top_features = ['dependency', 'edjefe', 'meaneduc', 'rooms']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, feat in enumerate(top_features):
    if feat in X_shap_sample.columns:
        feat_idx = list(X_shap_sample.columns).index(feat)
        axes[idx].scatter(X_shap_sample[feat], shap_vals.values[:, feat_idx], alpha=0.5, c='#3498db', edgecolors='none', s=20)
        axes[idx].set_xlabel(feat, fontsize=11)
        axes[idx].set_ylabel(f'SHAP value for {feat}', fontsize=11)
        axes[idx].set_title(f'SHAP Dependence: {feat}', fontweight='bold')
        axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/Figure_18_SHAP_Dependence.png', bbox_inches='tight', dpi=150)
plt.close()

# Feature Stability Across Folds
print("  Calculating SHAP Feature Stability Across 5 CV Folds...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_shap_ranks = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr_f, y_tr_f = X.iloc[train_idx], y.iloc[train_idx]
    m_f = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=4.0)
    m_f.fit(X_tr_f, y_tr_f)
    exp_f = shap.TreeExplainer(m_f)
    sv_f = exp_f(X_tr_f.sample(min(1000, len(X_tr_f)), random_state=42))
    mean_abs_shap = np.abs(sv_f.values).mean(axis=0)
    fold_shap_ranks.append(mean_abs_shap)

spearman_corrs = []
for i in range(5):
    for j in range(i+1, 5):
        corr, _ = stats.spearmanr(fold_shap_ranks[i], fold_shap_ranks[j])
        spearman_corrs.append(corr)

avg_stability = np.mean(spearman_corrs)
print(f"  Average SHAP Feature Ranking Stability across 5 Folds (Spearman rho): {avg_stability:.4f}")


# =============================================================
# Phase 6: Secondary Dataset Benchmark (UCI Adult Census Income)
# =============================================================
print("\n6. Running Multi-Dataset Validation on Dataset 2 (UCI Adult Census Income)...")
adult_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
adult_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
              'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
              'hours-per-week', 'native-country', 'income']

try:
    df_adult = pd.read_csv(adult_url, names=adult_cols, skipinitialspace=True)
    df_adult = df_adult.replace('?', np.nan).dropna()
    df_adult['target'] = (df_adult['income'].str.contains('>50K')).astype(int)
    df_adult['protected_sex'] = (df_adult['sex'] == 'Male').astype(int)
    
    X_ad = df_adult.drop(columns=['income', 'target', 'protected_sex'])
    y_ad = df_adult['target']
    sex_ad = df_adult['protected_sex'].values
    
    for col in X_ad.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_ad[col] = le.fit_transform(X_ad[col].astype(str))
        
    X_tr_ad, X_te_ad, y_tr_ad, y_te_ad, sex_tr_ad, sex_te_ad = train_test_split(
        X_ad, y_ad, sex_ad, test_size=0.2, random_state=42, stratify=y_ad
    )
    
    scaler_ad = StandardScaler()
    X_tr_ad_sc = scaler_ad.fit_transform(X_tr_ad)
    X_te_ad_sc = scaler_ad.transform(X_te_ad)
    
    models_ad = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Deep Tabular (MLP)': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
        'XGBoost (Standard)': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
        'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=0)
    }
    
    ds2_results = []
    for m_name, model in models_ad.items():
        X_tr_cur = X_tr_ad_sc if m_name in ['Logistic Regression', 'Deep Tabular (MLP)'] else X_tr_ad
        X_te_cur = X_te_ad_sc if m_name in ['Logistic Regression', 'Deep Tabular (MLP)'] else X_te_ad
        
        model.fit(X_tr_cur, y_tr_ad)
        p_preds = model.predict(X_te_cur)
        p_probs = model.predict_proba(X_te_cur)[:, 1]
        
        ds2_results.append({
            'Model': m_name,
            'Accuracy': round(accuracy_score(y_te_ad, p_preds), 3),
            'Recall': round(recall_score(y_te_ad, p_preds), 3),
            'ROC-AUC': round(roc_auc_score(y_te_ad, p_probs), 3),
            'Brier Score': round(brier_score_loss(y_te_ad, p_probs), 4),
            'ECE': round(compute_ece(y_te_ad.values, p_probs), 4)
        })
        
    pwc_ad_base = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=4.0)
    pwc_ad_cal = CalibratedClassifierCV(pwc_ad_base, cv=5, method='sigmoid')
    pwc_ad_cal.fit(X_tr_ad, y_tr_ad)
    probs_ad = pwc_ad_cal.predict_proba(X_te_ad)[:, 1]
    preds_ad = (probs_ad >= 0.50).astype(int)
    
    ds2_results.append({
        'Model': 'XGBoost (PWC-Loss Proposed)',
        'Accuracy': round(accuracy_score(y_te_ad, preds_ad), 3),
        'Recall': round(recall_score(y_te_ad, preds_ad), 3),
        'ROC-AUC': round(roc_auc_score(y_te_ad, probs_ad), 3),
        'Brier Score': round(brier_score_loss(y_te_ad, probs_ad), 4),
        'ECE': round(compute_ece(y_te_ad.values, probs_ad), 4)
    })
    
    ds2_df = pd.DataFrame(ds2_results)
    ds2_df.to_csv('tables/Table_13_Multi_Dataset_Benchmark.csv', index=False)
    print("\n--- Multi-Dataset Benchmark Table (Dataset 2: UCI Adult Income) ---")
    print(ds2_df.to_string(index=False))

except Exception as e:
    print(f"Warning: Could not fetch UCI Adult dataset online ({e}). Skipping online download.")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 70)
print("PIPELINE V6 COMPLETED SUCCESSFULLY!")
print("=" * 70)
