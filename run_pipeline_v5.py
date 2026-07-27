"""
Pipeline V5 — Enhanced experiments for paper revision.
Adds: EBM baseline, Fairness Evaluation, Calibration Analysis, Detailed Error Analysis.
Runs on top of the same data preprocessing as run_pipeline.py.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, brier_score_loss,
    classification_report, precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
import shap
import warnings
import scipy.stats as stats
from sklearn.model_selection import learning_curve
import time

warnings.filterwarnings('ignore')

# Set aesthetic parameters
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12
})

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)
os.makedirs('tables', exist_ok=True)

# =============================================================
# Phase 1: Load & Preprocess (same as v4, but keep gender cols)
# =============================================================
print("=" * 60)
print("PIPELINE V5: Enhanced Experiments for Paper Revision")
print("=" * 60)

print("\n1. Loading and Preprocessing Data...")
df_raw = pd.read_csv('data/train.csv')

# Save gender info BEFORE dropping columns (for fairness analysis)
# We need to track gender at the individual level, but our target is household-level
# parentesco1 == 1 means head of household
gender_head = df_raw.loc[df_raw['parentesco1'] == 1, ['idhogar', 'male']].copy()
gender_head.columns = ['idhogar', 'head_male']

# Save area info (urban=area1, rural=area2) 
area_info = df_raw.loc[df_raw['parentesco1'] == 1, ['idhogar', 'area1']].copy()
area_info.columns = ['idhogar', 'is_urban']

# Merge gender/area back to main df before dropping idhogar
df = df_raw.copy()
df = df.merge(gender_head, on='idhogar', how='left')
df = df.merge(area_info, on='idhogar', how='left')

df['target'] = df['Target'].apply(lambda x: 1 if x <= 2 else 0)

# Keep original Target for multi-class error analysis
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

# Separate protected attributes and multi-class target before model features
protected_gender = df['head_male'].values
protected_area = df['is_urban'].values
target_original = df['target_original'].values

X = df.drop(columns=['target', 'head_male', 'is_urban', 'target_original'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Get corresponding protected attributes for test set
test_indices = X_test.index
gender_test = protected_gender[test_indices]
area_test = protected_area[test_indices]
target_orig_test = target_original[test_indices]

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

# =============================================================
# Phase 2: Train ALL Models (including EBM)
# =============================================================
print("\n2. Training Models (LR, RF, XGB, XGB-Welfare, LightGBM, CatBoost, EBM)...")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'XGBoost (Welfare)': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=4.0),
    'LightGBM': lgb.LGBMClassifier(random_state=42, n_estimators=100, verbose=-1),
    'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=0),
    'EBM': ExplainableBoostingClassifier(random_state=42, n_jobs=-1, interactions=0, outer_bags=1)
}

results = []
predictions = {}
probabilities = {}
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

for name, model in models.items():
    print(f"  Training {name}...")
    X_tr = X_train_scaled if name == 'Logistic Regression' else X_train
    X_te = X_test_scaled if name == 'Logistic Regression' else X_test

    # Measure averaged timing across 5 runs for stability (1 run for EBM due to computational complexity)
    t_train_list = []
    t_infer_list = []
    num_runs = 1 if name == 'EBM' else 5
    for run_i in range(num_runs):
        st = time.time()
        model.fit(X_tr, y_train)
        et = time.time()
        t_train_list.append(et - st)

        si = time.time()
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        ei = time.time()
        t_infer_list.append(ei - si)

    start_train = np.mean(t_train_list)
    std_train = np.std(t_train_list)
    start_infer = np.mean(t_infer_list)
    std_infer = np.std(t_infer_list)

    predictions[name] = y_pred
    probabilities[name] = y_prob

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    fnr = 1.0 - rec

    cv_X = X_train_scaled if name == 'Logistic Regression' else X_train
    cv_scores = cross_val_score(model, cv_X, y_train, cv=5, scoring='roc_auc', n_jobs=-1)

    results.append({
        'Model': name,
        'Accuracy': round(acc, 3),
        'Precision': round(prec, 3),
        'Recall': round(rec, 3),
        'Exclusion Err (FNR)': f"{fnr*100:.1f}%",
        'F1-Score': round(f1, 3),
        'ROC-AUC (Test)': round(roc_auc, 3),
        'Brier Score': round(brier, 4),
        'CV_ROC-AUC': f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}",
        'Train Time (s)': f"{start_train:.3f} ± {std_train:.3f}",
        'Inference Time (s)': f"{start_infer:.3f} ± {std_infer:.3f}"
    })

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

ax_roc.plot([0, 1], [0, 1], 'k--')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curve Comparison')
ax_roc.legend(fontsize=8)
fig_roc.savefig('figures/Figure_4_ROC_Curve.png', bbox_inches='tight', dpi=150)
plt.close(fig_roc)

results_df = pd.DataFrame(results)
results_df.to_csv('tables/Table_5_Performance_Comparison.csv', index=False)
runtime_df = results_df[['Model', 'Train Time (s)', 'Inference Time (s)']]
runtime_df.to_csv('tables/Table_9_Runtime.csv', index=False)
print("\n--- Performance Results ---")
print(results_df.to_string(index=False))

# =============================================================
# Phase 3: McNemar's Test (updated with EBM)
# =============================================================
print("\n3. Running McNemar's Test...")
baseline_model = 'XGBoost (Welfare)'
mcnemar_results = []
for name in models.keys():
    if name != baseline_model:
        b = np.sum((predictions[baseline_model] == y_test) & (predictions[name] != y_test))
        c = np.sum((predictions[baseline_model] != y_test) & (predictions[name] == y_test))

        chi2 = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0
        p_val = stats.chi2.sf(chi2, 1)

        mcnemar_results.append({
            'Model 1': baseline_model,
            'Model 2': name,
            'Chi-squared': round(chi2, 2),
            'p-value': f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.3f}",
            'Significance': 'Yes' if p_val < 0.05 else 'No'
        })
mcnemar_df = pd.DataFrame(mcnemar_results)
mcnemar_df.to_csv('tables/Table_7_Statistical_Tests.csv', index=False)
print(mcnemar_df.to_string(index=False))

# =============================================================
# Phase 3.5: Confusion Matrix & Borderline Cases
# =============================================================
print("\n3.5 Running Error Analysis on best model...")
best_model_name = 'XGBoost (Welfare)'
best_y_pred = predictions[best_model_name]
best_y_prob = probabilities[best_model_name]

cm = confusion_matrix(y_test, best_y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Poor', 'Poor'], yticklabels=['Non-Poor', 'Poor'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title(f'Confusion Matrix: {best_model_name}')
plt.savefig('figures/Figure_13_Confusion_Matrix.png', bbox_inches='tight', dpi=150)
plt.close()

# Borderline analysis
borderline_mask = (best_y_prob > 0.4) & (best_y_prob < 0.6)
misclassified_mask = (best_y_pred != y_test.values)
borderline_misclassified = borderline_mask & misclassified_mask
total_misclassified = misclassified_mask.sum()
if total_misclassified > 0:
    pct_borderline = (borderline_misclassified.sum() / total_misclassified) * 100
    print(f"  Borderline misclassified: {pct_borderline:.1f}% of all errors (prob 0.4-0.6).")

# =============================================================
# Phase 4: SHAP Analysis
# =============================================================
print("\n4. Generating SHAP Analysis...")
best_model = models['XGBoost (Welfare)']
X_shap = X_train.sample(min(2000, len(X_train)), random_state=42)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer(X_shap)

plt.figure()
shap.summary_plot(shap_values, X_shap, show=False)
plt.savefig('figures/Figure_5_SHAP_Summary.png', bbox_inches='tight', dpi=150)
plt.close()

# Waterfall plot for a single household
plt.figure()
shap.plots.waterfall(shap_values[0], show=False)
plt.savefig('figures/Figure_7_Waterfall_Plot.png', bbox_inches='tight', dpi=150)
plt.close()

# Dependence plot for top feature
top_feature = X_shap.columns[np.argsort(-np.abs(shap_values.values).mean(0))[0]]
plt.figure()
shap.dependence_plot(top_feature, shap_values.values, X_shap, show=False)
plt.savefig('figures/Figure_8_Dependence_Plot.png', bbox_inches='tight', dpi=150)
plt.close()

# =============================================================
# Phase 5: EBM Global Explanation
# =============================================================
print("\n5. Generating EBM Global Explanation...")
ebm_model = models['EBM']
ebm_global = ebm_model.explain_global()

# Extract feature importances from EBM
ebm_names = ebm_global.data()['names']
ebm_scores = ebm_global.data()['scores']

# Sort by importance
sorted_idx = np.argsort(ebm_scores)[-15:]  # top 15
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(range(len(sorted_idx)), [ebm_scores[i] for i in sorted_idx],
        color='#4C72B0', edgecolor='white')
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels([ebm_names[i] for i in sorted_idx], fontsize=10)
ax.set_xlabel('Mean Absolute Score')
ax.set_title('EBM Global Feature Importance (Top 15)')
plt.tight_layout()
fig.savefig('figures/Figure_17_EBM_Global_Explanation.png', bbox_inches='tight', dpi=150)
plt.close(fig)

# =============================================================
# Phase 6: Calibration Analysis
# =============================================================
print("\n6. Generating Calibration Analysis...")

# Calibration curves for top models
calibration_models = ['XGBoost (Welfare)', 'XGBoost', 'Logistic Regression', 'EBM', 'Random Forest']
fig_cal, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
brier_scores_table = []

for i, name in enumerate(calibration_models):
    y_prob_cal = probabilities[name]
    brier = brier_score_loss(y_test, y_prob_cal)
    brier_scores_table.append({'Model': name, 'Brier Score': round(brier, 4)})

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_prob_cal, n_bins=10, strategy='uniform'
    )
    ax1.plot(mean_predicted_value, fraction_of_positives,
             marker='o', label=f'{name} (Brier={brier:.4f})',
             color=colors[i], linewidth=2, markersize=5)

    # Histogram of predicted probabilities
    ax2.hist(y_prob_cal, bins=20, alpha=0.4, label=name, color=colors[i], edgecolor='white')

ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
ax1.set_xlabel('Mean Predicted Probability')
ax1.set_ylabel('Fraction of Positives')
ax1.set_title('Reliability Diagram (Calibration Curve)')
ax1.legend(fontsize=8, loc='lower right')
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Predicted Probability')
ax2.set_ylabel('Count')
ax2.set_title('Distribution of Predicted Probabilities')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig_cal.savefig('figures/Figure_14_Calibration_Curve.png', bbox_inches='tight', dpi=150)
plt.close(fig_cal)

# Save calibration table
brier_df = pd.DataFrame(brier_scores_table)
brier_df.to_csv('tables/Table_11_Calibration.csv', index=False)
print("  Brier Scores:")
print(brier_df.to_string(index=False))

# =============================================================
# Phase 7: Fairness Analysis (Gender)
# =============================================================
print("\n7. Running Fairness Analysis (Gender)...")

def compute_fairness_metrics(y_true, y_pred, y_prob, protected, group_labels):
    """Compute fairness metrics for each group."""
    groups = np.unique(protected[~np.isnan(protected)])
    metrics = []

    for g in groups:
        mask = protected == g
        if mask.sum() == 0:
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]

        n = mask.sum()
        positive_rate = y_p.mean()  # Demographic Parity = P(Y_hat=1)

        # Among actually poor (y=1), what fraction correctly identified (TPR = Equal Opportunity)
        poor_mask = y_t == 1
        if poor_mask.sum() > 0:
            tpr = y_p[poor_mask].mean()
            fnr = 1.0 - tpr
        else:
            tpr = np.nan
            fnr = np.nan

        # Among actually non-poor (y=0), what fraction incorrectly flagged
        non_poor_mask = y_t == 0
        if non_poor_mask.sum() > 0:
            fpr = y_p[non_poor_mask].mean()
        else:
            fpr = np.nan

        acc = accuracy_score(y_t, y_p)

        label = group_labels.get(g, str(g))
        metrics.append({
            'Group': label,
            'N': n,
            'Accuracy': round(acc, 3),
            'Selection Rate (DP)': round(positive_rate, 3),
            'TPR (Eq. Opp.)': round(tpr, 3) if not np.isnan(tpr) else 'N/A',
            'FNR (Exclusion)': f"{fnr*100:.1f}%" if not np.isnan(fnr) else 'N/A',
            'FPR (Inclusion)': f"{fpr*100:.1f}%" if not np.isnan(fpr) else 'N/A'
        })

    return pd.DataFrame(metrics)


# Fairness by gender for XGBoost Welfare vs LR
fairness_all = []
for model_name in ['XGBoost (Welfare)', 'Logistic Regression', 'EBM']:
    y_p = predictions[model_name]
    y_pr = probabilities[model_name]
    gender_labels = {1.0: 'Male-headed HH', 0.0: 'Female-headed HH'}
    fair_df = compute_fairness_metrics(
        y_test.values, y_p, y_pr, gender_test.astype(float), gender_labels
    )
    fair_df['Model'] = model_name
    fairness_all.append(fair_df)

fairness_combined = pd.concat(fairness_all, ignore_index=True)
fairness_combined.to_csv('tables/Table_10_Fairness_Metrics.csv', index=False)
print("  Fairness Metrics (Gender):")
print(fairness_combined.to_string(index=False))

# Fairness visualization
fig_fair, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax_i, metric, title in zip(
    axes,
    ['Selection Rate (DP)', 'TPR (Eq. Opp.)', 'Accuracy'],
    ['Demographic Parity\n(Selection Rate)', 'Equal Opportunity\n(True Positive Rate)', 'Accuracy']
):
    # Convert to numeric for plotting
    plot_data = fairness_combined[['Model', 'Group', metric]].copy()
    plot_data[metric] = pd.to_numeric(plot_data[metric].astype(str).str.replace('%', ''), errors='coerce')

    pivot = plot_data.pivot(index='Model', columns='Group', values=metric)
    pivot.plot(kind='bar', ax=ax_i, edgecolor='white', width=0.7)
    ax_i.set_title(title, fontsize=11, fontweight='bold')
    ax_i.set_ylabel('Score')
    ax_i.set_xticklabels(ax_i.get_xticklabels(), rotation=25, ha='right', fontsize=9)
    ax_i.legend(fontsize=8)
    ax_i.grid(axis='y', alpha=0.3)
    ax_i.set_ylim(0, 1.05)

plt.tight_layout()
fig_fair.savefig('figures/Figure_15_Fairness_Analysis.png', bbox_inches='tight', dpi=150)
plt.close(fig_fair)

# =============================================================
# Phase 8: Detailed Error Analysis by Subgroup
# =============================================================
print("\n8. Running Detailed Error Analysis by Subgroup...")

y_pred_best = predictions['XGBoost (Welfare)']
y_prob_best = probabilities['XGBoost (Welfare)']
misclass = y_pred_best != y_test.values

# 8a. Error by original poverty level
error_by_target = []
for t in sorted(np.unique(target_orig_test)):
    mask = target_orig_test == t
    labels = {1: 'Extreme Poverty', 2: 'Moderate Poverty', 3: 'Vulnerable', 4: 'Non-Vulnerable'}
    n = mask.sum()
    err_rate = misclass[mask].mean()
    error_by_target.append({
        'Original Class': labels.get(t, str(t)),
        'N': n,
        'Error Rate': f"{err_rate*100:.1f}%",
        'Correct': f"{(1-err_rate)*100:.1f}%"
    })

error_target_df = pd.DataFrame(error_by_target)
print("  Error by Original Poverty Level:")
print(error_target_df.to_string(index=False))

# 8b. Error by education level of household head (binned)
edjefe_test = X_test['edjefe'].values
edu_bins = pd.cut(edjefe_test, bins=[-1, 0, 6, 12, 25], labels=['None', 'Primary (1-6)', 'Secondary (7-12)', 'Higher (13+)'])

error_by_edu = []
for cat in edu_bins.categories:
    mask = edu_bins == cat
    if mask.sum() == 0:
        continue
    n = mask.sum()
    err_rate = misclass[mask].mean()
    error_by_edu.append({
        'Education Level': cat,
        'N': n,
        'Error Rate': f"{err_rate*100:.1f}%"
    })

error_edu_df = pd.DataFrame(error_by_edu)

# 8c. Error by area (urban/rural)
error_by_area = []
for val, label in [(1.0, 'Urban'), (0.0, 'Rural')]:
    mask = area_test == val
    if mask.sum() == 0:
        continue
    n = mask.sum()
    err_rate = misclass[mask].mean()
    fnr_group = 1.0 - recall_score(y_test.values[mask], y_pred_best[mask], zero_division=0)
    error_by_area.append({
        'Area': label,
        'N': n,
        'Error Rate': f"{err_rate*100:.1f}%",
        'FNR (Exclusion)': f"{fnr_group*100:.1f}%"
    })

error_area_df = pd.DataFrame(error_by_area)

# Combine and save
error_combined = {
    'By Poverty Level': error_target_df,
    'By Education': error_edu_df,
    'By Area': error_area_df
}
with open('tables/Table_12_Error_By_Subgroup.csv', 'w') as f:
    for section, section_df in error_combined.items():
        f.write(f"# {section}\n")
        section_df.to_csv(f, index=False)
        f.write("\n")

# Visualization: Error by subgroup
fig_err, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: Error by original poverty level
ax = axes[0]
err_vals = [float(v.replace('%', '')) for v in error_target_df['Error Rate']]
colors_err = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71']
ax.barh(error_target_df['Original Class'], err_vals, color=colors_err[:len(err_vals)], edgecolor='white')
ax.set_xlabel('Error Rate (%)')
ax.set_title('Error by Poverty Level', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(err_vals):
    ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=10)

# Plot 2: Error by education
ax = axes[1]
edu_vals = [float(v.replace('%', '')) for v in error_edu_df['Error Rate']]
ax.barh(error_edu_df['Education Level'], edu_vals, color='#3498db', edgecolor='white')
ax.set_xlabel('Error Rate (%)')
ax.set_title('Error by Education Level', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(edu_vals):
    ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=10)

# Plot 3: Error by area
ax = axes[2]
area_vals = [float(v.replace('%', '')) for v in error_area_df['Error Rate']]
ax.barh(error_area_df['Area'], area_vals, color=['#9b59b6', '#1abc9c'], edgecolor='white')
ax.set_xlabel('Error Rate (%)')
ax.set_title('Error by Area', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(area_vals):
    ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=10)

plt.tight_layout()
fig_err.savefig('figures/Figure_16_Error_By_Subgroup.png', bbox_inches='tight', dpi=150)
plt.close(fig_err)

# =============================================================
# Phase 9: Ablation Study (same as v4)
# =============================================================
print("\n9. Running Ablation Study (Alpha Sweep)...")
alphas = [1, 2, 3, 4, 5, 6]
ablation_results = []

for alpha in alphas:
    m = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                          random_state=42, scale_pos_weight=float(alpha))
    m.fit(X_train, y_train)
    y_pred_ab = m.predict(X_test)

    acc = accuracy_score(y_test, y_pred_ab)
    rec = recall_score(y_test, y_pred_ab)
    prec = precision_score(y_test, y_pred_ab)

    ablation_results.append({
        'Alpha': alpha,
        'Accuracy': round(acc, 3),
        'Recall': round(rec, 3),
        'Precision': round(prec, 3)
    })

ablation_df = pd.DataFrame(ablation_results)
ablation_df.to_csv('tables/Table_8_Ablation_Study.csv', index=False)

plt.figure(figsize=(8, 6))
plt.plot(ablation_df['Alpha'], ablation_df['Accuracy'], marker='o', label='Accuracy', linewidth=2)
plt.plot(ablation_df['Alpha'], ablation_df['Recall'], marker='s', label='Recall', linewidth=2)
plt.plot(ablation_df['Alpha'], ablation_df['Precision'], marker='^', label='Precision', linewidth=2)
plt.xlabel('Welfare Penalty (α)')
plt.ylabel('Score')
plt.title('Sensitivity Analysis of Asymmetric Welfare Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('figures/Figure_12_Ablation.png', bbox_inches='tight', dpi=150)
plt.close()

# =============================================================
# Phase 10: Learning Curve
# =============================================================
print("\n10. Generating Learning Curve...")
train_sizes, train_scores, test_scores = learning_curve(
    models['XGBoost (Welfare)'], X_train, y_train, cv=5, scoring='roc_auc',
    n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score", linewidth=2)
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score", linewidth=2)
plt.title("Learning Curve (XGBoost Welfare)")
plt.xlabel("Training examples")
plt.ylabel("ROC-AUC Score")
plt.legend(loc="best")
plt.grid(True, alpha=0.3)
plt.savefig('figures/Figure_11_Learning_Curve.png', bbox_inches='tight', dpi=150)
plt.close()

# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("PIPELINE V5 COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nNew outputs created:")
print("  Figures:")
print("    - figures/Figure_14_Calibration_Curve.png")
print("    - figures/Figure_15_Fairness_Analysis.png")
print("    - figures/Figure_16_Error_By_Subgroup.png")
print("    - figures/Figure_17_EBM_Global_Explanation.png")
print("  Tables:")
print("    - tables/Table_10_Fairness_Metrics.csv")
print("    - tables/Table_11_Calibration.csv")
print("    - tables/Table_12_Error_By_Subgroup.csv")
print("\nUpdated outputs:")
print("    - tables/Table_5_Performance_Comparison.csv (with EBM + Brier)")
print("    - tables/Table_7_Statistical_Tests.csv (with EBM)")
print("    - figures/Figure_4_ROC_Curve.png (with EBM)")
