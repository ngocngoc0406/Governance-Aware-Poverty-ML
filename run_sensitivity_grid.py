import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, recall_score, brier_score_loss, roc_auc_score
import xgboost as xgb

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

os.makedirs('figures', exist_ok=True)
os.makedirs('tables', exist_ok=True)

# Load raw dataset
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

X = df.drop(columns=['target'])
y = df['target']

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

def custom_pwc_obj(alpha=4.0, gamma=2.0):
    def objective(preds, dtrain):
        labels = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-preds))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        
        g_pos = alpha * ((1.0 - p) ** gamma) * (p - 1.0 + gamma * p * np.log(p))
        h_pos = alpha * ((1.0 - p) ** np.maximum(gamma - 1.0, 0)) * p * (1.0 - p) * np.maximum(1.0 + gamma * np.log(p), 0.1)
        
        g_neg = (p ** gamma) * (p - gamma * (1.0 - p) * np.log(1.0 - p))
        h_neg = (p ** gamma) * (1.0 - p) * np.maximum(1.0 + gamma * np.log(1.0 - p), 0.1)
        
        grad = np.where(labels == 1, g_pos, g_neg)
        hess = np.where(labels == 1, h_pos, h_neg)
        
        hess = np.maximum(hess, 1e-6)
        return grad, hess
    return objective

# 1. Sensitivity Analysis Grid Search
alphas = [1.0, 2.0, 3.0, 4.0, 5.0]
gammas = [0.0, 1.0, 2.0, 3.0, 4.0]

ece_grid = np.zeros((len(alphas), len(gammas)))
fnr_grid = np.zeros((len(alphas), len(gammas)))

print("Running Grid Search Sensitivity Analysis across alpha x gamma...")
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

params = {
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'lambda': 1.0,
    'seed': 42,
    'disable_default_eval_metric': 1
}

for i, a in enumerate(alphas):
    for j, g in enumerate(gammas):
        bst = xgb.train(
            params, dtrain, num_boost_round=300,
            obj=custom_pwc_obj(alpha=a, gamma=g)
        )
        preds_margin = bst.predict(dtest)
        probs = 1.0 / (1.0 + np.exp(-preds_margin))
        preds_bin = (probs >= 0.5).astype(int)
        
        fnr = (1.0 - recall_score(y_test, preds_bin)) * 100.0
        ece = compute_ece(y_test.values, probs)
        
        ece_grid[i, j] = ece
        fnr_grid[i, j] = fnr

# Save Grid Search Sensitivity Matrix to CSV
sensitivity_df = pd.DataFrame(
    ece_grid,
    index=[f'alpha={a}' for a in alphas],
    columns=[f'gamma={g}' for g in gammas]
)
sensitivity_df.to_csv('tables/Table_14_Sensitivity_Grid_ECE.csv')
print("Sensitivity ECE Matrix saved to tables/Table_14_Sensitivity_Grid_ECE.csv")
print(sensitivity_df)

# Plot Sensitivity Heatmaps
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(
    ece_grid, annot=True, fmt=".4f", cmap="YlGnBu",
    xticklabels=gammas, yticklabels=alphas, ax=axes[0]
)
axes[0].set_title("Expected Calibration Error (ECE) Sensitivity Heatmap")
axes[0].set_xlabel("Focal Parameter (gamma)")
axes[0].set_ylabel("Asymmetric Weight (alpha)")

sns.heatmap(
    fnr_grid, annot=True, fmt=".1f", cmap="OrRd",
    xticklabels=gammas, yticklabels=alphas, ax=axes[1]
)
axes[1].set_title("Exclusion Error (FNR %) Sensitivity Heatmap")
axes[1].set_xlabel("Focal Parameter (gamma)")
axes[1].set_ylabel("Asymmetric Weight (alpha)")

plt.tight_layout()
plt.savefig('figures/Figure_20_Sensitivity_Heatmap.png', dpi=300)
plt.close()
print("Sensitivity Heatmap figure saved to figures/Figure_20_Sensitivity_Heatmap.png")

# 2. Generate Reliability Diagram (Calibration Curve)
bst_pwc = xgb.train(
    params, dtrain, num_boost_round=300,
    obj=custom_pwc_obj(alpha=4.0, gamma=2.0)
)
pwc_probs = 1.0 / (1.0 + np.exp(-bst_pwc.predict(dtest)))

bst_std = xgb.train(
    {**params, 'objective': 'binary:logistic'}, dtrain, num_boost_round=300
)
std_probs = bst_std.predict(dtest)

plt.figure(figsize=(7, 6))
fraction_of_positives_std, mean_predicted_value_std = calibration_curve(y_test, std_probs, n_bins=10)
fraction_of_positives_pwc, mean_predicted_value_pwc = calibration_curve(y_test, pwc_probs, n_bins=10)

plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
plt.plot(mean_predicted_value_std, fraction_of_positives_std, "s-", label=f"Standard XGBoost (ECE={compute_ece(y_test.values, std_probs):.4f})", color="tab:blue")
plt.plot(mean_predicted_value_pwc, fraction_of_positives_pwc, "o-", label=f"Proposed PWC-Loss (ECE={compute_ece(y_test.values, pwc_probs):.4f})", color="tab:green", linewidth=2)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Reliability Diagram (Probability Calibration Curves)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('figures/Figure_19_Reliability_Diagram.png', dpi=300)
plt.close()
print("Reliability Diagram figure saved to figures/Figure_19_Reliability_Diagram.png")
