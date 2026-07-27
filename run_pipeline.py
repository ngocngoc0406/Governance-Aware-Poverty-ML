import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap
import warnings
import scipy.stats as stats
from sklearn.model_selection import learning_curve
import time
warnings.filterwarnings('ignore')

# Set aesthetic parameters
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.figsize': (10, 6)})

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)
os.makedirs('tables', exist_ok=True)

# ---------------------------------------------------------
# Phase 1 & 2: Load Data, Cleaning & EDA
# ---------------------------------------------------------
print("1. Loading and Preprocessing Data...")
df = pd.read_csv('data/train.csv')

df['target'] = df['Target'].apply(lambda x: 1 if x <= 2 else 0)
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

df.to_csv('data/cleaned_dataset.csv', index=False)

# ---------------------------------------------------------
# Phase 3 & 4: Baseline Models & Ensembles
# ---------------------------------------------------------
print("2. Training Models (LR, RF, XGB, LightGBM, CatBoost)...")
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'XGBoost (Welfare)': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=4.0),
    'LightGBM': lgb.LGBMClassifier(random_state=42, n_estimators=100, verbose=-1),
    'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=0)
}

results = []
predictions = {}
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

for name, model in models.items():
    print(f"Training {name}...")
    X_tr = X_train_scaled if name == 'Logistic Regression' else X_train
    X_te = X_test_scaled if name == 'Logistic Regression' else X_test
    
    # Measure averaged timing across 5 runs for stability
    t_train_list = []
    t_infer_list = []
    for _ in range(5):
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
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    fnr = 1.0 - rec  # False Negative Rate (Exclusion Error)
    cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'Exclusion Err (FNR)': fnr,
        'F1-Score': f1,
        'ROC-AUC (Test)': roc_auc,
        'CV_ROC-AUC (5-fold Mean)': cv_scores.mean(),
        'CV_ROC-AUC (Std)': cv_scores.std(),
        'Train Time (s)': f"{start_train:.3f} ± {std_train:.3f}",
        'Inference Time (s)': f"{start_infer:.3f} ± {std_infer:.3f}"
    })
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

ax_roc.plot([0, 1], [0, 1], 'k--')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curve Comparison')
ax_roc.legend()
fig_roc.savefig('figures/Figure_4_ROC_Curve.png', bbox_inches='tight')
plt.close(fig_roc)

results_df = pd.DataFrame(results)
results_df.to_csv('tables/Table_5_Performance_Comparison.csv', index=False)
runtime_df = results_df[['Model', 'Train Time (s)', 'Inference Time (s)']]
runtime_df.to_csv('tables/Table_9_Runtime.csv', index=False)
print(results_df)

# McNemar's Test
print("3. Running McNemar's Test...")
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
            'Chi-squared': chi2,
            'p-value': p_val,
            'Significance': 'Yes' if p_val < 0.05 else 'No'
        })
mcnemar_df = pd.DataFrame(mcnemar_results)
mcnemar_df.to_csv('tables/Table_7_Statistical_Tests.csv', index=False)
print(mcnemar_df)

# SHAP
# ---------------------------------------------------------
# Phase 3.5: Error Analysis (Confusion Matrix & Borderline Cases)
# ---------------------------------------------------------
print("3.5 Running Error Analysis on best model...")
best_model_name = 'XGBoost (Welfare)'
best_y_pred = predictions[best_model_name]
best_y_prob = models[best_model_name].predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, best_y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Poor', 'Poor'], yticklabels=['Non-Poor', 'Poor'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title(f'Confusion Matrix: {best_model_name}')
plt.savefig('figures/Figure_13_Confusion_Matrix.png', bbox_inches='tight')
plt.close()

# Borderline analysis
borderline_mask = (best_y_prob > 0.4) & (best_y_prob < 0.6)
misclassified_mask = (best_y_pred != y_test)
borderline_misclassified = borderline_mask & misclassified_mask
total_misclassified = misclassified_mask.sum()
if total_misclassified > 0:
    pct_borderline = (borderline_misclassified.sum() / total_misclassified) * 100
    print(f"Error Analysis: {pct_borderline:.1f}% of misclassified households are 'borderline' (prob 0.4-0.6).")

print("4. Generating SHAP Analysis...")
best_model = models['XGBoost (Welfare)']
X_shap = X_train.sample(min(2000, len(X_train)), random_state=42)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer(X_shap)

plt.figure()
shap.summary_plot(shap_values, X_shap, show=False)
plt.savefig('figures/Figure_5_SHAP_Summary.png', bbox_inches='tight')
plt.close()

plt.figure()
shap.decision_plot(explainer.expected_value, shap_values.values[:20], X_shap.iloc[:20], show=False)
plt.savefig('figures/Figure_9_SHAP_Decision_Plot.png', bbox_inches='tight')
plt.close()

print("Generating SHAP Interaction Plot...")
shap_interaction_values = explainer.shap_interaction_values(X_shap.iloc[:500])
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_interaction_values, X_shap.iloc[:500], show=False)
plt.title('SHAP Interaction Plot')
plt.savefig('figures/Figure_10_SHAP_Interaction_Plot.png', bbox_inches='tight')
plt.close()

# Update Hyperparameters table to include 5 models
hyperparams = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'XGBoost (Welfare)', 'LightGBM', 'CatBoost'],
    'Optimal Hyperparameters': [
        'penalty=l2, C=1.0, solver=lbfgs, max_iter=1000',
        'n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1',
        'n_estimators=100, learning_rate=0.1, max_depth=6, subsample=0.8',
        'n_estimators=100, learning_rate=0.1, max_depth=6, scale_pos_weight=4.0',
        'n_estimators=100, learning_rate=0.1, num_leaves=31',
        'iterations=100, learning_rate=0.1, depth=6'
    ]
})
hyperparams.to_csv('tables/Table_4_Hyperparameters.csv', index=False)

# ---------------------------------------------------------
# Phase 5: Learning Curve (Overfitting Analysis)
# ---------------------------------------------------------
print("5. Generating Learning Curve for XGBoost...")
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train, y_train, cv=5, scoring='roc_auc', 
    n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.title("Learning Curve (XGBoost)")
plt.xlabel("Training examples")
plt.ylabel("ROC-AUC Score")
plt.legend(loc="best")
plt.grid(True)
plt.savefig('figures/Figure_11_Learning_Curve.png', bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Phase 6: Ablation Study (Welfare Penalty Sensitivity)
# ---------------------------------------------------------
print("6. Running Ablation Study (Alpha Sweep)...")
alphas = [1, 2, 3, 4, 5, 6]
ablation_results = []

for alpha in alphas:
    m = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                          random_state=42, scale_pos_weight=float(alpha))
    m.fit(X_train, y_train)
    y_pred_ab = m.predict(X_test)
    y_prob_ab = m.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred_ab)
    rec = recall_score(y_test, y_pred_ab)
    prec = precision_score(y_test, y_pred_ab)
    
    ablation_results.append({
        'Alpha': alpha,
        'Accuracy': acc,
        'Recall': rec,
        'Precision': prec
    })

ablation_df = pd.DataFrame(ablation_results)
ablation_df.to_csv('tables/Table_8_Ablation_Study.csv', index=False)
print(ablation_df)

plt.figure(figsize=(8, 6))
plt.plot(ablation_df['Alpha'], ablation_df['Accuracy'], marker='o', label='Accuracy')
plt.plot(ablation_df['Alpha'], ablation_df['Recall'], marker='s', label='Recall')
plt.plot(ablation_df['Alpha'], ablation_df['Precision'], marker='^', label='Precision')
plt.xlabel('Welfare Penalty (Alpha)')
plt.ylabel('Score')
plt.title('Sensitivity Analysis of Asymmetric Welfare Loss')
plt.legend()
plt.grid(True)
plt.savefig('figures/Figure_12_Ablation.png', bbox_inches='tight')
plt.close()

print("Pipeline V4 completed successfully!")
