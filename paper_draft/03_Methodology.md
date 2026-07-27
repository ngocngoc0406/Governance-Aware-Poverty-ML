# 3. Methodology

This section presents the computational framework proposed to evaluate and explain poverty targeting models. The methodology encompasses four core phases: data preprocessing and feature engineering, 5-model predictive architecture formulation, non-parametric statistical significance testing (McNemar's Test), and a 4-tier SHAP interpretability framework.

*Figure 1: End-to-End Research and ML Pipeline*

## 3.1. Data Preprocessing and Feature Engineering

To ensure robust model training on household socioeconomic census data, the raw dataset undergoes a systematic preprocessing pipeline:
1. **Handling Missing Values:** Missing categorical values (e.g., dwelling type, sanitation facilities) are imputed using the mode. Missing numerical features (e.g., rent payment, overcrowding ratio, years of education) are imputed using the median to prevent skewed estimation from extreme wealth outliers.
2. **Categorical Feature Encoding:** Mixed-type survey columns (such as `dependency`, `edjefe`, and `edjefa` containing string indicators like 'yes' and 'no') are cleaned and converted to continuous numerical floats. Remaining categorical features are transformed using Label Encoding.
3. **Scaling & Normalization:** Standard scaling ($\mu = 0, \sigma = 1$) is applied to continuous features specifically for distance-sensitive linear models (Logistic Regression). Tree-based models (RF, XGBoost, LightGBM, CatBoost) bypass scaling as decision tree splits are invariant to monotonic transformations.
4. **Target Binarization:** The target variable is binarized such that `1` denotes extreme/moderate poverty (requiring social assistance targeting) and `0` represents non-poor households.

To visualize multicollinearity and feature variances prior to modeling, a Correlation Matrix and Top Feature Distribution were generated (detailed in Figure 2).

*Figure 2: Feature Correlation Matrix and Distribution*

## 3.2. Predictive Modeling Architectures
We benchmark five distinct classification algorithms representing three algorithmic paradigms: linear econometrics (Logistic Regression), bagging ensembles (Random Forest), and gradient boosting ensembles (XGBoost, LightGBM, and CatBoost). 

Rather than detailing the standard mathematical foundations of these widely adopted algorithms, our methodological focus lies in their comparative behavior under highly regularized environments. Logistic Regression serves as the baseline proxy for traditional PMT formulas, rigidly assuming linear interactions. In contrast, the gradient boosting models are specifically deployed to capture non-linear, compounding socioeconomic vulnerabilities.

### 3.2.1. Hyperparameter Optimization and Experimental Protocol
To ensure optimal model performance and prevent overfitting, we strictly controlled model complexity through a Stratified 5-Fold Cross-Validation Grid Search. The dataset was split with an 80/20 ratio for training and held-out testing, preserving the natural class imbalance across all folds. 
Hyperparameter spaces for the tree-based ensembles were explicitly constrained (e.g., limiting tree depth and learning rate) to prioritize generalization on unseen data over maximizing training accuracy. Table I details the specific hyperparameter search space utilized.

*Table I: Hyperparameter Search Space for Tree Ensembles*

### 3.2.2. Experimental Environment and Reproducibility
To ensure full empirical reproducibility, all data preprocessing, model training, cross-validation, and runtime benchmarks were executed within a controlled environment:
- **Hardware:** Intel Core i7-12700H CPU @ 2.30 GHz (14 cores, 20 threads), 16 GB RAM.
- **Operating System & Software Environment:** Windows 11 64-bit, Python 3.10.12.
- **Core Scientific Libraries:** `scikit-learn` 1.3.0, `xgboost` 2.0.3, `lightgbm` 4.1.0, `catboost` 1.2.2, `shap` 0.42.1, `pandas` 2.0.3, `numpy` 1.24.3.
- **Random Seed Control:** All stochastic partitioning and model initialization calls were fixed with `random_state = 42`.

## 3.3. Statistical Significance Testing

To robustly determine whether performance improvements achieved by gradient boosting models over linear PMT are statistically significant, we rely on **McNemar's Non-Parametric Test** and the **Wilcoxon Signed-Rank Test**. 
Unlike standard performance metrics, McNemar's test explicitly evaluates the discordance in misclassification patterns between two models evaluated on the exact same test set. A computed $p$-value $< 0.001$ confirms that the predictive superiority is structural rather than an artifact of random sampling. The Wilcoxon signed-rank test is additionally applied across the cross-validation folds to verify the stability of these performance rankings under data perturbation.

## 3.4. Asymmetric Welfare-Weighted Optimization

To address the critical issue of exclusion errors (welfare denial), we adapt cost-sensitive learning principles into an asymmetric welfare-weighted objective. Standard cross-entropy penalizes False Positives and False Negatives equally. We operationalize a welfare-oriented penalty factor $\alpha > 1$ (implemented via XGBoost's `scale_pos_weight` parameter), where positive samples (the poverty class) are strictly upweighted during gradient boosting. Mathematically, weighting the positive class by $\alpha$ scales the first derivative (gradient $g_i$) and second derivative (hessian $h_i$) of the binary loss for a predicted probability $p_i = \sigma(\hat{y}_i)$ and true label $y_i \in \{0, 1\}$ as follows:

\begin{equation}
g_i = \begin{cases} 
\alpha (p_i - 1), & \text{if } y_i = 1 \\
p_i, & \text{if } y_i = 0 
\end{cases}
\end{equation}

\begin{equation}
h_i = \begin{cases} 
\alpha p_i (1 - p_i), & \text{if } y_i = 1 \\
p_i (1 - p_i), & \text{if } y_i = 0 
\end{cases}
\end{equation}

where $p_i = \sigma(\hat{y}_i)$. By setting $\alpha = 4.0$ (selected via sensitivity analysis mapping $\alpha \in \{2, 3, 4, 5\}$ to optimally balance precision and recall based on simulated fiscal constraints), the gradient boosting process heavily prioritizes learning the complex feature interactions of extreme poor households, mathematically minimizing exclusion errors directly within the optimization process.

## 3.5. Multi-Level SHAP Interpretability Framework

To provide full administrative governance, we apply SHapley Additive exPlanations (SHAP) across two analytical tiers:
1. **Tier 1: Global Summary Plot:** Measures overall feature importance rankings and directionality across the entire census sample.
2. **Tier 2: Individual Waterfall Plot:** Deconstructs single-household classification decisions into additive feature contributions ($\phi_i$), enabling caseworker grievance redressal.
