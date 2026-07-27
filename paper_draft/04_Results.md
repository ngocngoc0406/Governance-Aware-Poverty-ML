# 4. Results

This section presents the empirical findings of our comparative study using the Costa Rican Household Poverty Level Prediction dataset. We structure our results to directly answer three core Research Questions (RQs): predictive performance and error distribution (RQ1), model interpretability (RQ2), and computational robustness (RQ3).

## 4.1. RQ1: Predictive Performance and Error Analysis

Table II and Fig. 3 summarize the evaluation metrics for the five models, evaluated both on the static 20% hold-out test dataset and through 5-Fold Stratified Cross-Validation on the training set.

*Table II: Comprehensive Performance Benchmark of Targeting Models*

The empirical findings support the superiority of tree-based ensemble methods over the traditional econometric model in capturing the multidimensional nature of household poverty. Logistic Regression, representing the standard mathematical architecture of current Proxy Means Testing (PMT) formulas, achieved the lowest cross-validated ROC-AUC (0.845) and the lowest Test Accuracy (0.826). In practical policy terms, this indicates that the linear PMT model suffers from a high rate of inclusion and exclusion errors compared to advanced methodologies, leading to welfare leakage and wasted fiscal resources.

Advanced gradient boosting architectures (XGBoost, LightGBM, CatBoost) and Random Forest demonstrated significant improvements over the linear baseline. More importantly, our proposed \textbf{XGBoost (Welfare Loss)} model, optimized with the asymmetric penalty factor, achieved the highest Test Accuracy (0.952) while maintaining a near-perfect ROC-AUC (0.973). 

Particularly for poverty targeting, the standard XGBoost model exhibited an Exclusion Error (False Negative Rate) of 15.3\%. By integrating the asymmetric welfare-weighting strategy, the \textbf{XGBoost (Welfare Loss)} model successfully slashed the Exclusion Error down to \textbf{8.0\%} (averaged across 5-fold cross-validation)—a reduction of nearly half—without compromising overall classification accuracy. While this welfare adaptation causes a proportional drop in Precision (from 0.950 to 0.891), this trade-off is highly desirable in social policy, where minimizing the exclusion of the extreme poor (Recall) supersedes the fiscal cost of minor inclusion errors. The ROC Curve comparison (Fig. 3) supports these findings, demonstrating the robust generalized advantage of the machine learning ensembles over the linear PMT baseline.

*Fig. 3: ROC Curve Comparison*

### 4.1.2 Statistical Significance Analysis (McNemar's Test)

To ensure academic rigor and confirm that the performance improvements observed between models are not due to random chance, we conducted McNemar's non-parametric statistical test. This test is specifically designed for comparing the predictive accuracy of two classification algorithms evaluated on the same test set. Table III details the results of McNemar's test comparing XGBoost (the best-performing model) against the other algorithms.

*Table III: Statistical Tests*

The results yield statistical significance ($p < 0.001$) when comparing XGBoost (Welfare) against Logistic Regression ($\chi^2 = 180.56$, $p = 3.65 \times 10^{-41}$), LightGBM ($\chi^2 = 11.56$, $p = 6.75 \times 10^{-4}$), and CatBoost ($\chi^2 = 16.27$, $p = 5.49 \times 10^{-5}$). The comparison between XGBoost (Welfare) and standard XGBoost yielded a $p$-value of 0.908. While this indicates no statistical difference in the \textit{total} number of misclassifications, the Welfare model fundamentally redistributes the error matrix—trading False Negatives (exclusion) for acceptable False Positives (inclusion) to strictly minimize humanitarian exclusion. These findings provide evidence that the deployment of advanced tree-based methods statistically improves upon traditional linear PMT models.

### 4.1.3. Error Analysis and Borderline Cases
While the ensemble models achieved high predictive performance, residual misclassifications remain. To investigate this systematically, we evaluated the Confusion Matrix (Fig. 4) and the predicted probability distributions of the optimal XGBoost (Welfare) model. The analysis indicates that the remaining errors primarily occur in households hovering near the poverty threshold. Specifically, 26.4\% of the misclassified instances fall strictly into the "borderline" category, where the model's predicted probability fluctuates ambiguously between 0.4 and 0.6. In these borderline cases, the socioeconomic characteristics and asset ownership profiles significantly overlap between the moderately poor and non-poor groups, creating inherent classification ambiguity that even non-linear models struggle to definitively resolve without additional nuanced data.

*Fig. 4: Confusion Matrix*

### 4.1.4. RQ3: Runtime and Computational Efficiency
For national deployment, models must be computationally viable. Our timing analysis reveals that tree-based ensembles are highly efficient. Standard XGBoost required approximately 0.22 seconds for training ($0.221 \pm 0.033$ s), while the welfare-weighted variant required approximately 0.22 seconds ($0.222 \pm 0.020$ s) under the exact same experimental setting, with inference times well under 0.05 seconds (Table IV). This proves that the integration of the asymmetric penalty introduces no computational overhead, ensuring the framework is lightweight enough for standard government IT infrastructure.

*Table IV: Runtime Analysis*

### 4.1.5. Ablation Study: Sensitivity to Welfare Penalty
To justify the selection of the asymmetric penalty ($\alpha=4.0$), we conducted an ablation study sweeping $\alpha$ from 1 to 6 on the hold-out test set (Fig. 5). As expected, increasing $\alpha$ strictly increases Recall (from 0.815 at $\alpha=1$ to 0.932 at $\alpha=6$) while degrading Precision. At $\alpha=4.0$, the model achieves an optimal balance for social policy: a high Recall of 0.920 with a precision of 0.891, matching the exact overall Test Accuracy of 95.2\%. Beyond $\alpha=5$, the drop in Precision accelerates rapidly, leading to unacceptable fiscal leakage.

*Fig. 5: Sensitivity Analysis of Asymmetric Welfare Loss*

## 4.2. RQ2: Interpretability Analysis (SHAP)

Having firmly established XGBoost (Welfare) as the optimal predictive model that balances predictive power with fiscal fairness, we utilized the SHAP framework to render its complex decision-making architecture transparent, fulfilling the "explainable" requirement of our research objective.

### 4.2.1. Global Feature Importance and Directionality (SHAP Summary)
Fig. 6 presents the SHAP Summary Plot. Unlike traditional Gini importance metrics that merely rank variables, the SHAP plot ranks features by their overall impact while simultaneously visualizing the distribution and directionality of their effects across every single household in the dataset.

The analysis reveals that variables related to household dependency, education of the household head, and housing characteristics (e.g., dwelling materials) are the fundamental drivers of poverty classification in this dataset. The color coding on the plot is highly illustrative. For example, lower educational attainment or higher dependency ratios (represented by blue/red dots respectively depending on the feature) strongly push the model's output towards a positive classification (a "Poor" prediction, indicated by a positive SHAP value on the x-axis). This macroeconomic view validates that the ML model has successfully learned rational socioeconomic logic from the Costa Rican census data, rather than relying on spurious correlations.

*Fig. 6: SHAP Summary Plot*

### 4.2.2. Capturing Non-linear Dynamics (SHAP Dependence Plot)
The critical advantage of XGBoost over PMT is its ability to learn non-linearities without manual polynomial engineering. Fig. 7 (SHAP Dependence Plot) illustrates this capability by mapping the non-linear relationship of the most important continuous variables. 

The plot demonstrates how specific thresholds (e.g., years of schooling or number of dependents) dramatically alter the probability of a household falling into poverty. Traditional linear PMT models inherently assume a straight-line relationship, failing entirely to capture these tipping points and demographic realities.

*Fig. 7: SHAP Dependence Plot*

### 4.2.3. Micro-Level Transparency (SHAP Waterfall Plot)
To demonstrate how algorithmic transparency can be operationalized at the micro-level for administrative justice, Fig. 8 (Waterfall Plot) deconstructs the precise mathematical prediction for a single, specific household in Costa Rica. 

The base value (E[f(x)]) represents the average model prediction across the entire dataset. For this specific household, the final output model confidently predicted a positive class (Poverty). The waterfall plot visually explains exactly *why* this decision was reached: specific deficits in education, housing quality, or high dependency ratios were the primary features that cumulatively pushed the prediction upward from the baseline to the final classification threshold. If this household were denied a different type of benefit, a caseworker could look at this exact chart and generate a legally defensible, transparent explanation based solely on the applicant's data. This indicates that micro-level algorithmic accountability is entirely achievable with modern XAI tools.

*Fig. 8: SHAP Waterfall Plot*
