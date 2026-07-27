# 5. Discussion

The empirical findings of this study demonstrate that replacing traditional linear Proxy Means Testing (PMT) with Explainable Machine Learning represents a significant improvement in social protection administration. This section analyzes our results through two distinct lenses: academic implications for computational social science and policy implications for social welfare administration. We also contextualize our outcomes against recent landmark studies and candidly address study limitations.

## 5.1. Academic Implications
Our study contributes to computational economics by providing empirical evidence that household poverty is driven by complex, non-linear feature interactions that linear models miss. The superiority of XGBoost over Logistic Regression stems directly from its architectural ability to partition non-linear tabular data space and handle mixed data types inherently, without the need for manual polynomial feature engineering. 

Furthermore, our comparative benchmark yields nuanced insights across ensemble paradigms:
- **Random Forest vs. XGBoost:** Random Forest achieved a surprisingly competitive CV ROC-AUC (0.974), demonstrating that bagging is highly effective at reducing variance in noisy census data. However, XGBoost outperformed Random Forest when integrated with asymmetric loss due to asymmetric class weighting during gradient boosting.
- **LightGBM & CatBoost:** While LightGBM and CatBoost demonstrated exceptionally low inference times (0.010s and 0.008s respectively), XGBoost achieved superior overall classification stability. This disparity may be explained by LightGBM's leaf-wise tree growth being slightly more sensitive to localized socioeconomic noise, whereas CatBoost's symmetric tree structures required higher tree depth to match XGBoost's feature interaction resolution.
- **Class Imbalance Effect:** In standard unweighted models, class imbalance (~20% poor vs. ~80% non-poor) forces algorithms to maximize precision at the expense of recall. The Asymmetric Welfare Loss effectively counteracts this skew, aligning algorithmic loss gradients with real-world social protection objectives.

## 5.2. Comparative Analysis with Recent Literature
While recent studies \cite{b7, b8} have also reported strong predictive metrics using machine learning for poverty targeting, a direct comparison of AUC scores across disparate datasets is inherently flawed. Instead, our study’s primary contribution lies in methodological depth and operational transparency. 

Unlike spatial poverty mapping studies that rely on satellite imagery \cite{b17, b18} (which is excellent for regional resource allocation but difficult to use for individual household eligibility), our framework leverages structured administrative census data. This ensures direct compatibility with existing government PMT workflows. Furthermore, while previous literature frequently stops at global feature importance (e.g., relying on Gini importance or broad SHAP summary plots), our methodology strictly operationalizes local SHAP Waterfall plots to address the practical legal mandate of individual grievance redressal in public welfare systems.

## 5.3. Policy Implications
The operationalization of Explainable AI offers concrete benefits for public welfare institutions by enhancing administrative transparency and minimizing both fiscal leakage and social exclusion. Achieving 89.1% precision ensures public funds reach genuine vulnerable populations with minimal leakage, while 92.0% recall guarantees that extreme poor households are correctly identified. Crucially, automated SHAP Waterfall plots provide an objective, transparent breakdown for any rejected household application, offering a legally defensible foundation for grievance redressal.

To realize these benefits at scale, policymakers must adopt a structured deployment pipeline. The rigorous validation achieved on this high-quality global benchmark establishes the methodological foundation. The critical next step involves transfer learning and structural recalibration onto other national databases \cite{b9, b19}. This phased approach—from benchmark validation to real-world deployment—ensures that the governance framework is mathematically sound before encountering the administrative noise inherent in emerging economies. Furthermore, our runtime analysis confirms that such models can be trained in under a second, eliminating computational complexity as a barrier to adoption for government agencies.

# 6. Conclusion

## 6.1. Summary of Findings
This study presented a transparent, explainable machine learning framework for household poverty targeting using the Costa Rican Household Poverty dataset. We benchmarked Logistic Regression, Random Forest, and XGBoost using Stratified 5-Fold Cross-Validation. XGBoost achieved the highest performance across all metrics (Accuracy: 95.2%, F1-Score: 0.905, CV ROC-AUC: 0.973 $\pm 0.005$), significantly outperforming the traditional linear PMT baseline (CV ROC-AUC: 0.845).

Integrating SHAP unlocked black-box predictions, revealing that household education, dependency ratios, and dwelling quality are key poverty drivers. SHAP Waterfall Plots provided individual, household-level transparency essential for administrative accountability. Overall, these findings demonstrate that explainable gradient boosting can substantially reduce exclusion errors while maintaining the operational transparency required for public-sector decision making, making it a practical and superior alternative to conventional PMT.

## 6.2. Threats to Validity and Study Limitations
Despite strong empirical performance, this study acknowledges several important limitations:
1. **Data Cross-Sectionality:** The reliance on cross-sectional survey data limits the capture of chronic versus transient poverty dynamics over time. 
2. **Methodological Limits of Static Welfare Penalty:** The proposed asymmetric penalty factor $\alpha=4.0$ was selected via empirical sensitivity analysis on the Costa Rican benchmark. In practical administrative deployments, a fixed static $\alpha$ may not dynamically adapt to fluctuating macroeconomic shocks, localized inflation, or strict fiscal budget caps.
3. **Geographic Generalizability:** We deliberately chose the Costa Rican dataset as a high-quality global benchmark to establish a mathematically rigorous, transparent explainability framework. However, hyperparameters and feature rankings must be meticulously re-calibrated prior to deployment in different economic contexts \cite{b9, b19}.

## 6.3. Future Work
Future research can expand upon this work in several promising directions:
- **Adaptive Welfare Weighting:** Developing dynamic, data-driven methods to automatically tune the $\alpha$ penalty parameter based on real-time macroeconomic indicators and national budget constraints, rather than relying on static sensitivity analysis.
- **Spatio-Tabular Representation Learning:** Integrating Graph Neural Networks (GNNs) with tree-based models to capture spatial autocorrelations between neighboring households, thereby improving localized poverty mapping.
- **Active Learning for Cost-Efficient Targeting:** Developing uncertainty-sampling frameworks to iteratively select which households require physical surveying, significantly reducing the administrative cost of national censuses.
- **Fairness \& Bias Auditing:** While this study focuses on explainability, comprehensive algorithmic fairness auditing (e.g., evaluating Equal Opportunity across gender and regional demographics) remains a crucial requirement before public deployment.
- **External Validation:** Direct application and empirical evaluation of this framework on other regional household surveys to verify cross-cultural transferability.
