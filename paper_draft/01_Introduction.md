# Abstract

Poverty targeting remains a critical challenge for governments and international organizations seeking to allocate limited resources effectively, particularly across emerging economies. Traditional targeting methods, such as Proxy Means Testing (PMT), often suffer from significant inclusion and exclusion errors due to rigid linear assumptions. While advanced machine learning (ML) models offer superior predictive accuracy, their "black-box" nature hinders adoption in public policy where transparency and accountability are legally mandated. This study presents a comprehensive comparative analysis of five classification algorithms—Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost—using a gold-standard household socioeconomic dataset from Costa Rica. We frame this dataset as an empirical benchmark to establish a robust, explainable framework prior to cross-regional deployment in large-scale social safety nets. To bridge the gap between predictive performance and administrative interpretability, we integrate SHapley Additive exPlanations (SHAP) across global, local, and interaction levels. Our experimental results demonstrate that gradient boosting models consistently outperform traditional baselines, with XGBoost achieving an Accuracy of 95.1\% and a Stratified 5-Fold Cross-Validation ROC-AUC of 0.974. McNemar's statistical test confirms that boosting models achieve statistically significant improvements over linear PMT ($p < 0.001$). Furthermore, SHAP interaction plots reveal complex non-linear tipping points between education and dependency ratios. The findings provide policy implications for decision-makers seeking to deploy transparent AI for welfare targeting.

# 1. Introduction

## 1.1. Problem Statement
The eradication of extreme poverty remains the foremost priority of the United Nations' Sustainable Development Goals (SDG 1) \cite{b1}. Central to this mission is the efficient, equitable allocation of social safety net resources—a challenge of acute relevance across emerging economies globally \cite{b2, b3}. Historically, public welfare agencies have relied on Proxy Means Testing (PMT) \cite{b4} to identify eligible households. PMT leverages easily observable household characteristics—such as housing construction materials, ownership of durable assets, and demographic composition—to proxy unobservable household income or consumption levels.

Despite widespread adoption by multilateral development institutions, conventional PMT suffers from severe inclusion errors (welfare leakage to non-eligible households) and exclusion errors (denial of essential assistance to vulnerable households). Econometric audits reveal that exclusion errors in standard PMT formulas frequently exceed 40%. These systematic targeting failures stem directly from PMT's reliance on linear econometric models (Ordinary Least Squares or Logistic Regression), which fail to model the complex, non-linear interactions inherent in household socioeconomic dynamics.

## 1.2. Research Gap
In recent years, machine learning (ML) architectures—including Random Forest, XGBoost, LightGBM, and CatBoost—have demonstrated strong predictive capabilities, significantly reducing targeting errors by learning non-linear feature spaces. However, their "black-box" nature hinders practical adoption in public administration.

In social protection administration, algorithmic transparency is an indispensable legal mandate \cite{b5, b6}. Existing studies primarily optimize predictive accuracy \cite{b7, b8}. While cost-sensitive optimization has been widely studied in machine learning, its specific application to welfare-oriented poverty targeting remains limited. Furthermore, very few investigate whether explainable AI can satisfy the strict operational transparency required in welfare targeting while maintaining top-tier predictive performance \cite{b9, b10}. Deploying uninterpretable algorithms risks eroding public trust and violating administrative justice.

While frameworks like SHAP (SHapley Additive exPlanations) have gained traction, the literature lacks a comprehensive methodology that bridges predictive benchmarking with actionable governance tools.

## 1.3. Research Questions
This study addresses three fundamental research questions:
1. **RQ1 (Predictive-Transparency Tradeoff):** Can advanced gradient boosting algorithms achieve statistically significant error reduction over linear econometric baselines without sacrificing operational transparency?
2. **RQ2 (Local Accountability):** How can local SHAP visualizations (Waterfall Plots) be operationalized to provide legally defensible, individual-level explanations for administrative grievance redressal?
3. **RQ3 (Global Policy Drivers):** What macroeconomic insights can be extracted from global SHAP values regarding the non-linear dynamics of household poverty?

## 1.4. Key Contributions
The contributions of this paper are threefold:
- **Policy-Oriented Cost-Sensitive Framework:** We propose a welfare-oriented adaptation of cost-sensitive learning for gradient boosting. By mathematically penalizing exclusion errors (False Negatives) during the training process via asymmetric class weighting, our framework aligns machine learning with humanitarian constraints, reducing exclusion errors by nearly half compared to standard optimization.
- **Governance-Oriented Framework for Transparent Targeting:** A comprehensive ML-XAI architecture that explicitly addresses the tradeoff between predictive performance and public sector accountability.
- **Micro-level Grievance Redressal Mechanisms:** We demonstrate how local explainability tools can be operationalized by caseworkers to generate transparent justifications for welfare eligibility decisions.
