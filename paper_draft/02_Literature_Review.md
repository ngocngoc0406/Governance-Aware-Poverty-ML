# 2. Literature Review

The literature on poverty targeting, machine learning, and explainable AI can be categorized into three fundamental research streams: traditional proxy means testing (PMT) econometrics, modern machine learning predictive architectures, and explainable artificial intelligence (XAI) in public administration. This section reviews these streams and articulates the research gap.

## 2.1. Traditional Econometric Approaches to Poverty Targeting
For over three decades, Proxy Means Testing (PMT) has served as the core targeting mechanism for cash transfer programs across developing nations. Conventional PMT models employ linear regression architectures (OLS or Logistic Regression). However, landmark econometric evaluations demonstrate that linear PMT formulas exhibit severe targeting errors, frequently surpassing 40\% \cite{b4, b11}. These targeting failures stem from linear models' mathematical inability to capture non-linear, compounding socioeconomic vulnerabilities \cite{b2, b4}.

## 2.2. The Shift to Non-Linear Machine Learning
To overcome the rigid assumptions of linear PMT, recent literature explores non-parametric Machine Learning (ML) algorithms \cite{b12, b13}. Tree-based ensemble models—specifically Random Forest and Gradient Boosting architectures like XGBoost, LightGBM, and CatBoost—have set new performance benchmarks on structured survey data. Gradient boosting algorithms are exceptionally effective at discovering subtle non-linear interactions across high-dimensional feature spaces, consistently achieving superior F1-scores and ROC-AUC metrics compared to standard econometric models \cite{b7, b8}.

## 2.3. Operationalizing Explainable AI (XAI)
The primary barrier to deploying high-performing ML models in public policy is the "black-box" dilemma. Government institutions cannot legally deny citizens welfare benefits based on the output of an opaque algorithm. Conversely, Explainable AI (XAI), particularly SHAP (SHapley Additive exPlanations) \cite{b14}, has emerged as the gold standard for rendering complex models interpretable. Rooted in cooperative game theory, SHAP provides both global macroeconomic insights (identifying the most critical poverty drivers) and exact, micro-level explanations for individual algorithmic decisions. While previous studies have successfully applied SHAP to credit scoring \cite{b15} and urban poverty mapping \cite{b16}, its application as a direct grievance redressal tool in national welfare systems remains underexplored.

## 2.4. Articulating the Research Gap
Despite extensive literature exploring ML and XAI in poverty targeting, a critical gap remains in bridging the divide between predictive accuracy and administrative governance. 
First, existing poverty ML studies predominantly focus on optimizing classification metrics (e.g., AUC or F1-scores), treating explainability as a secondary feature rather than an operational requirement. 
Second, when XAI is employed, studies often rely on basic global feature rankings. Very few investigate whether local explainability (such as SHAP Waterfall plots) can satisfy the strict operational transparency required to process individual grievance redressal and justify welfare denials.

This study directly addresses this gap. Specifically, the novelty lies in integrating statistical validation and multi-level explainability into a decision-support framework for welfare targeting, demonstrating that statistically significant predictive gains can be achieved without sacrificing the micro-level transparency required for public sector accountability.
