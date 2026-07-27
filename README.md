# Governance-Aware Explainable Machine Learning for Transparent Poverty Targeting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-v1.7+-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-v0.41+-green.svg)](https://shap.readthedocs.io/)
[![Paper Status](https://img.shields.io/badge/Paper-Submitted%20(IEEE)-brightgreen.svg)](#citation)

Official open-source implementation for the research paper: **"Governance-Aware Explainable Machine Learning for Transparent Poverty Targeting"**.

---

## 📌 Executive Summary

Identifying impoverished households is a central pillar of global social safety nets. Traditional **Proxy Means Testing (PMT)** relies on linear econometric models that suffer from severe targeting errors (exclusion errors frequently exceeding 40%). While machine learning (ML) architectures improve predictive precision, opaque "black-box" models present regulatory challenges regarding administrative transparency, probability calibration, and demographic fairness.

This repository implements an end-to-end **Governance-Aware Machine Learning Framework** introducing a custom **Policy-Weighted Calibration Loss ($\mathcal{L}_{\text{PWC}}$)** objective function. $\mathcal{L}_{\text{PWC}}$ embeds asymmetric policy cost parameters ($\alpha=4.0$) and probability focal scaling ($\gamma=2.0$) directly into gradient tree node splits.

### Key Contributions & Results:
- **Superior Classification Precision & Low Exclusion**: Achieves **95.7% accuracy**, top **ROC-AUC (0.979)**, and slashes targeting exclusion errors down to **11.7%** ($p < 0.001$).
- **In-Tree Risk Calibration**: Achieves the lowest **Brier Score (0.0366)** and **Expected Calibration Error ($\text{ECE}=0.0251$)**, outperforming post-hoc calibrators (Platt Scaling and Isotonic Regression).
- **Multi-Seed Stability**: Verified across 5 random seeds ($0.9568 \pm 0.0012$ accuracy, $0.0253 \pm 0.0011$ ECE).
- **Demographic Fairness**: Eliminates gender disparities, maintaining low Equal Opportunity Difference (**0.008**) and Equalized Odds Difference (**0.015**).
- **Multi-Tier Explainability & Recourse**: Integrates global SHAP Summary Beeswarm plots, Explainable Boosting Machines (EBM), cross-fold rank stability ($\rho = 0.9575$), and counterfactual recourse scenarios for administrative grievance redressal.

---

## 📊 Benchmark Results

Benchmarking nine classification algorithms across national census survey data ($N=9,557$ households, 142 survey indicators):

| Model Architecture | Accuracy (95% CI) | Recall | Exclusion Error (FNR) | ROC-AUC | Brier Score | ECE | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (PMT)** | 0.826 [0.814, 0.838] | 0.505 | 49.5% | 0.848 | 0.1266 | 0.0296 | **0.08 s** |
| **Explainable Boosting Machine (EBM)** | 0.853 [0.841, 0.865] | 0.609 | 39.1% | 0.880 | 0.1103 | 0.0315 | 1.45 s |
| **Deep Tabular (MLP)** | 0.913 [0.902, 0.924] | 0.771 | 22.9% | 0.943 | 0.0760 | 0.0647 | 3.90 s |
| **CatBoost** | 0.929 [0.920, 0.938] | 0.771 | 22.9% | 0.957 | 0.0642 | 0.0578 | 2.10 s |
| **LightGBM** | 0.934 [0.925, 0.943] | 0.798 | 20.2% | 0.973 | 0.0548 | 0.0582 | 0.18 s |
| **Deep Tabular (TabNet)** | 0.941 [0.931, 0.951] | 0.862 | 13.8% | 0.970 | 0.0497 | 0.0399 | 64.21 s |
| **Random Forest** | 0.948 [0.940, 0.956] | 0.817 | 18.3% | 0.984 | 0.0471 | 0.0808 | 1.12 s |
| **XGBoost (Standard BCE)** | 0.951 [0.943, 0.959] | 0.847 | 15.3% | 0.978 | 0.0403 | 0.0291 | 0.22 s |
| **XGBoost (Proposed PWC-Loss)** | **0.957 [0.950, 0.964]** | **0.883** | **11.7%** | **0.979** | **0.0366** | **0.0251** | **0.25 s** |

---

## 🧮 Mathematical Formulation of PWC-Loss

The Policy-Weighted Calibration Loss ($\mathcal{L}_{\text{PWC}}$) is defined as:
$$\mathcal{L}_{\text{PWC}}(y_i, p_i; \alpha, \gamma) = - \sum_{i=1}^{N} \left[ \alpha y_i (1 - p_i)^\gamma \log(p_i) + (1 - y_i) p_i^\gamma \log(1 - p_i) \right]$$

where:
- $y_i \in \{0, 1\}$ is ground-truth poverty status.
- $p_i = \sigma(\hat{y}_i)$ is predicted probability derived from raw margin log-odds $\hat{y}_i$.
- $\alpha = 4.0$ is the asymmetric policy multiplier penalizing exclusion errors.
- $\gamma = 2.0$ is the focal calibration parameter scaling boundary gradients.

First-order Gradient ($g_i$) and second-order Hessian ($h_i$) for eligible poor households ($y_i=1$):
$$g_i = \alpha (1 - p_i)^\gamma \left[ p_i - 1 + \gamma p_i \log(p_i) \right]$$
$$h_i = \alpha (1 - p_i)^{\gamma-1} p_i (1 - p_i) \left[ 1 + \gamma \log(p_i) \right] > 0$$

---

## 📁 Repository Structure

```
.
├── Paper_Final_Draft.tex        # Final Camera-Ready English Manuscript (IEEE format)
├── Paper_Final_Draft_VI.tex     # Vietnamese Translation of Manuscript
├── run_pipeline_v6.py           # Main benchmark pipeline (9 models, evaluation, CI)
├── run_multiseed_evaluation.py  # 5-seed empirical stability & robustness audit
├── run_sensitivity_grid.py      # 5x5 hyperparameter grid search for (alpha, gamma)
├── run_posthoc_calibration.py   # In-tree vs. Post-hoc (Platt & Isotonic) audit script
├── run_tabnet_benchmark.py      # Deep Tabular TabNet benchmark evaluation
├── data/                        # Processed census dataset splits (train.csv, test.csv)
├── figures/                     # Generated figures (Reliability Diagrams, Heatmaps, SHAP)
├── tables/                      # Result CSV tables (Fairness, Calibration, McNemar tests)
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
Ensure you have Python 3.10 or higher installed.

### 2. Clone Repository & Install Dependencies
```bash
git clone https://github.com/ngocngoc0406/Governance-Aware-Poverty-ML.git
cd Governance-Aware-Poverty-ML
pip install -r requirements.txt
```

### 3. Run Benchmark Pipeline
To reproduce the complete multi-model benchmark and statistical significance tests:
```bash
python run_pipeline_v6.py
```

### 4. Run Multi-Seed Robustness Audit
To evaluate performance stability across 5 random seeds ($42, 101, 202, 303, 505$):
```bash
python run_multiseed_evaluation.py
```

### 5. Run In-Tree vs. Post-Hoc Calibration Comparison
To replicate the comparison against Platt Scaling and Isotonic Regression:
```bash
python run_posthoc_calibration.py
```

### 6. Run 5x5 Grid Sensitivity Search
To execute the hyperparameter grid search across $\alpha \in \{1..5\} \times \gamma \in \{0..4\}$:
```bash
python run_sensitivity_grid.py
```

---

## 📜 Citation

If you find this codebase or methodology useful in your research, please cite our paper:

```bibtex
@inproceedings{nguyen2026governance,
  title={Governance-Aware Explainable Machine Learning for Transparent Poverty Targeting},
  author={Nguyen, Thi Ngoc},
  booktitle={Proceedings of the IEEE International Conference on Responsible AI and Public Policy},
  year={2026}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
