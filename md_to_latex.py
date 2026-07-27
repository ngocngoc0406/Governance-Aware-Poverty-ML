import re

def convert_md_to_latex(md_text):
    # Abstract specific
    md_text = re.sub(r'# Abstract(.*?)# 1\. Introduction', r'\\begin{abstract}\1\\end{abstract}\n\n\\begin{IEEEkeywords}\nPoverty targeting, Proxy Means Testing (PMT), Explainable AI (XAI), SHAP, Machine Learning, XGBoost, Social Safety Nets.\n\\end{IEEEkeywords}\n\n\\section{Introduction}', md_text, flags=re.DOTALL)
    
    # Sections
    md_text = re.sub(r'^# \d*\.?\s*(.*?)$', r'\\section{\1}', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'^## \d*\.\d*\.?\s*(.*?)$', r'\\subsection{\1}', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'^### \d*\.\d*\.\d*\.?\s*(.*?)$', r'\\subsubsection{\1}', md_text, flags=re.MULTILINE)
    
    # Replace Table 5 area
    table5_latex = r'''
\begin{table*}[htbp]
\caption{Comprehensive Performance Comparison of Targeting Models}
\begin{center}
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Exclusion Err (FNR)} & \textbf{CV ROC-AUC} \\
\hline
Logistic Regression & 0.826 & 0.706 & 49.4\% & 0.846 $\pm 0.003$ \\
\hline
Random Forest & 0.948 & 0.965 & 18.2\% & 0.974 $\pm 0.004$ \\
\hline
XGBoost (Standard) & 0.951 & 0.950 & 15.3\% & \textbf{0.974} $\mathbf{\pm 0.004}$ \\
\hline
\textbf{XGBoost (Welfare)} & \textbf{0.952} & \textbf{0.891} & \textbf{8.0\%} & 0.973 $\pm 0.005$ \\
\hline
LightGBM & 0.934 & 0.922 & 20.1\% & 0.965 $\pm 0.003$ \\
\hline
CatBoost & 0.929 & 0.928 & 22.9\% & 0.945 $\pm 0.002$ \\
\hline
\end{tabular}
\label{tab_performance}
\end{center}
\end{table*}
'''
    md_text = re.sub(r'\*Table II:.*?\n(?=\n|#|\\[a-z]+)', lambda m: table5_latex, md_text, flags=re.DOTALL)

    # Replace Table 3 area (Hyperparameters)
    table3_latex = r'''\begin{table*}[htbp]
\caption{Hyperparameter Search Space and Optimal Configurations}
\begin{center}
\begin{tabular}{|l|l|l|}
\hline
\textbf{Model} & \textbf{Hyperparameter Grid} & \textbf{Optimal Hyperparameters} \\
\hline
Logistic Regression & \texttt{C}: [0.01, 0.1, 1.0, 10.0], \texttt{penalty}: ['l2'] & \texttt{C}=1.0, \texttt{penalty}='l2', \texttt{solver}='lbfgs' \\
\hline
Random Forest & \texttt{n\_estimators}: [100, 200], \texttt{max\_depth}: [None, 10, 20] & \texttt{n\_estimators}=100, \texttt{max\_depth}=None \\
\hline
XGBoost (Standard) & \texttt{max\_depth}: [3, 6, 9], \texttt{learning\_rate}: [0.01, 0.1, 0.3] & \texttt{max\_depth}=6, \texttt{learning\_rate}=0.1, \texttt{n\_estimators}=100 \\
\hline
\textbf{XGBoost (Welfare)} & \texttt{scale\_pos\_weight}: [1, 2, 4, 6], \texttt{learning\_rate}: [0.1] & \texttt{max\_depth}=6, \texttt{learning\_rate}=0.1, \texttt{scale\_pos\_weight}=4.0 \\
\hline
LightGBM & \texttt{num\_leaves}: [31, 50], \texttt{learning\_rate}: [0.01, 0.1] & \texttt{num\_leaves}=31, \texttt{learning\_rate}=0.1, \texttt{n\_estimators}=100 \\
\hline
CatBoost & \texttt{depth}: [4, 6, 8], \texttt{learning\_rate}: [0.01, 0.1] & \texttt{depth}=6, \texttt{learning\_rate}=0.1, \texttt{iterations}=100 \\
\hline
\end{tabular}
\label{tab_hyperparams}
\end{center}
\end{table*}'''
    md_text = re.sub(r'\*Table I:.*?\n(?=\n|#|\\[a-z]+)', lambda m: table3_latex, md_text, flags=re.DOTALL)

    # Replace Table 7 area (McNemar's test)
    table7_latex = r'''\begin{table*}[htbp]
\caption{McNemar's Non-Parametric Statistical Test (Baseline: XGBoost Welfare)}
\begin{center}
\begin{tabular}{|l|l|c|c|c|}
\hline
\textbf{Baseline Model} & \textbf{Comparison Model} & \textbf{$\chi^2$ Statistic} & \textbf{$p$-value} & \textbf{Significant?} \\
\hline
XGBoost (Welfare) & Logistic Regression & 180.56 & $3.65 \times 10^{-41}$ & Yes \\
\hline
XGBoost (Welfare) & Random Forest & 0.65 & 0.421 & No \\
\hline
XGBoost (Welfare) & XGBoost (Std) & 0.01 & 0.908 & No \\
\hline
XGBoost (Welfare) & LightGBM & 11.56 & $6.75 \times 10^{-4}$ & Yes \\
\hline
XGBoost (Welfare) & CatBoost & 16.27 & $5.49 \times 10^{-5}$ & Yes \\
\hline
\end{tabular}
\label{tab_mcnemar}
\end{center}
\end{table*}'''
    md_text = re.sub(r'\*Table III:.*?\n(?=\n|#|\\[a-z]+)', lambda m: table7_latex, md_text, flags=re.DOTALL)

    # Replace Table 9 area (Runtime Analysis)
    table9_latex = r'''\begin{table}[htbp]
\caption{Runtime Analysis of Evaluated Models (Mean $\pm$ Std across 5 runs)}
\begin{center}
\begin{tabular}{|l|c|c|}
\hline
\textbf{Model} & \textbf{Training Time (s)} & \textbf{Inference Time (s)} \\
\hline
Logistic Regression & $0.100 \pm 0.005$ & $0.003 \pm 0.000$ \\
Random Forest & $0.262 \pm 0.022$ & $0.088 \pm 0.018$ \\
XGBoost (Standard) & $0.221 \pm 0.033$ & $0.032 \pm 0.002$ \\
\textbf{XGBoost (Welfare)} & $\mathbf{0.222 \pm 0.020}$ & $\mathbf{0.032 \pm 0.001}$ \\
LightGBM & $0.458 \pm 0.676$ & $0.011 \pm 0.001$ \\
CatBoost & $0.540 \pm 0.129$ & $0.009 \pm 0.004$ \\
\hline
\end{tabular}
\label{tab_runtime}
\end{center}
\end{table}'''
    md_text = re.sub(r'\*Table IV:.*?\n(?=\n|#|\\[a-z]+)', lambda m: table9_latex, md_text, flags=re.DOTALL)
    md_text = re.sub(r'\*Table IX:.*?\n(?=\n|#|\\[a-z]+)', lambda m: table9_latex, md_text, flags=re.DOTALL)

    # Figures
    def make_figure(filename, caption, label):
        return f'''\\begin{{figure}}[htbp]
\\centerline{{\\includegraphics[width=\\columnwidth]{{figures/{filename}}}}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure}}'''

    md_text = re.sub(r'\*Figure 1:.*?\n(?=\n|#|\\[a-z]+)', lambda m: make_figure('Figure_1_Research_Pipeline.png', 'End-to-End Research and ML Pipeline', 'fig_pipeline'), md_text, flags=re.DOTALL)
    md_text = re.sub(r'\*Figure 2:.*?\n(?=\n|#|\\[a-z]+)', lambda m: make_figure('Figure_3_Correlation_Matrix.png', 'Feature Correlation Matrix', 'fig_feat_dist'), md_text, flags=re.DOTALL)
    md_text = re.sub(r'\*Fig\. 3:.*?\n(?=\n|#|\\[a-z]+)', lambda m: make_figure('Figure_4_ROC_Curve.png', 'ROC Curve Comparison', 'fig_roc'), md_text, flags=re.DOTALL)
    md_text = re.sub(r'\*Fig\. 4:.*?\n(?=\n|#|\\[a-z]+)', lambda m: make_figure('Figure_13_Confusion_Matrix.png', 'Confusion Matrix of XGBoost (Welfare)', 'fig_cm'), md_text, flags=re.DOTALL)
    md_text = re.sub(r'\*Fig\. 5:.*?\n(?=\n|#|\\[a-z]+)', lambda m: make_figure('Figure_12_Ablation.png', 'Sensitivity Analysis of Asymmetric Welfare Loss', 'fig_ablation'), md_text, flags=re.DOTALL)
    md_text = re.sub(r'\*Fig\. 6:.*?\n(?=\n|#|\\[a-z]+)', lambda m: make_figure('Figure_5_SHAP_Summary.png', 'SHAP Summary Plot (Global Feature Importance)', 'fig_shap_summary'), md_text, flags=re.DOTALL)
    md_text = re.sub(r'\*Fig\. 7:.*?\n(?=\n|#|\\[a-z]+)', lambda m: make_figure('Figure_8_Dependence_Plot.png', 'SHAP Dependence Plot', 'fig_shap_dep'), md_text, flags=re.DOTALL)
    md_text = re.sub(r'\*Fig\. 8:.*?\n(?=\n|#|\\[a-z]+)', lambda m: make_figure('Figure_7_Waterfall_Plot.png', 'SHAP Waterfall Plot (Local Interpretability)', 'fig_shap_waterfall'), md_text, flags=re.DOTALL)



    # Bold
    md_text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', md_text)
    # Italic
    md_text = re.sub(r'\*(.*?)\*', r'\\textit{\1}', md_text)

    # Inline Code
    def code_replacer(match):
        code_text = match.group(1)
        code_text = code_text.replace('_', r'\_')
        return r'\texttt{' + code_text + '}'
    
    md_text = re.sub(r'`([^`]+)`', code_replacer, md_text)

    # Specific math cleanupations
    md_text = re.sub(r'\$(\\mu = 0, \\sigma = 1)\$', r'$\1$', md_text)

    # Clean leftover raw pipe tables
    lines = md_text.split('\n')
    cleaned_lines = [l for l in lines if not (l.strip().startswith('|') and l.strip().endswith('|'))]
    md_text = '\n'.join(cleaned_lines)
        
    # Lists
    md_text = re.sub(r'^(\d+)\. (.*?)$', r'\\begin{enumerate}\n\\item \2\n\\end{enumerate}', md_text, flags=re.MULTILINE)
    md_text = md_text.replace('\\end{enumerate}\n\\begin{enumerate}\n', '')
    
    md_text = re.sub(r'^- (.*?)$', r'\\begin{itemize}\n\\item \1\n\\end{itemize}', md_text, flags=re.MULTILINE)
    md_text = md_text.replace('\\end{itemize}\n\\begin{itemize}\n', '')
    
    # Escape percent signs in text
    md_text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1\\%', md_text)

    return md_text

def insert_figures(latex_text):
    return latex_text

def main():
    md_files = [
        'paper_draft/01_Introduction.md',
        'paper_draft/02_Literature_Review.md',
        'paper_draft/03_Methodology.md',
        'paper_draft/04_Results.md',
        'paper_draft/05_Discussion_Conclusion.md'
    ]
    
    full_md = ""
    for f in md_files:
        with open(f, 'r', encoding='utf-8') as file:
            full_md += file.read() + "\n\n"
            
    body_latex = convert_md_to_latex(full_md)
    body_latex = insert_figures(body_latex)
    
    with open('paper_draft/IEEE-conference-template-062824.tex', 'r', encoding='utf-8') as f:
        template = f.read()
        
    header = r'''\title{Explainable Machine Learning for Transparent Poverty Targeting: A Comparative Study}

\author{\IEEEauthorblockN{Nguyen Thi Ngoc}
\IEEEauthorblockA{\textit{Vietnam-Korea University of Information and Communication Technology} \\
\textit{The University of Danang}\\
Da Nang, Vietnam}
}

\maketitle

'''
    
    final_latex = template.split(r'\begin{document}')[0] + r'\usepackage{placeins}' + '\n\\begin{document}\n' + header + body_latex + r'''

\FloatBarrier
\begin{thebibliography}{00}
\bibitem{b1} Sachs, J. D. (2012). From millennium development goals to sustainable development goals. \textit{The Lancet}, 379(9832), 2206-2211.
\bibitem{b2} Brown, C., Ravallion, M., \& van de Walle, D. (2018). A poor means test? Econometric targeting in Africa. \textit{Journal of Development Economics}, 134, 109-124.
\bibitem{b3} Hanna, R., \& Olken, B. A. (2018). Universal basic incomes versus targeted transfers: Anti-poverty programs in developing countries. \textit{Journal of Economic Perspectives}, 32(4), 201-226.
\bibitem{b4} Coady, D., Grosh, M., \& Hoddinott, J. (2004). Targeting of transfers in developing countries. \textit{World Bank Publications}.
\bibitem{b5} Goodman, B., \& Flaxman, S. (2017). European Union regulations on algorithmic decision-making and a ``right to explanation''. \textit{AI Magazine}, 38(3), 50-57.
\bibitem{b6} Lepri, B., Oliver, N., Letouzé, E., Pentland, A., \& Vinck, P. (2018). Fair, transparent, and accountable algorithmic decision-making processes. \textit{Philosophy \& Technology}, 31(4), 611-627.
\bibitem{b7} Aiken, E., Bellue, S., Karlan, D., Udry, C., \& Blumenstock, J. E. (2022). Machine learning and phone data can improve targeting of humanitarian aid. \textit{Nature}, 603(7903), 864-870.
\bibitem{b8} Blumenstock, J. E. (2016). Fighting poverty with data. \textit{Science}, 353(6301), 753-754.
\bibitem{b9} Kshirsagar, V., Waldinger, D., \& Levin, J. (2021). Machine learning to support public policies: A case study on poverty prediction in Latin America. \textit{Data \& Policy}, 3, e11.
\bibitem{b10} Norambuena, M., \& Torres, F. (2020). Predictive power of machine learning algorithms for poverty classification: Evidence from Costa Rica. \textit{Journal of Applied Economics}.
\bibitem{b11} Kidd, S., Gelders, B., \& Bailey-Athias, D. (2017). Exclusion by design: An assessment of the effectiveness of the proxy means test poverty targeting mechanism. \textit{International Labour Organization Working Paper}.
\bibitem{b12} Jean, N., Burke, M., Xie, M., Davis, W. M., Lobell, D. B., \& Ermon, S. (2016). Combining satellite imagery and machine learning to predict poverty. \textit{Science}, 353(6301), 790-794.
\bibitem{b13} Blumenstock, J., Cadamuro, G., \& On, R. (2015). Predicting poverty and wealth from mobile phone metadata. \textit{Science}, 350(6264), 1073-1076.
\bibitem{b14} Lundberg, S. M., \& Lee, S. I. (2017). A unified approach to interpreting model predictions. \textit{Advances in Neural Information Processing Systems}, 30.
\bibitem{b15} Bussmann, N., Giudici, P., Marinelli, D., \& Papenbrock, J. (2021). Explainable machine learning in credit risk management. \textit{Computational Economics}, 57(1), 203-216.
\bibitem{b16} Yeh, C., Perez, A., Driscoll, A., Azzari, G., Tang, Z., Lobell, D., ... \& Ermon, S. (2020). Using publicly available satellite imagery and deep learning to understand economic well-being in Africa. \textit{Nature Communications}, 11(1), 1-11.
\bibitem{b17} Babenko, B., Hersh, J., Newhouse, D., Ramakrishnan, A., \& Swartz, T. (2017). Poverty mapping using convolutional neural networks trained on high and medium resolution satellite images. \textit{arXiv preprint arXiv:1711.06848}.
\bibitem{b18} Head, A., Manguin, M., Tran, N., \& Blumenstock, J. E. (2017). Can human development be measured with satellite imagery?. \textit{ICTD}.
\bibitem{b19} McBride, L., \& Nichols, A. (2018). Retooling poverty targeting using out-of-sample validation and machine learning. \textit{The World Bank Economic Review}, 32(3), 531-550.
\bibitem{b20} Chen, T., \& Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In \textit{Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining} (pp. 785-794).
\bibitem{b21} Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... \& Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. \textit{Advances in Neural Information Processing Systems}, 30, 3146-3154.
\bibitem{b22} Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., \& Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. \textit{Advances in Neural Information Processing Systems}, 31, 6638-6648.
\bibitem{b23} Elkan, C. (2001). The foundations of cost-sensitive learning. In \textit{International Joint Conference on Artificial Intelligence} (Vol. 17, No. 1, pp. 973-978).
\bibitem{b24} Chawla, N. V., Bowyer, K. W., Hall, L. O., \& Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. \textit{Journal of Artificial Intelligence Research}, 16, 321-357.
\end{thebibliography}
\end{document}
'''
    
    with open('Paper_Final_Draft.tex', 'w', encoding='utf-8') as f:
        f.write(final_latex)
        
    print("LaTeX file created successfully as Paper_Final_Draft.tex")

if __name__ == '__main__':
    main()
