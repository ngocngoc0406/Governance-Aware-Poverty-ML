import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_pipeline():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Define steps
    steps = [
        "1. Raw Dataset\n(Costa Rican Poverty Survey)",
        "2. Data Cleaning\n(Handling Mixed Types, Missing Values)",
        "3. Feature Engineering\n(Categorical Encoding, Scaling)",
        "4. Train/Test Split\n(80% Train, 20% Test, Stratified)",
        "5. Cross Validation & Training\n(LR, RF, XGB, LGBM, CatBoost)",
        "6. Model Comparison\n(Accuracy, Precision, AUC)",
        "7. Statistical Significance\n(McNemar's Test)",
        "8. Interpretability\n(SHAP Analysis)",
        "9. Policy Recommendation\n(Targeting Efficiency)"
    ]
    
    # Box parameters
    box_w = 0.6
    box_h = 0.08
    start_y = 0.95
    gap_y = 0.1
    
    for i, step in enumerate(steps):
        # Draw box
        y = start_y - i * gap_y
        rect = patches.FancyBboxPatch(
            (0.5 - box_w/2, y - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.02",
            edgecolor='black', facecolor='#e0f2f1', lw=2
        )
        ax.add_patch(rect)
        
        # Add text
        ax.text(0.5, y, step, ha='center', va='center', fontsize=11, fontweight='bold', fontfamily='sans-serif')
        
        # Draw arrow to next step
        if i < len(steps) - 1:
            ax.annotate('', xy=(0.5, y - box_h/2 - 0.005), 
                        xytext=(0.5, y - box_h/2 - gap_y + box_h + 0.005),
                        arrowprops=dict(arrowstyle='<-', lw=2, color='gray'))

    plt.title("End-to-End Research and ML Pipeline", fontsize=14, fontweight='bold', y=1.02)
    plt.savefig('figures/Figure_1_Research_Pipeline.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    draw_pipeline()
    print("Pipeline figure generated at figures/Figure_1_Research_Pipeline.png")
