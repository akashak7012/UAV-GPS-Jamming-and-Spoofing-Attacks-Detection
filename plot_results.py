import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Styling configuration
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.autolayout': True
})

def plot_experiment(csv_file, output_filename, title):
    file_path = os.path.join(RESULTS_DIR, csv_file)
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping plot.")
        return
        
    df = pd.read_csv(file_path)
    
    # Check if necessary columns exist
    if "Name" not in df.columns and "Model" in df.columns:
        df = df.rename(columns={"Model": "Name"})
        
    names = df["Name"].tolist()
    accuracy = df["Accuracy"].tolist()
    macro_f1 = df["Macro_F1"].tolist()
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rects1 = ax.bar(x - width/2, accuracy, width, label='Accuracy', color='#4C72B0')
    rects2 = ax.bar(x + width/2, macro_f1, width, label='Macro F1', color='#DD8452')
    
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)  # Non-misleading y-axis
    ax.legend(loc='upper right')
    
    # Add gridlines behind bars
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
                        
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, output_filename), dpi=300)
    plt.close(fig)
    print(f"Generated: {output_filename}")


def main():
    print(f"Generating publication-quality figures in {FIGURES_DIR}...")
    
    # 1. Feature Group Comparison
    plot_experiment(
        "feature_group_results.csv",
        "feature_group_comparison.png",
        "Feature-Group Evaluation (Random Forest, Temporal Test)"
    )
    
    # 2. Sensor Ablation Comparison
    plot_experiment(
        "sensor_ablation.csv",
        "sensor_ablation_comparison.png",
        "Sensor Ablation Evaluation (Random Forest, Temporal Test)"
    )
    
    # 3. Model Comparison
    plot_experiment(
        "model_comparison.csv",
        "model_comparison.png",
        "Model Comparison (GPS Position + Velocity, Temporal Test)"
    )
    
    # 4. Robustness Comparison
    plot_experiment(
        "robustness.csv",
        "robustness_comparison.png",
        "Robustness Evaluation (Gradient Boosting, Temporal Test)"
    )
    
    # Generate README.md
    readme_content = """# Publication Figures

This directory contains publication-ready bar charts for the primary experiments evaluated on the temporal 70/30 split.

1. **`feature_group_comparison.png`**: Highlights the performance of isolating specific combinations of GPS features (Position, Velocity, Quality, Motion). Evaluated using Random Forest.
2. **`sensor_ablation_comparison.png`**: Demonstrates the impact of adding or removing specific non-GPS sensor streams (IMU, MAG, BARO). Evaluated using Random Forest.
3. **`model_comparison.png`**: Compares different machine learning architectures (Random Forest, Gradient Boosting, SVM, Logistic Regression, XGBoost) applied specifically to the optimal GPS Position + Velocity feature set.
4. **`robustness_comparison.png`**: Tests the winning Gradient Boosting model against specific challenging subsets of features (e.g., removing absolute latitude/longitude, or relying purely on relative motion/quality) to determine robustness.

### Methodology Notes:
*   All charts report both **Accuracy** and **Macro F1**.
*   The Y-axis is strictly fixed from 0.0 to 1.0 to prevent misleading visual scaling.
*   Values are evaluated strictly on the blind chronological temporal test sequence to prevent time-series data leakage.
"""
    with open(os.path.join(FIGURES_DIR, "README.md"), "w") as f:
        f.write(readme_content)
        
    print("All figures and README generated successfully.")

if __name__ == "__main__":
    main()

