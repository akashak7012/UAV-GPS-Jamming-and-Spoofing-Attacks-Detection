import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def main():
    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------
    DATA = os.path.join("Prepared", "experiments", "gps_imu_mag_baro.csv")
    OUTPUT_CSV = os.path.join("results", "temporal_robustness.csv")
    OUTPUT_PNG = os.path.join("results", "temporal_robustness.png")
    
    FEATURES = [
        "relative_alt",
        "vel_m_s", "vel_n_m_s", "vel_e_m_s", "vel_d_m_s", "cog_rad"
    ]
    
    SPLIT_RATIOS = [0.70, 0.75, 0.80]
    SEEDS = [42, 123, 2026]
    
    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------
    df = pd.read_csv(DATA)
    df = df.sort_values(["attack_type", "timestamp"]).reset_index(drop=True)
    
    results = []
    
    # --------------------------------------------------------
    # RUN EXPERIMENTS
    # --------------------------------------------------------
    for ratio in SPLIT_RATIOS:
        print(f"\n--- Testing Temporal Split Ratio: {int(ratio*100)}/{int((1-ratio)*100)} ---")
        
        # Split Data
        train_parts = []
        test_parts = []
        for attack_type, group in df.groupby("attack_type"):
            group = group.sort_values("timestamp")
            split_idx = int(len(group) * ratio)
            train_parts.append(group.iloc[:split_idx])
            test_parts.append(group.iloc[split_idx:])
            
        train_df = pd.concat(train_parts)
        test_df = pd.concat(test_parts)
        
        X_train = train_df[FEATURES]
        y_train = train_df["label"]
        X_test = test_df[FEATURES]
        y_test = test_df["label"]
        
        for seed in SEEDS:
            model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=seed
            )
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, pred)
            mac_prec = precision_score(y_test, pred, average="macro", zero_division=0)
            mac_rec = recall_score(y_test, pred, average="macro", zero_division=0)
            mac_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
            
            results.append({
                "Train_Ratio": ratio,
                "Model_Seed": seed,
                "Accuracy": acc,
                "Macro_Precision": mac_prec,
                "Macro_Recall": mac_rec,
                "Macro_F1": mac_f1
            })
            
            print(f"Seed {seed:4d} | Acc: {acc:.4f} | F1: {mac_f1:.4f}")

    # --------------------------------------------------------
    # SAVE & PLOT
    # --------------------------------------------------------
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    
    # Calculate means and stds for plotting
    agg_df = results_df.groupby("Train_Ratio").agg(
        Acc_mean=("Accuracy", "mean"),
        Acc_std=("Accuracy", "std"),
        F1_mean=("Macro_F1", "mean"),
        F1_std=("Macro_F1", "std")
    ).reset_index()
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(agg_df))
    width = 0.35
    
    plt.bar(x - width/2, agg_df["Acc_mean"], width, yerr=agg_df["Acc_std"], capsize=5, label='Accuracy', color='skyblue')
    plt.bar(x + width/2, agg_df["F1_mean"], width, yerr=agg_df["F1_std"], capsize=5, label='Macro F1', color='salmon')
    
    plt.xlabel("Temporal Train Ratio", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("GBM Performance vs Temporal Split Ratio\n(Error bars indicate model stochasticity across seeds)", fontsize=14)
    plt.xticks(x, [f"{int(r*100)}/{int((1-r)*100)}" for r in agg_df["Train_Ratio"]])
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    print(f"\nSaved CSV: {OUTPUT_CSV}")
    print(f"Saved Plot: {OUTPUT_PNG}")

if __name__ == "__main__":
    main()

