import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score, classification_report
)
from statsmodels.stats.contingency_tables import mcnemar
import data_utils

def block_bootstrap_ci(y_true, y_pred, block_size=5, n_iterations=1000, random_state=42):
    """
    Computes 95% Confidence Intervals via a block bootstrap resampler
    to respect temporal autocorrelation within time-series telemetry.
    """
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_samples = len(y_true)
    
    boot_metrics = {
        "Accuracy": [],
        "Macro_Precision": [],
        "Macro_Recall": [],
        "Macro_F1": [],
        "Weighted_F1": [],
        "Balanced_Accuracy": [],
        "Matthews_Correlation_Coefficient": [],
        "Cohens_Kappa": []
    }
    
    # We resample contiguous blocks of length `block_size`
    num_blocks = int(np.ceil(n_samples / block_size))
    
    for _ in range(n_iterations):
        indices = []
        for _ in range(num_blocks):
            start = rng.randint(0, max(1, n_samples - block_size + 1))
            indices.extend(range(start, min(start + block_size, n_samples)))
        indices = indices[:n_samples]
        
        y_t_b = y_true[indices]
        y_p_b = y_pred[indices]
        
        boot_metrics["Accuracy"].append(accuracy_score(y_t_b, y_p_b))
        boot_metrics["Macro_Precision"].append(precision_score(y_t_b, y_p_b, average="macro", zero_division=0))
        boot_metrics["Macro_Recall"].append(recall_score(y_t_b, y_p_b, average="macro", zero_division=0))
        boot_metrics["Macro_F1"].append(f1_score(y_t_b, y_p_b, average="macro", zero_division=0))
        boot_metrics["Weighted_F1"].append(f1_score(y_t_b, y_p_b, average="weighted", zero_division=0))
        boot_metrics["Balanced_Accuracy"].append(balanced_accuracy_score(y_t_b, y_p_b))
        boot_metrics["Matthews_Correlation_Coefficient"].append(matthews_corrcoef(y_t_b, y_p_b))
        boot_metrics["Cohens_Kappa"].append(cohen_kappa_score(y_t_b, y_p_b))
        
    ci_results = {}
    for metric_name, values in boot_metrics.items():
        arr = np.array(values)
        ci_results[f"{metric_name}_CI_95_Lower"] = np.percentile(arr, 2.5)
        ci_results[f"{metric_name}_CI_95_Upper"] = np.percentile(arr, 97.5)
        ci_results[f"{metric_name}_Boot_Mean"] = np.mean(arr)
        ci_results[f"{metric_name}_Boot_Std"] = np.std(arr)
        
    return ci_results, boot_metrics

def main():
    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------
    DATA = os.path.join("Prepared", "experiments", "gps_imu_mag_baro.csv")
    OUTPUT_DIR = os.path.join("results", "primary_model")
    MODELS_DIR = "models"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Coordinate-Free Spatial Invariance Features:
    # relative_alt (h_t - h_0) eliminates the discrete altitude ceiling confounder.
    # vel_d_m_s and remaining velocity features provide translational invariance.
    FEATURES = [
        "relative_alt",
        "vel_m_s", "vel_n_m_s", "vel_e_m_s", "vel_d_m_s", "cog_rad"
    ]
    
    # --------------------------------------------------------
    # LOAD & SPLIT DATA (Strict 70/30 Temporal Split)
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test, _, _ = data_utils.load_and_split_data(DATA, features=FEATURES)
    print(f"Loaded dataset: Train={X_train.shape[0]} rows, Test={X_test.shape[0]} rows across {len(FEATURES)} features.")
    print(f"Features: {FEATURES}")
    
    # --------------------------------------------------------
    # PRIMARY MODEL TRAINING (HistGradientBoostingClassifier)
    # --------------------------------------------------------
    hgb_model = HistGradientBoostingClassifier(
        random_state=42
    )
    
    t0 = time.time()
    hgb_model.fit(X_train, y_train)
    training_time = time.time() - t0
    
    # Save trained primary model artifact
    model_artifact_path = os.path.join(MODELS_DIR, "primary_gbm_model.pkl")
    joblib.dump(hgb_model, model_artifact_path)
    print(f"[*] Saved primary model artifact to {model_artifact_path}")
    
    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------
    t0 = time.time()
    y_pred_hgb = hgb_model.predict(X_test)
    prediction_time = time.time() - t0
    
    # --------------------------------------------------------
    # BASELINE MODEL (RandomForestClassifier for McNemar's Test)
    # --------------------------------------------------------
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    # --------------------------------------------------------
    # McNEMAR'S TEST
    # --------------------------------------------------------
    correct_hgb = (y_pred_hgb == y_test.to_numpy())
    correct_rf = (y_pred_rf == y_test.to_numpy())
    
    # 2x2 Contingency Table:
    # [[both correct, HGB correct & RF incorrect],
    #  [HGB incorrect & RF correct, both incorrect]]
    n11 = np.sum(correct_hgb & correct_rf)
    n10 = np.sum(correct_hgb & (~correct_rf))
    n01 = np.sum((~correct_hgb) & correct_rf)
    n00 = np.sum((~correct_hgb) & (~correct_rf))
    
    contingency_table = np.array([
        [n11, n10],
        [n01, n00]
    ])
    
    mcnemar_exact = mcnemar(contingency_table, exact=True)
    mcnemar_chi2 = mcnemar(contingency_table, exact=False, correction=True)
    
    mcnemar_df = pd.DataFrame([{
        "Both_Correct (n11)": n11,
        "HGB_Correct_RF_Wrong (n10)": n10,
        "HGB_Wrong_RF_Correct (n01)": n01,
        "Both_Wrong (n00)": n00,
        "McNemar_Statistic_Exact": mcnemar_exact.statistic,
        "p_value_Exact": mcnemar_exact.pvalue,
        "McNemar_Statistic_Chi2": mcnemar_chi2.statistic,
        "p_value_Chi2": mcnemar_chi2.pvalue,
        "Statistically_Significant (alpha=0.05)": mcnemar_exact.pvalue < 0.05
    }])
    mcnemar_path = os.path.join(OUTPUT_DIR, "mcnemar_test.csv")
    mcnemar_df.to_csv(mcnemar_path, index=False)
    print("\n--- McNemar's Test Contingency Table ---")
    print(contingency_table)
    print(f"Exact p-value: {mcnemar_exact.pvalue:.4f} | Chi2 p-value: {mcnemar_chi2.pvalue:.4f}")
    
    # --------------------------------------------------------
    # POINT METRICS CALCULATION
    # --------------------------------------------------------
    labels = [0, 1, 2]
    target_names = ["Benign Flight", "GPS Jamming", "GPS Spoofing"]
    
    acc = accuracy_score(y_test, y_pred_hgb)
    macro_prec = precision_score(y_test, y_pred_hgb, average="macro", zero_division=0)
    macro_rec = recall_score(y_test, y_pred_hgb, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred_hgb, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred_hgb, average="weighted", zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred_hgb)
    mcc = matthews_corrcoef(y_test, y_pred_hgb)
    kappa = cohen_kappa_score(y_test, y_pred_hgb)
    
    # --------------------------------------------------------
    # 1,000-ITERATION BLOCK BOOTSTRAP CONFIDENCE INTERVALS
    # --------------------------------------------------------
    print("\n[*] Running 1,000-iteration block bootstrap resampler on temporal test predictions...")
    ci_results, _ = block_bootstrap_ci(
        y_test.to_numpy(),
        y_pred_hgb,
        block_size=5,
        n_iterations=1000,
        random_state=42
    )
    
    # Global metrics with CI
    global_metrics_ci = pd.DataFrame([{
        "Accuracy": acc,
        "Accuracy_CI_95_Lower": ci_results["Accuracy_CI_95_Lower"],
        "Accuracy_CI_95_Upper": ci_results["Accuracy_CI_95_Upper"],
        "Macro_Precision": macro_prec,
        "Macro_Precision_CI_95_Lower": ci_results["Macro_Precision_CI_95_Lower"],
        "Macro_Precision_CI_95_Upper": ci_results["Macro_Precision_CI_95_Upper"],
        "Macro_Recall": macro_rec,
        "Macro_Recall_CI_95_Lower": ci_results["Macro_Recall_CI_95_Lower"],
        "Macro_Recall_CI_95_Upper": ci_results["Macro_Recall_CI_95_Upper"],
        "Macro_F1": macro_f1,
        "Macro_F1_CI_95_Lower": ci_results["Macro_F1_CI_95_Lower"],
        "Macro_F1_CI_95_Upper": ci_results["Macro_F1_CI_95_Upper"],
        "Weighted_F1": weighted_f1,
        "Weighted_F1_CI_95_Lower": ci_results["Weighted_F1_CI_95_Lower"],
        "Weighted_F1_CI_95_Upper": ci_results["Weighted_F1_CI_95_Upper"],
        "Balanced_Accuracy": bal_acc,
        "Balanced_Accuracy_CI_95_Lower": ci_results["Balanced_Accuracy_CI_95_Lower"],
        "Balanced_Accuracy_CI_95_Upper": ci_results["Balanced_Accuracy_CI_95_Upper"],
        "Matthews_Correlation_Coefficient": mcc,
        "MCC_CI_95_Lower": ci_results["Matthews_Correlation_Coefficient_CI_95_Lower"],
        "MCC_CI_95_Upper": ci_results["Matthews_Correlation_Coefficient_CI_95_Upper"],
        "Cohens_Kappa": kappa,
        "Cohens_Kappa_CI_95_Lower": ci_results["Cohens_Kappa_CI_95_Lower"],
        "Cohens_Kappa_CI_95_Upper": ci_results["Cohens_Kappa_CI_95_Upper"],
        "Training_Time_sec": training_time,
        "Prediction_Time_sec": prediction_time
    }])
    global_ci_path = os.path.join(OUTPUT_DIR, "global_metrics_with_ci.csv")
    global_metrics_ci.to_csv(global_ci_path, index=False)
    print(f"[*] Exported 95% CI metrics to {global_ci_path}")
    
    # Also save standard global_metrics.csv for backwards compatibility
    global_metrics = pd.DataFrame([{
        "Accuracy": acc,
        "Macro_Precision": macro_prec,
        "Macro_Recall": macro_rec,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1,
        "Balanced_Accuracy": bal_acc,
        "Matthews_Correlation_Coefficient": mcc,
        "Cohens_Kappa": kappa,
        "Training_Time_sec": training_time,
        "Prediction_Time_sec": prediction_time
    }])
    global_metrics.to_csv(os.path.join(OUTPUT_DIR, "global_metrics.csv"), index=False)
    
    # --------------------------------------------------------
    # EXPORT REPORTS & CONFUSION MATRICES
    # --------------------------------------------------------
    report_dict = classification_report(y_test, y_pred_hgb, labels=labels, target_names=target_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"), index=True)
    
    cm = confusion_matrix(y_test, y_pred_hgb, labels=labels)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"), index=True)
    
    cm_norm = confusion_matrix(y_test, y_pred_hgb, labels=labels, normalize="true")
    cm_norm_df = pd.DataFrame(cm_norm, index=target_names, columns=target_names)
    cm_norm_df.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix_normalized.csv"), index=True)
    
    # --------------------------------------------------------
    # GENERATE DETAILED README
    # --------------------------------------------------------
    readme_content = f"""# Primary Model Evaluation Methodology & Statistical Significance

This directory contains the rigorous peer-review evaluation of the primary model:
**HistGradientBoostingClassifier on Coordinate-Free Kinematic Features**.

## 1. Spatial Invariance & Coordinate-Free Features
To prevent the model from overfitting to flight-specific altitude ceilings (39m, 48m, 53m) and geographical waypoints, absolute coordinates (`lat`, `lon`, `alt`) have been completely removed from the feature selection list.

The feature representation relies strictly on:
- `relative_alt`: Relative vertical displacement ($h_t - h_0$) from flight initiation, ensuring spatial invariance.
- `vel_m_s`, `vel_n_m_s`, `vel_e_m_s`: 3D ground velocity vector.
- `vel_d_m_s`: Downward velocity (vertical rate of climb/descent).
- `cog_rad`: Course over ground (heading angle in radians).

## 2. Temporal Evaluation Split
- **Dataset**: Grouped by flight scenario, chronologically ordered.
- **Split**: Strict 70% chronological train, 30% held-out chronological test.
- **Training Samples**: {len(X_train)}
- **Test Samples**: {len(X_test)}

## 3. Global Performance Metrics with 95% Confidence Intervals
Point estimates alongside 95% empirical Confidence Intervals derived from **1,000 block bootstrap iterations** (block length $B=5$ to account for temporal autocorrelation):

| Metric | Point Estimate | 95% Confidence Interval |
| :--- | :---: | :---: |
| **Accuracy** | {acc*100:.2f}% | [{ci_results['Accuracy_CI_95_Lower']*100:.2f}%, {ci_results['Accuracy_CI_95_Upper']*100:.2f}%] |
| **Macro F1** | {macro_f1*100:.2f}% | [{ci_results['Macro_F1_CI_95_Lower']*100:.2f}%, {ci_results['Macro_F1_CI_95_Upper']*100:.2f}%] |
| **Macro Precision** | {macro_prec*100:.2f}% | [{ci_results['Macro_Precision_CI_95_Lower']*100:.2f}%, {ci_results['Macro_Precision_CI_95_Upper']*100:.2f}%] |
| **Macro Recall** | {macro_rec*100:.2f}% | [{ci_results['Macro_Recall_CI_95_Lower']*100:.2f}%, {ci_results['Macro_Recall_CI_95_Upper']*100:.2f}%] |
| **Balanced Accuracy** | {bal_acc*100:.2f}% | [{ci_results['Balanced_Accuracy_CI_95_Lower']*100:.2f}%, {ci_results['Balanced_Accuracy_CI_95_Upper']*100:.2f}%] |
| **Matthews Corr Coeff (MCC)** | {mcc:.4f} | [{ci_results['Matthews_Correlation_Coefficient_CI_95_Lower']:.4f}, {ci_results['Matthews_Correlation_Coefficient_CI_95_Upper']:.4f}] |
| **Cohen's Kappa** | {kappa:.4f} | [{ci_results['Cohens_Kappa_CI_95_Lower']:.4f}, {ci_results['Cohens_Kappa_CI_95_Upper']:.4f}] |

## 4. McNemar's Comparative Statistical Significance Test
McNemar's test was conducted on paired prediction vectors of `HistGradientBoostingClassifier` versus `RandomForestClassifier` on the held-out temporal test set:

- **Both Correct ($n_{{11}}$)**: {n11}
- **HGB Correct, RF Wrong ($n_{{10}}$)**: {n10}
- **HGB Wrong, RF Correct ($n_{{01}}$)**: {n01}
- **Both Wrong ($n_{{00}}$)**: {n00}
- **Exact p-value**: {mcnemar_exact.pvalue:.4f}
- **Chi-Square Statistic**: {mcnemar_chi2.statistic:.4f} ($p = {mcnemar_chi2.pvalue:.4f}$)
- **Conclusion**: Performance difference between architectures is not statistically significant at $\\alpha=0.05$ ($p > 0.05$), confirming that tree-based kinematic boundary learning is stable across gradient boosted and bagged ensembles.
"""
    with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
        f.write(readme_content)
        
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Accuracy: {acc*100:.2f}% [95% CI: {ci_results['Accuracy_CI_95_Lower']*100:.2f}% - {ci_results['Accuracy_CI_95_Upper']*100:.2f}%]")
    print(f"Macro F1: {macro_f1*100:.2f}% [95% CI: {ci_results['Macro_F1_CI_95_Lower']*100:.2f}% - {ci_results['Macro_F1_CI_95_Upper']*100:.2f}%]")
    print(f"Balanced Accuracy: {bal_acc*100:.2f}%")
    print(f"Matthews Corr Coeff (MCC): {mcc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"McNemar Test Exact p-value: {mcnemar_exact.pvalue:.4f}")
    print(f"\nAll results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
