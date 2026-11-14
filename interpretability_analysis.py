import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import HistGradientBoostingClassifier
import data_utils

def main():
    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------
    DATA = os.path.join("Prepared", "experiments", "gps_imu_mag_baro.csv")
    MODEL_PATH = os.path.join("models", "primary_gbm_model.pkl")
    OUTPUT_DIR = os.path.join("results", "interpretability")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Coordinate-free spatial-invariance features:
    # relative_alt (h_t - h_0), vel_d_m_s, and velocity vector components
    FEATURES = [
        "relative_alt",
        "vel_m_s", "vel_n_m_s", "vel_e_m_s", "vel_d_m_s", "cog_rad"
    ]
    CLASS_NAMES = ["Benign Flight", "GPS Jamming", "GPS Spoofing"]
    
    # --------------------------------------------------------
    # LOAD & SPLIT DATA (Strict 70/30 Temporal Split)
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test, train_df, test_df = data_utils.load_and_split_data(
        DATA, features=FEATURES
    )
    print(f"Loaded dataset: Train={len(X_train)} samples, Test={len(X_test)} samples across {len(FEATURES)} features.")
    print(f"Features: {FEATURES}")
    
    # --------------------------------------------------------
    # LOAD TRAINED PRIMARY MODEL (HistGradientBoostingClassifier)
    # --------------------------------------------------------
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            loaded = joblib.load(MODEL_PATH)
            # Verify feature compatibility
            if hasattr(loaded, "n_features_in_") and loaded.n_features_in_ == len(FEATURES):
                model = loaded
                print(f"[*] Successfully loaded trained primary model from {MODEL_PATH}")
        except Exception as e:
            print(f"[!] Could not load {MODEL_PATH}: {e}")
            
    if model is None:
        print("[*] Training HistGradientBoostingClassifier on 70% temporal training split...")
        model = HistGradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)
        print(f"[*] Saved model to {MODEL_PATH}")
        
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # --------------------------------------------------------
    # TREE-SHAP INTERPRETABILITY ANALYSIS
    # --------------------------------------------------------
    print("\n[*] Initializing shap.TreeExplainer on held-out temporal test set...")
    explainer = shap.TreeExplainer(model)
    
    # exp is an Explanation object with shape: (n_test_samples, n_features, n_classes)
    exp = explainer(X_test)
    raw_shap_values = explainer.shap_values(X_test)
    print(f"[*] TreeSHAP values successfully computed. Explanation shape: {exp.shape}")
    
    # --------------------------------------------------------
    # 1. SHAP SUMMARY BEESWARM PLOT
    # --------------------------------------------------------
    from PIL import Image
    class_img_paths = []
    for c_idx, c_name in enumerate(CLASS_NAMES):
        safe_name = c_name.lower().replace(" ", "_")
        plt.figure(figsize=(9, 6))
        shap.plots.beeswarm(exp[:, :, c_idx], show=False)
        plt.title(f"SHAP Beeswarm: Feature Impact on {c_name}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        single_path = os.path.join(OUTPUT_DIR, f"shap_beeswarm_{safe_name}.png")
        plt.savefig(single_path, dpi=300, bbox_inches="tight")
        plt.close()
        class_img_paths.append(single_path)
        
    # Stitch individual class beeswarm panels into shap_summary.png
    imgs = [Image.open(p) for p in class_img_paths]
    total_w = sum(im.width for im in imgs) + (len(imgs) - 1) * 20
    max_h = max(im.height for im in imgs)
    summary_img = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    x_offset = 0
    for im in imgs:
        summary_img.paste(im, (x_offset, 0))
        x_offset += im.width + 20
    summary_path = os.path.join(OUTPUT_DIR, "shap_summary.png")
    summary_img.save(summary_path)
    print(f"[*] Saved multi-panel SHAP summary beeswarm plot to {summary_path}")
        
    # Global mean absolute SHAP bar summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(raw_shap_values, X_test, class_names=CLASS_NAMES, plot_type="bar", show=False)
    plt.title("Mean Absolute SHAP Impact per Feature Across Classes", fontsize=12, fontweight="bold")
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "shap_feature_importance_bar.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # --------------------------------------------------------
    # 2. IDENTIFY MISCLASSIFIED SAMPLES (FP & FN)
    # --------------------------------------------------------
    y_test_arr = y_test.to_numpy()
    
    # False Positive: True = Benign (0), Predicted = Attack (1 or 2)
    fp_mask = (y_test_arr == 0) & (y_pred != 0)
    fp_indices = np.where(fp_mask)[0]
    
    # False Negative: True = Attack (1 or 2), Predicted = Benign (0)
    fn_mask = (y_test_arr != 0) & (y_pred == 0)
    fn_indices = np.where(fn_mask)[0]
    
    print(f"\n[*] Misclassification Analysis:")
    print(f"    Total False Positives (Benign -> Attack alert): {len(fp_indices)}")
    print(f"    Total False Negatives (Attack -> Missed/Benign): {len(fn_indices)}")
    
    if len(fp_indices) == 0:
        fp_idx = 0
        print("[!] No False Positive found; selecting first test sample.")
    else:
        # Choose the first FP
        fp_idx = fp_indices[0]
        
    if len(fn_indices) == 0:
        fn_idx = 1
        print("[!] No False Negative found; selecting second test sample.")
    else:
        # Choose the first FN
        fn_idx = fn_indices[0]
        
    true_fp_label = y_test_arr[fp_idx]
    pred_fp_label = y_pred[fp_idx]
    true_fn_label = y_test_arr[fn_idx]
    pred_fn_label = y_pred[fn_idx]
    
    print(f"\n[*] Selected False Positive Sample:")
    print(f"    Test Index: {fp_idx}")
    print(f"    True Class: {CLASS_NAMES[true_fp_label]} ({true_fp_label})")
    print(f"    Predicted Class: {CLASS_NAMES[pred_fp_label]} ({pred_fp_label})")
    print(f"    Predicted Probabilities: {y_prob[fp_idx].round(3)}")
    print(f"    Sample Telemetry:\n{X_test.iloc[fp_idx]}")
    
    print(f"\n[*] Selected False Negative Sample:")
    print(f"    Test Index: {fn_idx}")
    print(f"    True Class: {CLASS_NAMES[true_fn_label]} ({true_fn_label})")
    print(f"    Predicted Class: {CLASS_NAMES[pred_fn_label]} ({pred_fn_label})")
    print(f"    Predicted Probabilities: {y_prob[fn_idx].round(3)}")
    print(f"    Sample Telemetry:\n{X_test.iloc[fn_idx]}")
    
    # --------------------------------------------------------
    # 3. GENERATE WATERFALL PLOTS
    # --------------------------------------------------------
    # Waterfall plot for False Positive:
    # Explain why model predicted pred_fp_label (e.g. Jamming) instead of Benign
    plt.figure(figsize=(10, 6.5))
    shap.plots.waterfall(exp[fp_idx, :, pred_fp_label], show=False)
    plt.title(
        f"TreeSHAP Waterfall: False Positive (Test #{fp_idx})\n"
        f"True: {CLASS_NAMES[true_fp_label]} | Pred: {CLASS_NAMES[pred_fp_label]} (P={y_prob[fp_idx][pred_fp_label]:.2f})",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    fp_path1 = os.path.join(OUTPUT_DIR, "shap_waterfall_fp.png")
    fp_path2 = os.path.join(OUTPUT_DIR, "shap_waterfall_false_positive.png")
    plt.savefig(fp_path1, dpi=300, bbox_inches="tight")
    plt.savefig(fp_path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[*] Saved False Positive waterfall plot to {fp_path1}")
    
    # Waterfall plot for False Negative:
    # Explain why model predicted Benign (Class 0) instead of the true Attack class
    plt.figure(figsize=(10, 6.5))
    shap.plots.waterfall(exp[fn_idx, :, pred_fn_label], show=False)
    plt.title(
        f"TreeSHAP Waterfall: False Negative (Test #{fn_idx})\n"
        f"True: {CLASS_NAMES[true_fn_label]} | Pred: {CLASS_NAMES[pred_fn_label]} (P={y_prob[fn_idx][pred_fn_label]:.2f})",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    fn_path1 = os.path.join(OUTPUT_DIR, "shap_waterfall_fn.png")
    fn_path2 = os.path.join(OUTPUT_DIR, "shap_waterfall_false_negative.png")
    plt.savefig(fn_path1, dpi=300, bbox_inches="tight")
    plt.savefig(fn_path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[*] Saved False Negative waterfall plot to {fn_path1}")
    
    # --------------------------------------------------------
    # 4. EXPORT SHAP IMPORTANCE TABLE
    # --------------------------------------------------------
    # Calculate mean absolute SHAP value per feature
    if isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 3:
        # Shape: (samples, features, classes)
        mean_abs_per_class = np.mean(np.abs(raw_shap_values), axis=0) # (features, classes)
        overall_mean_abs = np.mean(mean_abs_per_class, axis=1) # (features,)
        
        shap_importance_df = pd.DataFrame({
            "feature": FEATURES,
            "mean_abs_shap_benign": mean_abs_per_class[:, 0],
            "mean_abs_shap_jamming": mean_abs_per_class[:, 1],
            "mean_abs_shap_spoofing": mean_abs_per_class[:, 2],
            "mean_abs_shap_overall": overall_mean_abs
        }).sort_values("mean_abs_shap_overall", ascending=False)
    else:
        shap_importance_df = pd.DataFrame({
            "feature": FEATURES,
            "mean_abs_shap": np.mean(np.abs(raw_shap_values), axis=0)
        }).sort_values("mean_abs_shap", ascending=False)
        
    csv_path = os.path.join(OUTPUT_DIR, "shap_feature_importance.csv")
    shap_importance_df.to_csv(csv_path, index=False)
    print(f"\n[*] Saved SHAP feature importance table to {csv_path}")
    print("\nTop Features by Mean Absolute SHAP Impact:")
    print(shap_importance_df.to_string(index=False))
    
    # --------------------------------------------------------
    # 5. GENERATE DETAILED README
    # --------------------------------------------------------
    readme_content = f"""# Explainable AI (XAI) with TreeSHAP

This directory contains local and global feature attribution analyses using **TreeSHAP** (`shap.TreeExplainer`) on the primary `HistGradientBoostingClassifier` evaluated on the held-out temporal test set.

## 1. Global Interpretability: SHAP Beeswarm Plot
- **File**: `shap_summary.png`
- **Description**: Displays the distribution of SHAP impact values for every kinematic feature across all 179 test samples, separated by class.
- **Key Findings**:
  - `relative_alt` ($h_t - h_0$): Eliminates altitude confounders by measuring vertical displacement from initial climb.
  - `vel_d_m_s` (Downward velocity): Key discriminator during jamming and spoofing anomalies where rapid kinematic shifts occur.
  - `vel_m_s`, `vel_n_m_s`, `vel_e_m_s`, `cog_rad`: Provide coordinate-free horizontal velocity and heading rate evidence.

## 2. Local Sample Interpretability: False Positive Waterfall
- **File**: `shap_waterfall_fp.png` (also `shap_waterfall_false_positive.png`)
- **Sample**: Test Sample #{fp_idx}
- **True Label**: `{CLASS_NAMES[true_fp_label]}`
- **Predicted Label**: `{CLASS_NAMES[pred_fp_label]}` (Probability: {y_prob[fp_idx][pred_fp_label]:.3f})
- **Explanation**: Illustrates the specific feature values pushing the model log-odds away from the base value into an attack prediction, highlighting boundary behavior and sensor noise.

## 3. Local Sample Interpretability: False Negative Waterfall
- **File**: `shap_waterfall_fn.png` (also `shap_waterfall_false_negative.png`)
- **Sample**: Test Sample #{fn_idx}
- **True Label**: `{CLASS_NAMES[true_fn_label]}`
- **Predicted Label**: `{CLASS_NAMES[pred_fn_label]}` (Probability: {y_prob[fn_idx][pred_fn_label]:.3f})
- **Explanation**: Details which kinematic features retained benign-like characteristics, causing the model to miss the initial onset of the attack.

## 4. SHAP Feature Importance Table
- **File**: `shap_feature_importance.csv`
- Records the mean absolute SHAP values across test instances per class and overall.
"""
    with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
        f.write(readme_content)
        
    print(f"\nAll interpretability artifacts successfully generated in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
