# Explainable AI (XAI) with TreeSHAP

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
- **Sample**: Test Sample #61
- **True Label**: `Benign Flight`
- **Predicted Label**: `GPS Jamming` (Probability: 0.714)
- **Explanation**: Illustrates the specific feature values pushing the model log-odds away from the base value into an attack prediction, highlighting boundary behavior and sensor noise.

## 3. Local Sample Interpretability: False Negative Waterfall
- **File**: `shap_waterfall_fn.png` (also `shap_waterfall_false_negative.png`)
- **Sample**: Test Sample #93
- **True Label**: `GPS Jamming`
- **Predicted Label**: `Benign Flight` (Probability: 0.999)
- **Explanation**: Details which kinematic features retained benign-like characteristics, causing the model to miss the initial onset of the attack.

## 4. SHAP Feature Importance Table
- **File**: `shap_feature_importance.csv`
- Records the mean absolute SHAP values across test instances per class and overall.
