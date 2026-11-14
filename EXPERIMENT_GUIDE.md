# Experimental Reproduction & Methodological Guide

This guide is intended for researchers seeking to understand the methodological decisions, reproduce the findings, or extend the evaluation pipeline for the UAV GPS Jamming and Spoofing Attack Detection project.

---

## 1. Addressing Methodological Pitfalls in Cyber-Physical ML

### A. Temporal Autocorrelation & Data Leakage
In standard machine learning paradigms, `train_test_split` randomly shuffles rows. In UAV aviation telemetry, measurements sampled sequentially at 1Hz–50Hz are heavily auto-correlated. Randomly sampling these rows leaks future state information into the training set (forward-looking data leakage).

**Our Causal Solution:**
1. **Backward-Only Synchronization (`prepare_fusion_dataset.py`)**: Data is synchronized using `pd.merge_asof(direction="backward", tolerance=1_000_000)`. When aligning IMU, Magnetometer, and Barometer readings to the GPS master timeline, the algorithm strictly accesses sensor readings from the past or current timestamp (within 1 second tolerance).
2. **Chronological Splitting (`data_utils.py`)**: Telemetry is sorted chronologically per flight trajectory. The first 70% of each flight is assigned to the training split, and the final 30% is held out exclusively for testing.

### B. Spatial Confounders & Geographical Memorization
Models trained on absolute coordinates (`lat`, `lon`, `alt`) frequently overfit to discrete mission altitudes (e.g., flight ceilings of 39m, 48m, and 53m) or geographical flight boundaries, giving an illusion of high accuracy (>93%) that collapses when deployed in a new airspace.

**Our Spatial Invariance Solution:**
- Absolute coordinates (`lat`, `lon`, `alt`) are strictly dropped.
- Vertical displacement is captured via `relative_alt` ($h_t - h_0$), representing altitude change relative to the initial sample of each flight.
- 3D velocity vectors (`vel_d_m_s`, `vel_n_m_s`, `vel_e_m_s`, `vel_m_s`) and course over ground (`cog_rad`) provide complete translational and rotational coordinate invariance.

---

## 2. Core Experimental Benchmarks

### 1. Primary Model Evaluation & Statistical Significance (`evaluate_primary_model.py`)
- **Architecture:** `HistGradientBoostingClassifier(random_state=42)`
- **Features:** `['relative_alt', 'vel_m_s', 'vel_n_m_s', 'vel_e_m_s', 'vel_d_m_s', 'cog_rad']`
- **Statistical Significance:**
  - Evaluates 95% Confidence Intervals via a 1,000-iteration **Block Bootstrap** ($B=5$ samples, respecting temporal autocorrelation).
  - Conducts **McNemar's test** against `RandomForestClassifier(n_estimators=100)` to evaluate architectural significance on paired test prediction vectors.
- **Results:** 
  - Accuracy: **78.21%** [95% CI: 67.60%, 88.83%]
  - Macro F1: **78.89%** [95% CI: 67.77%, 88.82%]
  - McNemar $p = 0.0784$ (no statistically significant difference at $\alpha=0.05$).

### 2. Explainable AI with TreeSHAP (`interpretability_analysis.py`)
- Initializes `shap.TreeExplainer` on the primary model over the held-out temporal test set.
- Generates global multi-class beeswarm plots (`results/interpretability/shap_summary.png`).
- Explains local misclassifications using `shap.plots.waterfall` on:
  - **False Positive (Test #61):** True Benign flight misclassified as GPS Jamming due to wind-induced vertical velocity shifts.
  - **False Negative (Test #93):** True GPS Jamming misclassified as Benign during early onset before course angle disruption fully manifested.

### 3. Cross-Airframe LOAO Benchmark (`run_loao_experiments.py`)
- Implements 4-fold Leave-One-Airframe-Out validation across four distinct UAV airframes:
  - `PLANE` (Fixed-wing)
  - `QUAD` (Multicopter)
  - `TAIL` (Tailsitter)
  - `VTOL` (Tiltrotor VTOL)
- Evaluates feature sets: `kinematics_only`, `pos_speed_only`, `discrepancy_rolling_only`, and `complete_features`.

### 4. Temporal Robustness Sensitivity (`temporal_robustness_test.py`)
- Tests stability across chronological split ratios: 70/30, 75/25, and 80/20 across multiple random seeds (42, 123, 2026).
- *Finding:* Model retains ~78–80% accuracy, demonstrating that kinematic boundaries are robust across flight phases.

### 5. Real-Time Streaming Simulation (`realtime_simulation.py`)
- Streams the held-out test sequence row-by-row, recording per-sample latency and running accuracy.
- *Finding:* Average inference latency is ~28.7 ms/sample on CPU, validating real-time feasibility for onboard companion computers.

---

## 3. Extending the Framework

If you wish to test new features or architectures:
1. **Feature Engineering:** Add new coordinate-free features in `create_experiment_datasets.py` and ensure they are translational-invariant.
2. **Temporal Split:** Adjust splitting boundaries in `data_utils.py` if testing rolling-origin cross-validation.
3. **Model Selection:** Add candidate estimators to `evaluate_primary_model.py`. Ensure scalers or normalizers are fit strictly on `X_train` to prevent test-set data leakage.
