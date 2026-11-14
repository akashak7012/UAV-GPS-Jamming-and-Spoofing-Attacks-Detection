# Robust Multi-Class Detection of UAV GPS Jamming and Spoofing Attacks Using Coordinate-Free Kinematic Discrepancy

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/scikit--learn-1.6+-orange.svg)](https://scikit-learn.org/)
[![XAI](https://img.shields.io/badge/Explainability-TreeSHAP-red.svg)](https://shap.readthedocs.io/)

An end-to-end, scientifically validated machine learning framework for detecting and classifying GPS jamming and spoofing attacks against Unmanned Aerial Vehicles (UAVs). The pipeline enforces **translational spatial invariance**, **causal temporal validation**, **statistical significance testing**, and **local/global Explainable AI (TreeSHAP)** to eliminate methodological data leakage and spatial overfitting prevalent in existing literature.

---

## 1. Research Motivation & Methodological Contributions

Traditional machine learning approaches for UAV cyber-attack detection often report near-perfect metrics (e.g., >99% accuracy) due to subtle methodological flaws:
1. **Temporal Data Leakage:** Random `train_test_split` shuffles autocorrelated time-series frames, allowing models to interpolate between temporally adjacent samples.
2. **Spatial Confounders:** Classifiers trained on absolute geographical coordinates (`lat`, `lon`, `alt`) memorize flight-specific waypoints or test ceilings (e.g., discrete flight altitudes of 39m, 48m, and 53m) rather than true physical attack dynamics.
3. **Auxiliary Sensor Noise:** Indiscriminately fusing high-frequency IMU, magnetometer, and barometer signals can introduce platform-specific noise without improving detection signal-to-noise ratio.

### Our Solutions:
- **Coordinate-Free Spatial Invariance:** Absolute coordinates (`lat`, `lon`, `alt`) are completely excluded. Vertical motion is captured via relative altitude ($h_t - h_0$), representing displacement from flight initiation, paired with 3D velocity vectors and course-over-ground angles.
- **Strict Causal Chronological Validation:** Telemetry is split chronologically (first 70% for training, final 30% held-out for testing). Sensor alignment uses strictly backward causal merges (`pd.merge_asof`) with a 1-second maximum tolerance to prevent future lookahead.
- **Statistical Significance & Empirical Confidence Intervals:** Replaced single-point metrics with **95% Confidence Intervals via a 1,000-iteration block bootstrap resampler** ($B=5$) to preserve temporal autocorrelation, alongside **McNemar's test** against Random Forest ensembles.
- **Explainable AI (TreeSHAP):** Global multi-class beeswarm analysis and local waterfall attributions for misclassified samples (False Positives and False Negatives) to verify physical reasoning.
- **Cross-Airframe Benchmark (LOAO):** 4-fold Leave-One-Airframe-Out validation across fixed-wing, multicopter, tailsitter, and VTOL architectures.

---

## 2. Attack Taxonomy & Operational Classes

The framework discriminates between three distinct operational flight regimes:

| Class ID | Class Label | Description | Physical Kinematic Impact |
|:---:|:---|:---|:---|
| **0** | **Benign Flight** | Authentic, nominal flight under valid GPS lock. | Smooth velocity gradients, consistent heading angles, nominal relative altitude. |
| **1** | **GPS Jamming** | High-power RF noise overpowering satellite signals. | Severe signal degradation, loss of fix, sudden velocity fluctuations, vertical descent anomalies. |
| **2** | **GPS Spoofing** | False counterfeit satellite signals injected by adversary. | Coordinate drift, course-over-ground discrepancies, false trajectory diversions. |

---

## 3. Feature Engineering: Spatial Invariance

To guarantee translational invariance and prevent geographic memorization, the primary classifier relies strictly on **6 coordinate-free kinematic features**:

| Feature | Physical Meaning | Unit | Rationale |
|:---|:---|:---:|:---|
| `relative_alt` | Altitude relative to initial sample ($h_t - h_0$) | $\text{m}$ | Removes discrete flight altitude ceiling confounders (39m, 48m, 53m). |
| `vel_m_s` | Total GPS ground speed magnitude | $\text{m/s}$ | Reflects overall kinetic energy and velocity surges. |
| `vel_n_m_s` | Northward horizontal velocity | $\text{m/s}$ | Directional planar motion component. |
| `vel_e_m_s` | Eastward horizontal velocity | $\text{m/s}$ | Directional planar motion component. |
| `vel_d_m_s` | Downward vertical velocity | $\text{m/s}$ | Sensitive to rapid climb/descent anomalies during jamming or capture. |
| `cog_rad` | Course over ground (heading angle) | $\text{rad}$ | Captures sudden heading discontinuities characteristic of spoofing transitions. |

---

## 4. Evaluated Results & Statistical Verification

Evaluated strictly on the **179-sample held-out chronological test set** (Train: 414 samples, Test: 179 samples):

### Global Performance with 95% Block Bootstrap Confidence Intervals
*Derived from 1,000 block bootstrap iterations with block length $B=5$ (accounting for temporal autocorrelation):*

| Metric | Point Estimate | 95% Confidence Interval |
|:---|:---:|:---:|
| **Accuracy** | **78.21%** | **[67.60%, 88.83%]** |
| **Macro F1-Score** | **78.89%** | **[67.77%, 88.82%]** |
| **Macro Precision** | **81.89%** | **[71.03%, 90.88%]** |
| **Macro Recall** | **77.74%** | **[67.11%, 89.38%]** |
| **Balanced Accuracy** | **77.74%** | **[67.11%, 89.38%]** |
| **Matthews Correlation Coefficient (MCC)** | **0.6694** | **[0.5073, 0.8249]** |
| **Cohen's Kappa** | **0.6588** | **[0.4844, 0.8198]** |

> **Note on Performance:** The coordinate-free model reports **78.21% accuracy** (compared to earlier naive models reporting >93% by memorizing fixed flight ceilings). This represents genuine, spatially-invariant kinematic detection that generalizes across missions.

### McNemar's Comparative Significance Test
Paired contingency analysis comparing `HistGradientBoostingClassifier` against `RandomForestClassifier`:

```text
                  Random Forest
               Correct    Incorrect
HGB  Correct     134          6
   Incorrect      15         24
```
- **Exact McNemar p-value:** `0.0784` ($p > 0.05$)
- **Continuity-Corrected $\chi^2$ Statistic:** `3.0476` ($p = 0.0809$)
- **Conclusion:** No statistically significant performance divergence between boosting and bagging ensembles at $\alpha=0.05$, demonstrating that tree-based kinematic boundary learning is structurally robust across model paradigms.

---

## 5. Explainable AI (TreeSHAP)

Global and local model interpretability is implemented in [`interpretability_analysis.py`](file:///run/media/akash/Windows-SSD/Projects/cyber/UAVAttackData/interpretability_analysis.py) via `shap.TreeExplainer`:

- **Multi-Class SHAP Beeswarm Plot:** [`results/interpretability/shap_summary.png`](file:///run/media/akash/Windows-SSD/Projects/cyber/UAVAttackData/results/interpretability/shap_summary.png) illustrates the distribution of attributions across Benign, Jamming, and Spoofing classes.
  - Course rate (`cog_rad`) and relative altitude (`relative_alt`) are the primary discriminators across all attack types.
- **False Positive Attribution (Test #61):** [`results/interpretability/shap_waterfall_fp.png`](file:///run/media/akash/Windows-SSD/Projects/cyber/UAVAttackData/results/interpretability/shap_waterfall_fp.png) shows how transient wind gusts simulated vertical velocity shifts (`vel_d_m_s = -0.435 m/s`, $+0.75$ SHAP) that induced a false jamming alarm.
- **False Negative Attribution (Test #93):** [`results/interpretability/shap_waterfall_fn.png`](file:///run/media/akash/Windows-SSD/Projects/cyber/UAVAttackData/results/interpretability/shap_waterfall_fn.png) highlights that calm early-onset heading (`cog_rad = 0.195 rad`, $+9.02$ SHAP towards Benign) temporarily masked the attack onset.

---

## 6. Repository Structure

```text
UAVAttackData/
├── .gitignore                         # Excludes raw telemetry, venv, and binary models
├── requirements.txt                   # Production dependencies (scikit-learn, shap, statsmodels)
├── README.md                          # Primary research documentation
├── FINAL_RESEARCH_REPORT.md           # Formal engineering research report
├── EXPERIMENT_GUIDE.md                # Experimental reproduction guide
│
├── Core Python Scripts/
│   ├── data_utils.py                  # Causal chronological 70/30 split loader
│   ├── prepare_fusion_dataset.py      # Causal backward sensor synchronization
│   ├── create_experiment_datasets.py  # Spatial-invariance feature engineering
│   ├── evaluate_primary_model.py      # HistGradientBoosting, Bootstrap CI & McNemar test
│   ├── run_loao_experiments.py        # Leave-One-Airframe-Out (LOAO) cross-airframe benchmark
│   ├── temporal_robustness_test.py    # Temporal stability across split ratios (70/30, 75/25, 80/20)
│   ├── interpretability_analysis.py   # TreeSHAP beeswarm and FP/FN waterfall plots
│   ├── realtime_simulation.py         # Sequential streaming inference simulation
│   ├── predict.py                     # Batch / single-file CLI prediction utility
│   ├── plot_results.py                # Publication-grade chart generation
│   ├── eda_analysis.py                # Exploratory data distribution analysis
│   └── draw_architecture.py           # System architecture diagram generator
│
├── Prepared/
│   └── experiments/                   # Preprocessed benchmark datasets
│       ├── gps_only.csv               # GPS features (593 rows)
│       ├── gps_imu.csv                # GPS + IMU fusion (593 rows)
│       ├── gps_imu_mag_baro.csv       # Multi-sensor fusion (593 rows)
│       └── full_joined_dataset.csv    # Cross-airframe LOAO dataset (4 airframes)
│
└── results/                           # Experimental outputs, metrics, and figures
    ├── primary_model/                 # Metrics with 95% CI & McNemar test results
    ├── interpretability/              # SHAP summary beeswarm and waterfall plots
    ├── loao/                          # LOAO cross-airframe benchmark tables
    └── figures/                       # Publication figures and system diagrams
```

---

## 7. Quickstart & Reproduction Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/akashak7012/UAV-GPS-Jamming-and-Spoofing-Attacks-Detection
cd UAVAttackData

# Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Feature Extraction & Dataset Preparation
Extract coordinate-free spatial invariance features from synchronized records:
```bash
python create_experiment_datasets.py
```

### 3. Primary Model Evaluation & Statistical Significance
Train `HistGradientBoostingClassifier`, run the 1,000-iteration block bootstrap resampler, and conduct McNemar's test:
```bash
python evaluate_primary_model.py
```
*Outputs saved to `results/primary_model/global_metrics_with_ci.csv` and `results/primary_model/mcnemar_test.csv`.*

### 4. Explainable AI with TreeSHAP
Generate the global multi-class beeswarm plot and local error case waterfall plots:
```bash
python interpretability_analysis.py
```
*Outputs saved to `results/interpretability/shap_summary.png` and `results/interpretability/shap_waterfall_*.png`.*

### 5. Cross-Airframe LOAO Benchmark
Run 4-fold Leave-One-Airframe-Out cross-validation across PLANE, QUAD, TAIL, and VTOL:
```bash
python run_loao_experiments.py
```
*Outputs saved to `results/loao/sample_level_results.csv` and `results/loao/episode_level_results.csv`.*

### 6. Temporal Robustness Validation
Evaluate sensitivity to changing chronological split boundaries (70/30, 75/25, 80/20):
```bash
python temporal_robustness_test.py
```

### 7. Real-Time Streaming Simulation
Simulate per-sample streaming telemetry inference on the held-out temporal sequence:
```bash
python realtime_simulation.py
```

### 8. CLI Batch Prediction
Run inference on custom telemetry:
```bash
python predict.py --input Prepared/experiments/gps_only.csv --output custom_predictions.csv
```

---

## 8. Citation & Academic Reference
If using this codebase or methodology in academic research, please reference:
```bibtex
@article{uav_gps_attack_detection_2026,
  title={Robust Multi-Class Detection of UAV GPS Jamming and Spoofing Attacks Using Coordinate-Free Kinematic Discrepancy},
  author={UAV Cyber Research Team},
  journal={IEEE Transactions on Aerospace and Electronic Systems / Cyber-Physical Systems},
  year={2026}
}
```
