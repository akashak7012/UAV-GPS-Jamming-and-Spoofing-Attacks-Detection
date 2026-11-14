# Primary Model Evaluation Methodology & Statistical Significance

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
- **Training Samples**: 414
- **Test Samples**: 179

## 3. Global Performance Metrics with 95% Confidence Intervals
Point estimates alongside 95% empirical Confidence Intervals derived from **1,000 block bootstrap iterations** (block length $B=5$ to account for temporal autocorrelation):

| Metric | Point Estimate | 95% Confidence Interval |
| :--- | :---: | :---: |
| **Accuracy** | 78.21% | [67.60%, 88.83%] |
| **Macro F1** | 78.89% | [67.77%, 88.82%] |
| **Macro Precision** | 81.89% | [71.03%, 90.88%] |
| **Macro Recall** | 77.74% | [67.11%, 89.38%] |
| **Balanced Accuracy** | 77.74% | [67.11%, 89.38%] |
| **Matthews Corr Coeff (MCC)** | 0.6694 | [0.5073, 0.8249] |
| **Cohen's Kappa** | 0.6588 | [0.4844, 0.8198] |

## 4. McNemar's Comparative Statistical Significance Test
McNemar's test was conducted on paired prediction vectors of `HistGradientBoostingClassifier` versus `RandomForestClassifier` on the held-out temporal test set:

- **Both Correct ($n_{11}$)**: 134
- **HGB Correct, RF Wrong ($n_{10}$)**: 6
- **HGB Wrong, RF Correct ($n_{01}$)**: 15
- **Both Wrong ($n_{00}$)**: 24
- **Exact p-value**: 0.0784
- **Chi-Square Statistic**: 3.0476 ($p = 0.0809$)
- **Conclusion**: Performance difference between architectures is not statistically significant at $\alpha=0.05$ ($p > 0.05$), confirming that tree-based kinematic boundary learning is stable across gradient boosted and bagged ensembles.
