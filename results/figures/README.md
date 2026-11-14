# Publication Figures

This directory contains publication-ready bar charts for the primary experiments evaluated on the temporal 70/30 split.

1. **`feature_group_comparison.png`**: Highlights the performance of isolating specific combinations of GPS features (Position, Velocity, Quality, Motion). Evaluated using Random Forest.
2. **`sensor_ablation_comparison.png`**: Demonstrates the impact of adding or removing specific non-GPS sensor streams (IMU, MAG, BARO). Evaluated using Random Forest.
3. **`model_comparison.png`**: Compares different machine learning architectures (Random Forest, Gradient Boosting, SVM, Logistic Regression, XGBoost) applied specifically to the optimal GPS Position + Velocity feature set.
4. **`robustness_comparison.png`**: Tests the winning Gradient Boosting model against specific challenging subsets of features (e.g., removing absolute latitude/longitude, or relying purely on relative motion/quality) to determine robustness.

### Methodology Notes:
*   All charts report both **Accuracy** and **Macro F1**.
*   The Y-axis is strictly fixed from 0.0 to 1.0 to prevent misleading visual scaling.
*   Values are evaluated strictly on the blind chronological temporal test sequence to prevent time-series data leakage.
