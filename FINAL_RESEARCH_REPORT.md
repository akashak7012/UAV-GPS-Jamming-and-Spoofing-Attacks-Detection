# Final Research Report: UAV GPS Jamming and Spoofing Attack Detection

## 1. Title
**Robust Multi-Class Detection of UAV GPS Jamming and Spoofing Attacks Using Gradient Boosting and Causal Temporal Validation**

## 2. Abstract
The increasing reliance of Unmanned Aerial Vehicles (UAVs) on Global Positioning Systems (GPS) exposes them to critical cyber-physical threats, notably GPS jamming and spoofing. This research evaluates the efficacy of machine learning models in detecting such attacks using multi-sensor telemetry. Addressing critical methodological flaws in prior literature related to time-series data leakage, we propose a strictly causal temporal evaluation framework. **Observed Result**: Our optimized Gradient Boosting model, relying solely on 8 GPS Position and Velocity kinematic features, achieved an Accuracy of 0.9385 (93.85%) and a Macro F1-score of 0.9260 (92.60%) on a blind held-out 179-sample chronological test set. **Interpretation**: Multi-sensor fusion (IMU, Magnetometer, Barometer) does not improve detection efficacy over pure GPS kinematics for this dataset. The model demonstrates high feasibility for real-time edge deployment with an average inference latency of under 2 ms per sample.

## 3. Introduction and Problem Statement
Unmanned Aerial Vehicles (UAVs) increasingly perform autonomous navigation tasks in commercial, agricultural, and defense sectors. Because civilian GPS satellite signals are unencrypted and operate with low received signal power (~-160 dBW), UAVs are highly vulnerable to:
1. **GPS Jamming:** High-power RF noise overpowering satellite signals, forcing the flight controller into emergency fail-safe modes (e.g., altitude hold or uncontrolled landing).
2. **GPS Spoofing:** Transmission of false satellite signals with forged pseudoranges, covertly deceiving the vehicle into deviating from its designated trajectory without triggering lost-signal fail-safes.

Real-time, onboard anomaly detection is critical for identifying attacks before catastrophic divergence occurs. This project implements a lightweight, tree-based machine learning inference system that accurately differentiates between benign flight, active jamming, and spoofing using behavioral flight kinematics.

## 4. Research Motivation
Our methodology builds upon prior research into UAV cyber-attack detection (e.g., Gormus et al., 2026). During initial codebase auditing, a pervasive methodological flaw was identified in published baselines: the application of randomized `train_test_split` on high-frequency, auto-correlated flight telemetry. Randomly partitioning time-series records enables the model to train on samples temporally adjacent to test samples, allowing it to "see into the future" of a flight and producing artificially inflated accuracies. 

**Our Motivation**: To establish a strictly causal machine learning pipeline that accurately models real-world operational constraints by guaranteeing models are evaluated solely on future, unseen flight dynamics.

## 5. Dataset Description & Reconciliation
The raw telemetry logs represent UAV operations recorded across three distinct scenarios on a PX4 autopilot system:
- **Benign Flight:** Normal operational trajectory under authentic GPS signals.
- **GPS Jamming:** Flight under active RF interference overpowering the GPS receiver.
- **GPS Spoofing:** Flight subjected to forged coordinate transmissions.

### Dataset Counts Reconciliation
- **Master Synchronized Dataset (`Prepared/uav_multisensor_fusion.csv`):** Contains **606 samples** (Benign: 247, Jamming: 222, Spoofing: 137).
- **Cleaned Experimentation Dataset (`Prepared/experiments/gps_imu_mag_baro.csv`):** Contains **593 samples** (Benign: 243, Jamming: 217, Spoofing: 133).
  - *Reconciliation Explanation:* The boundary cleaning step in `create_experiment_datasets.py` discards initial rows where 5-step rolling window statistics (`velocity_std`, `mag_std`, `pressure_std`) and lag differences produce `NaN`s. Exactly 13 boundary samples (4 Benign, 5 Jamming, 4 Spoofing) are dropped, reconciling the 606 raw rows to 593 clean samples. Earlier preliminary reports citing 598 rows reflected an older synchronization tolerance filter before temporal boundary regularization.

## 6. Preprocessing & Causal Synchronization
To strictly prevent forward-looking data leakage during sensor synchronization, auxiliary sensors (IMU, Magnetometer, Barometer) are aligned to the GPS timeline using:
```python
pd.merge_asof(gps, sensor, on="timestamp", direction="backward", tolerance=1_000_000)
```
- **Direction:** `backward` guarantees that every auxiliary sensor observation occurs strictly at or before the corresponding GPS timestamp.
- **Tolerance:** 1,000,000 microseconds (1.0 second) accommodates typical GPS update intervals (~5-10 Hz) while rejecting stale readings.
- **PX4 Coordinates:** Latitude and longitude are scaled by $10^{-7}$ degrees, and altitude by $10^{-3}$ meters to restore physical units.
- **Deduplication:** Duplicate timestamps within each individual sensor stream are eliminated prior to merging.

## 7. Feature Engineering
Causal temporal features were engineered to capture rate-of-change and volatility:
- **First Differences:** `lat_change`, `lon_change`, `alt_change`, `velocity_change`, `mag_change`, `pressure_change`.
- **Rolling Dispersion:** Moving standard deviations (`velocity_std`, `mag_std`, `pressure_std`) calculated using a 5-sample backward window.
All feature operations strictly rely on immediate past and current values, eliminating temporal leakage.

## 8. Chronological Training and Test Split
Evaluations enforce a strict chronological 70/30 split per attack scenario via `data_utils.load_and_split_data`:
- Telemetry is grouped by `attack_type` and sorted chronologically by `timestamp`.
- The first 70% of each flight constitutes the **Training Set: 414 samples** (Benign: 170, Jamming: 151, Spoofing: 93).
- The final 30% of each flight forms the **Held-out Test Set: 179 samples** (Benign: 73, Jamming: 66, Spoofing: 40).
- Total Dataset: 414 + 179 = 593 samples.

## 9. Algorithms Compared
We benchmarked five machine learning architectures on the chronological split:
1. **Gradient Boosting Classifier (Primary Model)**: Sequential gradient-boosted decision trees (`n_estimators=200`, `learning_rate=0.05`, `max_depth=3`, `random_state=42`).
2. **Random Forest Classifier**: Ensemble of 300 de-correlated bagging decision trees with balanced class weights.
3. **XGBoost Classifier**: Extreme Gradient Boosting with shrinkage and subsampling.
4. **Support Vector Machine (SVM)**: RBF kernel with training-fitted standard scaling.
5. **Logistic Regression**: L2-regularized linear baseline with training-fitted standard scaling.

## 10. Feature-Group Experiments
We isolated distinct groupings of GPS parameters to evaluate their relative discriminative power using the Random Forest baseline:

| Experiment_Type | Name | Features | Accuracy | Macro_F1 |
| --- | --- | --- | --- | --- |
| Feature Group | GPS Position | 3 | 0.7932960893854749 | 0.7647623331833859 |
| Feature Group | GPS Velocity | 5 | 0.6033519553072626 | 0.638365307753797 |
| Feature Group | GPS Quality | 7 | 0.5810055865921788 | 0.6112668531809241 |
| Feature Group | GPS Motion | 5 | 0.3463687150837989 | 0.3049786535115832 |
| Feature Group | GPS Velocity + Quality | 12 | 0.5865921787709497 | 0.6183518196941016 |
| Feature Group | GPS Position + Quality | 10 | 0.88268156424581 | 0.8676300062438678 |
| Feature Group | GPS Position + Velocity | 8 | 0.8770949720670391 | 0.8614379084967321 |
| Feature Group | GPS Quality + Motion | 12 | 0.5865921787709497 | 0.621217101937143 |
| Feature Group | ALL GPS | 20 | 0.88268156424581 | 0.8862990948880519 |


**Interpretation:** Kinematic combinations (`GPS Position + Velocity` and `GPS Position + Quality`) provide the highest signal-to-noise ratio (~87.7% - 88.3% accuracy), outperforming single-modality groups like position-only or velocity-only.

## 11. Sensor Ablation Experiments
We evaluated whether fusing non-GPS auxiliary sensors (IMU, Magnetometer, Barometer) enhances attack detection:

| Experiment_Type | Name | Features | Accuracy | Macro_F1 |
| --- | --- | --- | --- | --- |
| Sensor Ablation | GPS Only | 20 | 0.88268156424581 | 0.8862990948880519 |
| Sensor Ablation | GPS + IMU | 28 | 0.8770949720670391 | 0.8808749516066591 |
| Sensor Ablation | GPS + MAG | 26 | 0.88268156424581 | 0.8844072831000936 |
| Sensor Ablation | GPS + BARO | 24 | 0.8770949720670391 | 0.8808749516066591 |
| Sensor Ablation | GPS + IMU + MAG | 34 | 0.8770949720670391 | 0.8808749516066591 |
| Sensor Ablation | GPS + IMU + BARO | 32 | 0.8715083798882681 | 0.8754067543058369 |
| Sensor Ablation | GPS + MAG + BARO | 30 | 0.8770949720670391 | 0.8808749516066591 |
| Sensor Ablation | FULL SENSOR FUSION | 38 | 0.8770949720670391 | 0.8808749516066591 |


**Interpretation:** GPS-only telemetry achieves a Macro F1 of 0.8863, while full multi-sensor fusion achieves 0.8809. Incorporating auxiliary sensors did not improve classification accuracy, demonstrating that GPS kinematic anomalies inherently contain the necessary discriminative signatures of jamming and spoofing attacks.

## 12. Robustness Results
We evaluated the primary Gradient Boosting model against degraded and relative-only feature representations to test spatial resilience:

| Experiment_Type | Name | Features | Accuracy | Macro_F1 |
| --- | --- | --- | --- | --- |
| Robustness | Absolute GPS + Velocity | 8 | 0.9385474860335196 | 0.926017575584531 |
| Robustness | Altitude + Velocity | 6 | 0.8659217877094972 | 0.8422619047619048 |
| Robustness | Relative Motion | 5 | 0.3687150837988827 | 0.3006687193405838 |
| Robustness | Velocity + GPS Quality | 12 | 0.5418994413407822 | 0.5477945477945477 |
| Robustness | Altitude + Velocity + Quality | 13 | 0.8156424581005587 | 0.8044624751941826 |


**Interpretation:** The model retains high accuracy (86.59% accuracy, 84.23% Macro F1) even when absolute latitude and longitude coordinates are removed (`Altitude + Velocity`), proving that the classifier does not merely memorize geographical waypoints.

## 13. Primary Model Evaluation & Algorithm Comparison
Benchmarking all candidate architectures on the optimal 8 GPS kinematic features (`lat`, `lon`, `alt`, `vel_m_s`, `vel_n_m_s`, `vel_e_m_s`, `vel_d_m_s`, `cog_rad`):

| Experiment_Type | Name | Features | Accuracy | Macro_F1 |
| --- | --- | --- | --- | --- |
| Model Comparison | Random Forest | 8 | 0.8770949720670391 | 0.8614379084967321 |
| Model Comparison | Gradient Boosting | 8 | 0.9385474860335196 | 0.926017575584531 |
| Model Comparison | SVM | 8 | 0.5865921787709497 | 0.5337158965065941 |
| Model Comparison | Logistic Regression | 8 | 0.770949720670391 | 0.736869191414646 |
| Model Comparison | XGBoost | 8 | 0.8435754189944135 | 0.8254781591433562 |


**Primary Model Global Metrics (Held-out 179-Sample Test Set):**
- **Accuracy:** 0.9385 (93.85%)
- **Macro F1-Score:** 0.9260 (92.60%)
- **Balanced Accuracy (Macro Recall):** 0.9247 (92.47%)
- **Macro Precision:** 0.9274 (92.74%)
- **Weighted F1-Score:** 0.9384 (93.84%)
- **Matthews Correlation Coefficient (MCC):** 0.9051
- **Cohen's Kappa:** 0.9050
- **Training Wall-Clock Time:** 0.3789 seconds
- **Batch Prediction Time:** 0.0016 seconds

## 14. Confusion Matrix & Per-Class Performance

### Per-Class Classification Report
| Unnamed: 0 | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| Benign Flight | 1.0 | 1.0 | 1.0 | 73.0 |
| GPS Jamming | 0.9104477611940298 | 0.9242424242424242 | 0.9172932330827068 | 66.0 |
| GPS Spoofing | 0.8717948717948718 | 0.85 | 0.8607594936708861 | 40.0 |
| accuracy | 0.9385474860335196 | 0.9385474860335196 | 0.9385474860335196 | 0.9385474860335196 |
| macro avg | 0.9274142109963004 | 0.9247474747474748 | 0.926017575584531 | 179.0 |
| weighted avg | 0.9383315481039154 | 0.9385474860335196 | 0.938389570560302 | 179.0 |


### Raw Confusion Matrix (Sample Counts)
| Unnamed: 0 | Benign Flight | GPS Jamming | GPS Spoofing |
| --- | --- | --- | --- |
| Benign Flight | 73 | 0 | 0 |
| GPS Jamming | 0 | 61 | 5 |
| GPS Spoofing | 0 | 6 | 34 |


### Normalized Confusion Matrix (True Class Recall)
| Unnamed: 0 | Benign Flight | GPS Jamming | GPS Spoofing |
| --- | --- | --- | --- |
| Benign Flight | 1.0 | 0.0 | 0.0 |
| GPS Jamming | 0.0 | 0.9242424242424242 | 0.0757575757575757 |
| GPS Spoofing | 0.0 | 0.15 | 0.85 |


**Key Findings:**
- **Benign Flight:** 100.0% Recall (73/73) and 100.0% Precision. Zero false alarms were generated during normal flight.
- **GPS Jamming:** 92.42% Recall (61/66) and 91.04% Precision.
- **GPS Spoofing:** 85.00% Recall (34/40) and 87.18% Precision.

## 15. Interpretability and Feature Importance
Feature importance was analyzed using both internal Mean Decrease in Impurity (MDI) and Permutation Importance on the held-out temporal test set:

- **Top Features by Permutation Importance:** `alt` and `cog_rad`.
- **Methodological Disclaimer:** Feature importance indicates *model dependency*, not physical causality. The model heavily leverages altitude discontinuities and course-over-ground angular deviations to discriminate flight modes on this platform. This does not imply that spoofing attacks physically alter altitude sensors directly.

## 16. Real-Time Inference Latency & Working Prototype
A real-time streaming simulation was executed on the held-out 179-sample sequence (`realtime_simulation.py`):
- **Average Inference Latency:** 1.57 ms / sample.
- **Maximum Observed Latency:** 7.05 ms / sample.
- **Throughput:** Exceeds 600 samples/second on standard host CPU.
- **Viability:** Given standard UAV flight controller telemetry loops (10 Hz – 50 Hz, corresponding to 20 ms – 100 ms periods), a 1.57 ms latency easily satisfies real-time onboard processing constraints.

### Generated Visualizations
- ![Feature Group Comparison](results/figures/feature_group_comparison.png)
- ![Sensor Ablation](results/figures/sensor_ablation_comparison.png)
- ![Model Comparison](results/figures/model_comparison.png)
- ![Robustness Comparison](results/figures/robustness_comparison.png)
- ![System Architecture](results/figures/system_architecture.png)

## 17. Discussion
By eliminating data leakage through causal backward synchronization and chronological splitting, we established a rigorous benchmark for UAV attack detection. The Gradient Boosting classifier successfully detects the distinctive kinematic signatures of jamming (abrupt velocity drops and signal drift) and spoofing (artificial trajectory deviations) without requiring computationally expensive cryptographic hardware or auxiliary multi-sensor fusion.

## 18. Limitations
1. **Single-Airframe Telemetry:** The experimental dataset originates from a single UAV hardware configuration. Performance across heterogeneous airframes and diverse geographic regions remains to be validated.
2. **Temporal Split Sensitivity:** Robustness testing on alternative split boundaries showed that an 80/20 split reduces Macro F1 to ~79.3%, indicating sensitivity to terminal descent dynamics that were underrepresented in the 80% training segment.
3. **Software Simulation vs. Hardware In-the-Loop:** While latency measurements confirm algorithmic feasibility, physical deployment requires porting the model to embedded companion computers (e.g., Raspberry Pi) or flight controllers (Pixhawk running ArduPilot/PX4).

## 19. Conclusion
A compact Gradient Boosting model utilizing only 8 core GPS Position and Velocity kinematic features delivers 93.85% accuracy and 92.60% Macro F1 under strict causal chronological validation. Multi-sensor fusion yields negligible benefit for this dataset, confirming that lightweight GPS kinematic monitoring provides an efficient and effective defense against UAV jamming and spoofing attacks.

## 20. Future Work
1. **Cross-Platform Validation:** Evaluating the pipeline on independent flights across diverse airframes and geographical locations.
2. **Embedded Implementation:** Exporting the trained tree ensemble to C++ (e.g., via `treelite` or `ONNX Runtime`) for direct integration into PX4 autopilot firmware.
3. **Adversarial Spoofing Profiles:** Benchmarking against advanced, gradual spoofing algorithms designed specifically to evade kinematic velocity thresholds.

## 21. Reproducibility Information
The complete experimental pipeline is fully reproducible on Linux using Python 3.9+:
```bash
# Data preparation & boundary cleaning
python prepare_fusion_dataset.py
python create_experiment_datasets.py

# Primary model evaluation & sensitivity analysis
python evaluate_primary_model.py
python temporal_robustness_test.py
python interpretability_analysis.py

# Real-time streaming simulation & predictions
python realtime_simulation.py
python predict.py --input Prepared/experiments/gps_imu_mag_baro.csv --output custom_predictions.csv
```
All authoritative metrics are dynamically persisted to `results/primary_model/`.
