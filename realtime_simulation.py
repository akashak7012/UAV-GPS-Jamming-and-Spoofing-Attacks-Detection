import os
import sys
import time
import pandas as pd
import numpy as np
import joblib

import data_utils

def main():
    print("=" * 70)
    print("UAV ATTACK DETECTION - REAL-TIME SIMULATION")
    print("=" * 70)
    
    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------
    DATA_PATH = os.path.join("Prepared", "experiments", "gps_imu_mag_baro.csv")
    MODEL_PATH = os.path.join("models", "primary_gbm_model.pkl")
    OUTPUT_CSV = os.path.join("results", "realtime_simulation.csv")
    
    FEATURES = [
        "relative_alt",
        "vel_m_s", "vel_n_m_s", "vel_e_m_s", "vel_d_m_s", "cog_rad"
    ]
    
    CLASS_MAPPING = {
        0: "Benign Flight",
        1: "GPS Jamming",
        2: "GPS Spoofing"
    }

    # --------------------------------------------------------
    # LOAD RESOURCES
    # --------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)
        
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[*] Loaded trained model from {MODEL_PATH}")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model: {e}")
        sys.exit(1)
        
    print("[*] Loading and splitting data to isolate Temporal Test Set...")
    try:
        # We only need the test dataframe to simulate inference
        _, _, _, _, _, test_df = data_utils.load_and_split_data(DATA_PATH)
        
        # Sort chronologically to simulate sequential streaming
        test_df = test_df.sort_values("timestamp").reset_index(drop=True)
        print(f"[*] Isolated {len(test_df)} temporal test samples for simulation.\n")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load dataset: {e}")
        sys.exit(1)

    # --------------------------------------------------------
    # SIMULATION VARIABLES
    # --------------------------------------------------------
    total_samples = len(test_df)
    correct_predictions = 0
    inference_times = []
    
    # Class-wise stats
    class_stats = {
        cls: {"total": 0, "correct": 0, "predicted": 0} 
        for cls in CLASS_MAPPING.values()
    }
    
    results = []
    
    print("-" * 70)
    print("STARTING REAL-TIME TELEMETRY STREAM")
    print("-" * 70)
    
    # --------------------------------------------------------
    # REAL-TIME LOOP
    # --------------------------------------------------------
    for idx, row in test_df.iterrows():
        timestamp = row.get("timestamp", idx)
        true_label_id = row["label"]
        true_class = CLASS_MAPPING.get(true_label_id, "Unknown")
        
        # 1. Prepare single sequential sample (reshape to 2D array)
        X_stream = pd.DataFrame([row[FEATURES]])
        
        # 2. Inference
        t0 = time.perf_counter()
        pred_id = model.predict(X_stream)[0]
        probs = model.predict_proba(X_stream)[0]
        t1 = time.perf_counter()
        
        inf_time_ms = (t1 - t0) * 1000
        inference_times.append(inf_time_ms)
        
        pred_class = CLASS_MAPPING.get(pred_id, "Unknown")
        
        # 3. Track Statistics
        is_correct = (pred_id == true_label_id)
        if is_correct:
            correct_predictions += 1
            class_stats[true_class]["correct"] += 1
            
        class_stats[true_class]["total"] += 1
        class_stats[pred_class]["predicted"] += 1
        
        running_acc = (correct_predictions / (idx + 1)) * 100
        
        # 4. Print Status
        prob_str = f"B: {probs[0]:.2f} | J: {probs[1]:.2f} | S: {probs[2]:.2f}"
        print(f"[{timestamp}] Pred: {pred_class:15s} | True: {true_class:15s} | {prob_str} | Run Acc: {running_acc:5.1f}% | Latency: {inf_time_ms:5.2f}ms")
        
        # 5. Store for CSV
        results.append({
            "timestamp": timestamp,
            "true_class": true_class,
            "predicted_class": pred_class,
            "prob_benign": probs[0],
            "prob_jamming": probs[1],
            "prob_spoofing": probs[2],
            "inference_time_ms": inf_time_ms,
            "is_correct": is_correct
        })

    # --------------------------------------------------------
    # SAVE & REPORT
    # --------------------------------------------------------
    print("-" * 70)
    print("END OF STREAM")
    print("-" * 70)
    
    # Save CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    
    # Final Metrics
    acc = (correct_predictions / total_samples) * 100
    avg_inf = np.mean(inference_times)
    max_inf = np.max(inference_times)
    
    print("\n" + "=" * 70)
    print("SIMULATION FINAL REPORT")
    print("=" * 70)
    print(f"Total Samples Processed : {total_samples}")
    print(f"Correct Predictions     : {correct_predictions}")
    print(f"Overall Accuracy        : {acc:.2f}%")
    print(f"Average Inference Time  : {avg_inf:.3f} ms / sample")
    print(f"Maximum Inference Time  : {max_inf:.3f} ms / sample")
    
    print("\nClass-wise Results:")
    print(f"{'Class':<15} | {'Total':<6} | {'Correct':<7} | {'Recall':<7}")
    print("-" * 43)
    for cls, stats in class_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        recall = (correct / total * 100) if total > 0 else 0
        print(f"{cls:<15} | {total:<6} | {correct:<7} | {recall:.1f}%")
        
    print("\n" + "=" * 70)
    print(f"Detailed timeline saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

