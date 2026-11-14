import os
import argparse
import pandas as pd
import joblib

def main():
    parser = argparse.ArgumentParser(description="UAV Attack Detection Inference System")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file for prediction")
    parser.add_argument("--output", type=str, default="prediction_output.csv", help="Output CSV path")
    args = parser.parse_args()

    print("=" * 60)
    print("UAV ATTACK DETECTION - INFERENCE SYSTEM")
    print("=" * 60)

    # 1. Configuration
    MODEL_PATH = os.path.join("models", "primary_gbm_model.pkl")
    
    REQUIRED_FEATURES = [
        "relative_alt", 
        "vel_m_s", "vel_n_m_s", "vel_e_m_s", "vel_d_m_s", "cog_rad"
    ]
    
    CLASS_MAPPING = {
        0: "Benign Flight",
        1: "GPS Jamming",
        2: "GPS Spoofing"
    }

    # 2. Check Input File
    if not os.path.exists(args.input):
        print(f"CRITICAL ERROR: Input file not found: {args.input}")
        return

    # 3. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: Trained model not found at {MODEL_PATH}. Please run the master pipeline first.")
        return
        
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[*] Successfully loaded primary Gradient Boosting model from {MODEL_PATH}")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model: {e}")
        return

    # 4. Load & Validate Data
    try:
        df = pd.read_csv(args.input)
        print(f"[*] Loaded {len(df)} records from {args.input}")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to read input CSV: {e}")
        return

    # Derive relative_alt if only alt is present
    if "relative_alt" not in df.columns and "alt" in df.columns:
        print("[*] Calculating 'relative_alt' from 'alt' (h_t - h_0) for spatial invariance...")
        df["relative_alt"] = df["alt"] - df["alt"].iloc[0]

    missing_features = [f for f in REQUIRED_FEATURES if f not in df.columns]
    if missing_features:
        print(f"\nCRITICAL ERROR: Missing required features in input data:\n  {missing_features}")
        print("\nThe model requires the following features:")
        print(f"  {REQUIRED_FEATURES}")
        return

    # Extract timestamp if available, else create dummy
    if "timestamp" in df.columns:
        timestamps = df["timestamp"]
    else:
        print("[!] Warning: 'timestamp' column not found. Using row indices as timestamps.")
        timestamps = df.index

    # 5. Extract Features
    X = df[REQUIRED_FEATURES]

    # 6. Predict
    print("[*] Running inference...")
    try:
        # Note: GBM model doesn't require explicit scaling because it's tree-based, 
        # and the master pipeline didn't use a scaler pipeline for the GBM.
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
    except Exception as e:
        print(f"CRITICAL ERROR: Inference failed: {e}")
        return

    # 7. Format Output
    results_df = pd.DataFrame({
        "timestamp": timestamps,
        "predicted_class_id": predictions,
        "predicted_class": [CLASS_MAPPING.get(p, "Unknown") for p in predictions],
        "prob_benign": probabilities[:, 0] if probabilities.shape[1] > 0 else 0,
        "prob_jamming": probabilities[:, 1] if probabilities.shape[1] > 1 else 0,
        "prob_spoofing": probabilities[:, 2] if probabilities.shape[1] > 2 else 0
    })

    # 8. Save
    try:
        results_df.to_csv(args.output, index=False)
        print(f"[*] Successfully saved predictions to {args.output}")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to save output CSV: {e}")
        return

    # 9. Summary
    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY")
    print("=" * 60)
    print(f"Total records processed : {len(results_df)}")
    
    counts = results_df["predicted_class"].value_counts()
    for cls in CLASS_MAPPING.values():
        count = counts.get(cls, 0)
        percentage = (count / len(results_df)) * 100
        print(f"  {cls:15s} : {count:5d} ({percentage:5.1f}%)")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

