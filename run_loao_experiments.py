import os
import sys
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, average_precision_score,
    confusion_matrix
)

# Output directories
csv_dir = os.path.join("results", "loao")
os.makedirs(csv_dir, exist_ok=True)

# Load dataset
dataset_path = os.path.join("Prepared", "experiments", "full_joined_dataset.csv")
if not os.path.exists(dataset_path):
    dataset_path = os.path.join(csv_dir, "full_joined_dataset.csv")
df = pd.read_csv(dataset_path)

# Feature subsets
feature_sets = {
    "kinematics_only": ["vel_m_s", "vel_n_m_s", "vel_e_m_s", "vel_d_m_s", "cog_rad"],
    "pos_speed_only": ["pos_speed"],
    "discrepancy_rolling_only": ["kinematic_discrepancy", "roll_mean_5", "roll_std_5"],
    "complete_features": ["vel_m_s", "vel_n_m_s", "vel_e_m_s", "vel_d_m_s", "cog_rad", "pos_speed", "kinematic_discrepancy", "roll_mean_5", "roll_std_5"]
}

# Models
def get_models():
    return {
        "GradientBoosting": HistGradientBoostingClassifier(random_state=42, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    }

airframes = ["PLANE", "QUAD", "TAIL", "VTOL"]

# Episode confirmation helper (M=3 consecutive positive predictions)
def evaluate_episodes(test_df, y_pred, y_prob):
    """
    Evaluates episode-level metrics with strict pre-attack isolation.
    """
    eval_df = test_df.copy()
    eval_df["y_pred"] = y_pred
    eval_df["y_prob"] = y_prob
    
    episode_results = []
    
    # Process each recording in test set
    for rec_id, rec_grp in eval_df.groupby("recording_id"):
        af = rec_grp["airframe"].iloc[0]
        scn = rec_grp["scenario"].iloc[0]
        rec_grp = rec_grp.sort_values("timestamp").reset_index(drop=True)
        
        pos_mask = (rec_grp["label"] == 1)
        has_attack = pos_mask.any()
        
        # Debounced False Alarm Events helper
        def count_fa_events(sub_df):
            if len(sub_df) < 3:
                return 0
            preds = sub_df["y_pred"].to_numpy()
            events = 0
            in_alarm = False
            zero_count = 0
            for i in range(2, len(preds)):
                if not in_alarm:
                    if preds[i] == 1 and preds[i-1] == 1 and preds[i-2] == 1:
                        events += 1
                        in_alarm = True
                        zero_count = 0
                else:
                    if preds[i] == 0:
                        zero_count += 1
                        if zero_count >= 5:
                            in_alarm = False
                    else:
                        zero_count = 0
            return events

        if not has_attack:
            # Entire flight is authentic
            fa_events = count_fa_events(rec_grp)
            dur_s = (rec_grp["timestamp"].max() - rec_grp["timestamp"].min()) / 1e6
            episode_results.append({
                "recording_id": rec_id,
                "airframe": af,
                "scenario": scn,
                "is_attack_flight": False,
                "detected": np.nan,
                "t_onset_us": np.nan,
                "t_confirm_us": np.nan,
                "ttd_s": np.nan,
                "pre_attack_fa_events": 0,
                "termination_alarm": False,
                "post_attack_fa_events": 0,
                "total_fa_events": fa_events,
                "authentic_dur_s": dur_s
            })
        else:
            # Flight contains attack
            t_onset = rec_grp.loc[pos_mask, "timestamp"].min()
            t_term = rec_grp.loc[pos_mask, "timestamp"].max()
            
            pre_atk = rec_grp[rec_grp["timestamp"] < t_onset].copy()
            in_atk = rec_grp[(rec_grp["timestamp"] >= t_onset) & (rec_grp["timestamp"] <= t_term)].copy()
            post_atk = rec_grp[rec_grp["timestamp"] > t_term].copy()
            
            # 1. Pre-attack FA events
            pre_fa = count_fa_events(pre_atk)
            
            # 2. In-attack Episode Confirmation (M=3 consecutive positive predictions)
            detected = False
            t_confirm = np.nan
            ttd_s = np.nan
            
            in_preds = in_atk["y_pred"].to_numpy()
            in_ts = in_atk["timestamp"].to_numpy()
            
            if len(in_preds) >= 3:
                for idx in range(2, len(in_preds)):
                    if in_preds[idx] == 1 and in_preds[idx-1] == 1 and in_preds[idx-2] == 1:
                        detected = True
                        t_confirm = in_ts[idx]
                        ttd_s = (t_confirm - t_onset) / 1e6
                        break
            
            # 3. Termination Alarm Check (first 2.0s after t_term)
            term_window = post_atk[post_atk["timestamp"] <= t_term + 2_000_000]
            term_alarm = (term_window["y_pred"] == 1).any() if len(term_window) > 0 else False
            
            # 4. Post-attack FA events
            post_fa = count_fa_events(post_atk)
            
            pre_dur_s = (pre_atk["timestamp"].max() - pre_atk["timestamp"].min()) / 1e6 if len(pre_atk) > 1 else 0.0
            post_dur_s = (post_atk["timestamp"].max() - post_atk["timestamp"].min()) / 1e6 if len(post_atk) > 1 else 0.0
            auth_dur_s = pre_dur_s + post_dur_s
            
            episode_results.append({
                "recording_id": rec_id,
                "airframe": af,
                "scenario": scn,
                "is_attack_flight": True,
                "detected": detected,
                "t_onset_us": t_onset,
                "t_confirm_us": t_confirm,
                "ttd_s": ttd_s,
                "pre_attack_fa_events": pre_fa,
                "termination_alarm": term_alarm,
                "post_attack_fa_events": post_fa,
                "total_fa_events": pre_fa + post_fa,
                "authentic_dur_s": auth_dur_s
            })
            
    return episode_results

# MASTER EXPERIMENT RUNNER
sample_results = []
episode_all_records = []
all_predictions = []

print("=== STARTING LOAO CROSS-AIRFRAME EXPERIMENTS ===")

for feat_name, f_cols in feature_sets.items():
    print(f"\n>>> Running Feature Set: {feat_name} ({len(f_cols)} features) <<<")
    
    models = get_models()
    for model_name, model in models.items():
        # Only run all models for complete_features; for ablations, run GradientBoosting
        if feat_name != "complete_features" and model_name != "GradientBoosting":
            continue
            
        print(f"  --> Model: {model_name}")
        fold_metrics = []
        
        for fold_id, af in enumerate(airframes, start=1):
            train_mask = (df["airframe"] != af)
            test_mask = (df["airframe"] == af)
            
            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()
            
            X_train = train_df[f_cols].to_numpy()
            y_train = train_df["label"].to_numpy()
            
            X_test = test_df[f_cols].to_numpy()
            y_test = test_df["label"].to_numpy()
            
            # Scaler fit on train only
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Fit model
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test_scaled)
            else:
                y_prob = y_pred.astype(float)
                
            # Sample-level metrics
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            pr_auc = average_precision_score(y_test, y_prob)
            
            # Save predictions for record
            pred_record = pd.DataFrame({
                "feature_set": feat_name,
                "model_name": model_name,
                "fold_id": fold_id,
                "held_out_airframe": af,
                "recording_id": test_df["recording_id"],
                "timestamp": test_df["timestamp"],
                "y_true": y_test,
                "y_pred": y_pred,
                "y_prob": y_prob
            })
            all_predictions.append(pred_record)
            
            # Episode-level evaluation
            ep_eval = evaluate_episodes(test_df, y_pred, y_prob)
            for ep in ep_eval:
                ep["feature_set"] = feat_name
                ep["model_name"] = model_name
                ep["fold_id"] = fold_id
                episode_all_records.append(ep)
                
            sample_results.append({
                "feature_set": feat_name,
                "model_name": model_name,
                "fold_id": fold_id,
                "held_out_airframe": af,
                "test_total": len(y_test),
                "test_pos": int(y_test.sum()),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "balanced_acc": bal_acc,
                "pr_auc": pr_auc
            })
            
            print(f"    Fold {fold_id} ({af:5s}): Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, BalAcc={bal_acc:.4f}, PR-AUC={pr_auc:.4f}")

# Convert to DataFrames
sample_results_df = pd.DataFrame(sample_results)
episode_df = pd.DataFrame(episode_all_records)
all_preds_df = pd.concat(all_predictions, ignore_index=True)

# Save machine-readable CSVs
sample_csv_path = os.path.join(csv_dir, "sample_level_results.csv")
sample_results_df.to_csv(sample_csv_path, index=False)
print("\nSaved sample-level results:", sample_csv_path)

episode_csv_path = os.path.join(csv_dir, "episode_level_results.csv")
episode_df.to_csv(episode_csv_path, index=False)
print("Saved episode-level results:", episode_csv_path)

preds_csv_path = os.path.join(csv_dir, "all_model_predictions.csv")
all_preds_df.to_csv(preds_csv_path, index=False)
print("Saved all model predictions:", preds_csv_path)

print("\n=== EXPERIMENT EXECUTION COMPLETED SUCCESSFULLY ===")
