import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import data_utils

def main():
    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------
    DATA = os.path.join("Prepared", "experiments", "gps_imu_mag_baro.csv")
    EDA_DIR = os.path.join("results", "figures", "eda")
    STATS_FILE = os.path.join("results", "eda_statistics.csv")
    os.makedirs(EDA_DIR, exist_ok=True)
    
    # Core GPS features for EDA
    GPS_FEATURES = [
        "lat", "lon", "alt",
        "vel_m_s", "vel_n_m_s", "vel_e_m_s", "vel_d_m_s", "cog_rad"
    ]
    
    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------
    df = pd.read_csv(DATA)
    # Sort chronologically
    df = df.sort_values(["attack_type", "timestamp"]).reset_index(drop=True)
    
    X_train, X_test, y_train, y_test, train_df, test_df = data_utils.load_and_split_data(DATA)
    
    # --------------------------------------------------------
    # 1. SUMMARY STATISTICS
    # --------------------------------------------------------
    stats_df = train_df[GPS_FEATURES].describe().transpose()
    stats_df.to_csv(STATS_FILE)
    print(f"Saved summary statistics to {STATS_FILE}")
    
    # --------------------------------------------------------
    # 2. CLASS DISTRIBUTION PLOT
    # --------------------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='attack_type', palette='viridis', order=["Benign Flight", "GPS Jamming", "GPS Spoofing"])
    plt.title("Overall Class Distribution")
    plt.ylabel("Number of Samples")
    plt.xlabel("Flight Type")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "class_distribution.png"), dpi=300)
    plt.close()
    
    # --------------------------------------------------------
    # 3. PCA 2D PLOT
    # --------------------------------------------------------
    # Fit scaler and PCA ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[GPS_FEATURES])
    X_test_scaled = scaler.transform(test_df[GPS_FEATURES])
    
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    plt.figure(figsize=(10, 7))
    colors = {"Benign Flight": "#2ecc71", "GPS Jamming": "#e74c3c", "GPS Spoofing": "#3498db"}
    
    # Plot train set (faded)
    for attack_type, color in colors.items():
        mask = train_df["attack_type"] == attack_type
        plt.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1], label=f"Train - {attack_type}", alpha=0.3, c=color, marker='o')
        
    # Plot test set (solid)
    for attack_type, color in colors.items():
        mask = test_df["attack_type"] == attack_type
        plt.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1], label=f"Test - {attack_type}", alpha=0.9, c=color, marker='X', s=100)
        
    plt.title("PCA of GPS Features (Fitted on Train Set Only)\nNote: PCA is for visualization, not required for the tree-based classifier")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "pca_2d.png"), dpi=300)
    plt.close()
    
    # --------------------------------------------------------
    # 4. CORRELATION MATRIX
    # --------------------------------------------------------
    plt.figure(figsize=(10, 8))
    corr = train_df[GPS_FEATURES].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title("Correlation Matrix of GPS Features (Train Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "correlation_matrix.png"), dpi=300)
    plt.close()
    
    # --------------------------------------------------------
    # 5. GPS FEATURE DISTRIBUTIONS BY CLASS
    # --------------------------------------------------------
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()
    
    for i, feature in enumerate(GPS_FEATURES):
        sns.boxplot(data=df, x="attack_type", y=feature, ax=axes[i], palette="viridis", order=["Benign Flight", "GPS Jamming", "GPS Spoofing"])
        axes[i].set_title(f"Distribution of {feature}")
        axes[i].set_xlabel("")
        axes[i].set_ylabel(feature)
        
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "feature_distributions.png"), dpi=300)
    plt.close()
    
    # --------------------------------------------------------
    # 6. GPS POSITION VISUALIZATION (Lat vs Lon)
    # --------------------------------------------------------
    plt.figure(figsize=(10, 8))
    for attack_type, color in colors.items():
        mask = df["attack_type"] == attack_type
        plt.plot(df.loc[mask, "lon"], df.loc[mask, "lat"], label=attack_type, color=color, alpha=0.7, linewidth=2, marker='.', markersize=5)
        
    plt.title("2D GPS Flight Path (Longitude vs Latitude)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "gps_position_path.png"), dpi=300)
    plt.close()
    
    # --------------------------------------------------------
    # 7. FEATURE IMPORTANCE (Baseline Random Forest)
    # --------------------------------------------------------
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_df[GPS_FEATURES], train_df["label"])
    
    imp_df = pd.DataFrame({
        "Feature": GPS_FEATURES,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(imp_df["Feature"], imp_df["Importance"], color='#2ecc71')
    plt.title("EDA Baseline Feature Importance (Random Forest)")
    plt.xlabel("Gini Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "eda_feature_importance.png"), dpi=300)
    plt.close()
    
    print(f"All EDA visualizations generated in {EDA_DIR}/")

if __name__ == "__main__":
    main()

