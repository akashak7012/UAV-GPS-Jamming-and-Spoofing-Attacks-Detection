import pandas as pd
import os


INPUT = os.path.join("Prepared", "uav_multisensor_fusion.csv")
OUTPUT_DIR = os.path.join("Prepared", "experiments")

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT)

# ------------------------------------------------------------
# Spatial Invariance: Calculate relative_alt (h_t - h_0)
# Altitude relative to the first sample of that specific recording/flight
# ------------------------------------------------------------
df["relative_alt"] = df.groupby("attack_type")["alt"].transform(lambda x: x - x.iloc[0])

# ------------------------------------------------------------
# Remove rows where temporal features cannot be calculated.
# These are only the initial rows of each flight.
# ------------------------------------------------------------

temporal_features = [
    "lat_change",
    "lon_change",
    "alt_change",
    "velocity_change",
    "mag_change",
    "velocity_std",
    "mag_std",
    "pressure_std"
]

df = df.dropna(subset=temporal_features).reset_index(drop=True)

print("Rows after removing temporal-feature NaNs:", len(df))


# ============================================================
# COMMON GPS FEATURES
# ============================================================

gps_features = [
    "timestamp",

    # Coordinate-Free Spatial Invariance
    "relative_alt",

    # Navigation
    "lat",
    "lon",
    "alt",
    "vel_m_s",
    "vel_n_m_s",
    "vel_e_m_s",
    "vel_d_m_s",
    "cog_rad",

    # GPS quality
    "eph",
    "epv",
    "hdop",
    "vdop",
    "noise_per_ms",
    "vel_ned_valid",
    "satellites_used",

    # Temporal GPS features
    "lat_change",
    "lon_change",
    "alt_change",
    "velocity_change",
    "velocity_std"
]


# ============================================================
# IMU FEATURES
# ============================================================

imu_features = [
    "imu_angle_x",
    "imu_angle_y",
    "imu_angle_z",

    "imu_velocity_x",
    "imu_velocity_y",
    "imu_velocity_z",

    "imu_angle_mag",
    "imu_velocity_mag"
]


# ============================================================
# MAGNETOMETER FEATURES
# ============================================================

mag_features = [
    "mag_x",
    "mag_y",
    "mag_z",
    "mag_magnitude",
    "mag_change",
    "mag_std"
]


# ============================================================
# BAROMETER FEATURES
# ============================================================

baro_features = [
    "pressure",
    "temperature",
    "pressure_change",
    "pressure_std"
]


# ============================================================
# TARGET
# ============================================================

target = [
    "label",
    "attack_type"
]


# ============================================================
# DATASET 1 — GPS ONLY
# ============================================================

gps_df = df[gps_features + target].copy()

gps_path = os.path.join(
    OUTPUT_DIR,
    "gps_only.csv"
)

gps_df.to_csv(
    gps_path,
    index=False
)

print("\nGPS-only:")
print(gps_df.shape)
print(gps_df["attack_type"].value_counts())


# ============================================================
# DATASET 2 — GPS + IMU
# ============================================================

gps_imu_df = df[
    gps_features +
    imu_features +
    target
].copy()

gps_imu_path = os.path.join(
    OUTPUT_DIR,
    "gps_imu.csv"
)

gps_imu_df.to_csv(
    gps_imu_path,
    index=False
)

print("\nGPS + IMU:")
print(gps_imu_df.shape)
print(gps_imu_df["attack_type"].value_counts())


# ============================================================
# DATASET 3 — FULL MULTI-SENSOR
# ============================================================

full_df = df[
    gps_features +
    imu_features +
    mag_features +
    baro_features +
    target
].copy()

full_path = os.path.join(
    OUTPUT_DIR,
    "gps_imu_mag_baro.csv"
)

full_df.to_csv(
    full_path,
    index=False
)

print("\nFULL MULTI-SENSOR:")
print(full_df.shape)
print(full_df["attack_type"].value_counts())


print("\nSaved datasets:")
print(gps_path)
print(gps_imu_path)
print(full_path)