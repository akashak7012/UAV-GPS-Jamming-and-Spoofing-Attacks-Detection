import pandas as pd
import numpy as np
import os
import glob


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = r"Live GPS Spoofing and Jamming"

SCENARIOS = {
    "Benign Flight": 0,
    "GPS Jamming": 1,
    "GPS Spoofing": 2
}

OUTPUT_DIR = "Prepared"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Maximum allowed timestamp difference for sensor matching.
# timestamps are in microseconds.
TOLERANCE_US = 1_000_000


# ============================================================
# FIND FILE
# ============================================================

def find_file(folder, pattern):
    files = glob.glob(os.path.join(folder, pattern))

    if not files:
        raise FileNotFoundError(
            f"Could not find {pattern} in {folder}"
        )

    # Ignore processed files if necessary
    files = [
        f for f in files
        if "Processed" not in f
    ]

    if not files:
        raise FileNotFoundError(
            f"No raw file found for {pattern}"
        )

    return files[0]


# ============================================================
# GPS PREPROCESSING
# ============================================================

def prepare_gps(path):

    df = pd.read_csv(path)

    # Keep only features that we have decided are safe
    # for the primary experiment.
    columns = [
        "timestamp",
        "lat",
        "lon",
        "alt",
        "vel_m_s",
        "vel_n_m_s",
        "vel_e_m_s",
        "vel_d_m_s",
        "eph",
        "epv",
        "hdop",
        "vdop",
        "noise_per_ms",
        "cog_rad",
        "vel_ned_valid",
        "satellites_used"
    ]

    df = df[columns].copy()

    # --------------------------------------------------------
    # PX4 GPS scaling
    #
    # lat/lon are stored as integer scaled coordinates.
    # PX4 vehicle_gps_position uses 1e-7 degrees.
    #
    # altitude fields are in millimeters.
    # --------------------------------------------------------

    df["lat"] = df["lat"] * 1e-7
    df["lon"] = df["lon"] * 1e-7
    df["alt"] = df["alt"] * 1e-3

    # Remove duplicate timestamps
    df = (
        df.sort_values("timestamp")
          .drop_duplicates("timestamp")
          .reset_index(drop=True)
    )

    return df


# ============================================================
# IMU PREPROCESSING
# ============================================================

def prepare_imu(path):

    df = pd.read_csv(path)

    columns = [
        "timestamp",
        "delta_angle[0]",
        "delta_angle[1]",
        "delta_angle[2]",
        "delta_velocity[0]",
        "delta_velocity[1]",
        "delta_velocity[2]"
    ]

    df = df[columns].copy()

    df = (
        df.sort_values("timestamp")
          .drop_duplicates("timestamp")
          .reset_index(drop=True)
    )

    # Rename columns
    df = df.rename(columns={
        "delta_angle[0]": "imu_angle_x",
        "delta_angle[1]": "imu_angle_y",
        "delta_angle[2]": "imu_angle_z",

        "delta_velocity[0]": "imu_velocity_x",
        "delta_velocity[1]": "imu_velocity_y",
        "delta_velocity[2]": "imu_velocity_z"
    })

    # Derived magnitudes
    df["imu_angle_mag"] = np.sqrt(
        df["imu_angle_x"] ** 2 +
        df["imu_angle_y"] ** 2 +
        df["imu_angle_z"] ** 2
    )

    df["imu_velocity_mag"] = np.sqrt(
        df["imu_velocity_x"] ** 2 +
        df["imu_velocity_y"] ** 2 +
        df["imu_velocity_z"] ** 2
    )

    return df


# ============================================================
# MAGNETOMETER PREPROCESSING
# ============================================================

def prepare_magnetometer(path):

    df = pd.read_csv(path)

    columns = [
        "timestamp",
        "magnetometer_ga[0]",
        "magnetometer_ga[1]",
        "magnetometer_ga[2]"
    ]

    df = df[columns].copy()

    df = (
        df.sort_values("timestamp")
          .drop_duplicates("timestamp")
          .reset_index(drop=True)
    )

    df = df.rename(columns={
        "magnetometer_ga[0]": "mag_x",
        "magnetometer_ga[1]": "mag_y",
        "magnetometer_ga[2]": "mag_z"
    })

    df["mag_magnitude"] = np.sqrt(
        df["mag_x"] ** 2 +
        df["mag_y"] ** 2 +
        df["mag_z"] ** 2
    )

    return df


# ============================================================
# BAROMETER PREPROCESSING
# ============================================================

def prepare_baro(path):

    df = pd.read_csv(path)

    columns = [
        "timestamp",
        "pressure",
        "temperature"
    ]

    df = df[columns].copy()

    df = (
        df.sort_values("timestamp")
          .drop_duplicates("timestamp")
          .reset_index(drop=True)
    )

    return df


# ============================================================
# SENSOR SYNCHRONIZATION
# ============================================================

def synchronize(gps, imu, mag, baro):

    # GPS is our MASTER TIMELINE.

    gps = gps.sort_values("timestamp")
    imu = imu.sort_values("timestamp")
    mag = mag.sort_values("timestamp")
    baro = baro.sort_values("timestamp")

    # --------------------------------------------------------
    # GPS + IMU
    # --------------------------------------------------------

    merged = pd.merge_asof(
        gps,
        imu,
        on="timestamp",
        direction="backward",
        tolerance=TOLERANCE_US
    )

    # --------------------------------------------------------
    # GPS + MAG
    # --------------------------------------------------------

    merged = pd.merge_asof(
        merged.sort_values("timestamp"),
        mag,
        on="timestamp",
        direction="backward",
        tolerance=TOLERANCE_US
    )

    # --------------------------------------------------------
    # GPS + BARO
    # --------------------------------------------------------

    merged = pd.merge_asof(
        merged.sort_values("timestamp"),
        baro,
        on="timestamp",
        direction="backward",
        tolerance=TOLERANCE_US
    )

    return merged


# ============================================================
# DERIVED TEMPORAL FEATURES
# ============================================================

def create_temporal_features(df):

    df = df.sort_values("timestamp").copy()

    # GPS position changes
    df["lat_change"] = df["lat"].diff()
    df["lon_change"] = df["lon"].diff()
    df["alt_change"] = df["alt"].diff()

    # Relative altitude (h_t - h_0) for spatial invariance
    df["relative_alt"] = df["alt"] - df["alt"].iloc[0]

    # GPS velocity changes
    df["velocity_change"] = df["vel_m_s"].diff()

    # Sensor changes
    df["mag_change"] = df["mag_magnitude"].diff()
    df["pressure_change"] = df["pressure"].diff()

    # Rolling statistics
    window = 5

    df["velocity_std"] = (
        df["vel_m_s"]
        .rolling(window)
        .std()
    )

    df["mag_std"] = (
        df["mag_magnitude"]
        .rolling(window)
        .std()
    )

    df["pressure_std"] = (
        df["pressure"]
        .rolling(window)
        .std()
    )

    return df


# ============================================================
# PROCESS ONE SCENARIO
# ============================================================

def process_scenario(name, label):

    folder = os.path.join(ROOT, name)

    print("\n" + "=" * 70)
    print("PROCESSING:", name)
    print("=" * 70)

    gps_path = find_file(
        folder,
        "*vehicle_gps_position_0.csv"
    )

    imu_path = find_file(
        folder,
        "*vehicle_imu_0.csv"
    )

    mag_path = find_file(
        folder,
        "*vehicle_magnetometer_0.csv"
    )

    baro_path = find_file(
        folder,
        "*sensor_baro_0.csv"
    )

    print("GPS :", os.path.basename(gps_path))
    print("IMU :", os.path.basename(imu_path))
    print("MAG :", os.path.basename(mag_path))
    print("BARO:", os.path.basename(baro_path))

    gps = prepare_gps(gps_path)
    imu = prepare_imu(imu_path)
    mag = prepare_magnetometer(mag_path)
    baro = prepare_baro(baro_path)

    print("\nRows before synchronization:")
    print("GPS :", len(gps))
    print("IMU :", len(imu))
    print("MAG :", len(mag))
    print("BARO:", len(baro))

    df = synchronize(
        gps,
        imu,
        mag,
        baro
    )

    # Add temporal features
    df = create_temporal_features(df)

    # Attack label
    df["label"] = label
    df["attack_type"] = name

    # Remove rows where synchronization failed
    sensor_columns = [
        "imu_angle_mag",
        "imu_velocity_mag",
        "mag_magnitude",
        "pressure",
        "temperature"
    ]

    before = len(df)

    df = df.dropna(
        subset=sensor_columns
    ).reset_index(drop=True)

    after = len(df)

    print("\nRows after synchronization:", after)
    print("Rows removed:", before - after)

    output_name = {
        0: "benign",
        1: "jamming",
        2: "spoofing"
    }[label]

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{output_name}_fusion.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("Saved:", output_path)

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    datasets = []

    for name, label in SCENARIOS.items():

        df = process_scenario(
            name,
            label
        )

        datasets.append(df)

    # Combine all scenarios
    combined = pd.concat(
        datasets,
        ignore_index=True
    )

    combined = combined.sort_values(
        ["label", "timestamp"]
    ).reset_index(drop=True)

    output_path = os.path.join(
        OUTPUT_DIR,
        "uav_multisensor_fusion.csv"
    )

    combined.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("FINAL DATASET")
    print("=" * 70)

    print("Shape:", combined.shape)

    print("\nClass distribution:")
    print(combined["attack_type"].value_counts())

    print("\nMissing values:")
    print(
        combined.isna().sum()
        .sort_values(ascending=False)
        .head(15)
    )

    print("\nSaved:", output_path)