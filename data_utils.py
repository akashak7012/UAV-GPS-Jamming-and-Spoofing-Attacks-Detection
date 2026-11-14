import os
import pandas as pd

def load_and_split_data(path, features=None):
    """
    Loads data and splits into 70/30 temporally by attack type.
    Optionally filters by specific features if provided.
    
    Returns:
        X_train, X_test, y_train, y_test, train_df, test_df
    """
    path = os.path.normpath(str(path).replace("\\", "/"))
    df = pd.read_csv(path)
    
    # Sort chronologically within each class
    df = df.sort_values(
        ["attack_type", "timestamp"]
    ).reset_index(drop=True)
    
    train_parts = []
    test_parts = []
    
    for attack_type, group in df.groupby("attack_type"):
        group = group.sort_values("timestamp")
        split = int(len(group) * 0.70)
        
        train_parts.append(group.iloc[:split])
        test_parts.append(group.iloc[split:])
        
    train_df = pd.concat(train_parts)
    test_df = pd.concat(test_parts)
    
    if features is not None:
        X_train = train_df[features]
        X_test = test_df[features]
    else:
        # Default: all columns except metadata
        exclude = ["timestamp", "label", "attack_type"]
        cols = [c for c in df.columns if c not in exclude]
        X_train = train_df[cols]
        X_test = test_df[cols]
        
    y_train = train_df["label"]
    y_test = test_df["label"]
    
    return X_train, X_test, y_train, y_test, train_df, test_df

