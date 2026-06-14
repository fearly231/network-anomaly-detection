import numpy as np
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_loader import COLUMN_NAMES


CAT_COLUMNS = ["protocol_type", "service", "flag"]
LABEL_COLUMN = "label"
DIFFICULTY_COLUMN = "difficulty_level"
TARGET_COLUMN = "target"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def prepare_autoencoder_data(
    train_path: Path,
    test_path: Path,
    validation_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
):
    
    train_df = pd.read_csv(train_path, sep=",", header=None, names=COLUMN_NAMES, low_memory=False)
    test_df = pd.read_csv(test_path, sep=",", header=None, names=COLUMN_NAMES, low_memory=False)

   
    train_df = train_df.drop(columns=[DIFFICULTY_COLUMN])
    test_df = test_df.drop(columns=[DIFFICULTY_COLUMN])

    # Log-transform highly skewed features to prevent outliers from dominating reconstruction error
    skewed_features = ["duration", "src_bytes", "dst_bytes"]
    for col in skewed_features:
        train_df[col] = np.log1p(train_df[col])
        test_df[col] = np.log1p(test_df[col])
    
    for column in CAT_COLUMNS:
        train_df[column] = train_df[column].astype("object")
        test_df[column] = test_df[column].astype("object")

    numeric_columns = [
        column
        for column in COLUMN_NAMES
        if column not in CAT_COLUMNS + [LABEL_COLUMN, DIFFICULTY_COLUMN]
    ]
    for column in numeric_columns:
        train_df[column] = pd.to_numeric(train_df[column], errors="coerce")
        test_df[column] = pd.to_numeric(test_df[column], errors="coerce")

    
    train_df[TARGET_COLUMN] = (train_df[LABEL_COLUMN] != "normal").astype(int)
    test_df[TARGET_COLUMN] = (test_df[LABEL_COLUMN] != "normal").astype(int)

    
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    
    combined_df = pd.get_dummies(combined_df, columns=CAT_COLUMNS)

    feature_columns = [
        column
        for column in combined_df.columns
        if column not in {LABEL_COLUMN, TARGET_COLUMN}
    ]
    combined_features = combined_df[feature_columns].astype("float64")

    
    train_rows = len(train_df)
    X_train_full = combined_features.iloc[:train_rows].to_numpy()
    X_test = combined_features.iloc[train_rows:].to_numpy()
    y_train_full = train_df[TARGET_COLUMN].to_numpy()
    y_test = test_df[TARGET_COLUMN].to_numpy()

    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    
    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]

    
    scaler = StandardScaler(with_mean=True)
    X_train_normal = scaler.fit_transform(X_train_normal)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train_normal, X_val, y_val, X_test, y_test, scaler, combined_features


def main() -> None:
    train_path = Path("data/raw/KDDTrain+.txt")
    test_path = Path("data/raw/KDDTest+.txt")

    X_train_normal, X_val, y_val, X_test, y_test, scaler, combined_features = prepare_autoencoder_data(
        train_path=train_path,
        test_path=test_path,
    )

    print("NSL-KDD autoencoder preprocessing complete")
    print(f"Columns defined: {len(COLUMN_NAMES)}")
    print(f"Final feature count: {X_train_normal.shape[1]}")
    print(f"Combined feature count before split: {combined_features.shape[1]}")
    print(f"X_train_normal shape: {X_train_normal.shape}")
    print(f"X_val shape:          {X_val.shape}")
    print(f"y_val shape:          {y_val.shape}")
    print(f"X_test shape:         {X_test.shape}")
    print(f"y_test shape:         {y_test.shape}")


if __name__ == "__main__":
    main()