import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras

import sys
sys.path.append("src")
from prepare_nsl_kdd_autoencoder import prepare_autoencoder_data
from train_autoencoder import to_dense, reconstruction_errors

def main():
    train_path = Path("data/raw/KDDTrain+.txt")
    test_path = Path("data/raw/KDDTest+.txt")

    print("Loading data...")
    X_train_scaled, X_val_scaled, y_val_binary, X_test_scaled, y_test_binary, _, _ = prepare_autoencoder_data(
        train_path=train_path,
        test_path=test_path,
    )

    X_train_dense = to_dense(X_train_scaled)
    X_val_dense = to_dense(X_val_scaled)
    X_test_dense = to_dense(X_test_scaled)

    input_dim = X_train_dense.shape[1]
    
    print("Building and training autoencoder...")
    # Build
    inputs = keras.Input(shape=(input_dim,), name="input_layer")
    encoded = keras.layers.Dense(32, activation="relu", name="encoder_32")(inputs)
    bottleneck = keras.layers.Dense(16, activation="relu", name="bottleneck_16")(encoded)
    decoded = keras.layers.Dense(32, activation="relu", name="decoder_32")(bottleneck)
    outputs = keras.layers.Dense(input_dim, activation="linear", name="reconstruction")(decoded)

    autoencoder = keras.Model(inputs=inputs, outputs=outputs, name="nsl_kdd_autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")

    # Fit
    autoencoder.fit(
        X_train_dense,
        X_train_dense,
        epochs=10,
        batch_size=256,
        validation_split=0.1,
        shuffle=True,
        verbose=0,
    )

    # Errors
    val_errors = reconstruction_errors(autoencoder, X_val_dense)
    train_errors = reconstruction_errors(autoencoder, X_train_dense)
    test_errors = reconstruction_errors(autoencoder, X_test_dense)

    print("\n--- Method 1: Percentile based on Training Set (Normal only) ---")
    # Setting threshold at 95th percentile of normal training errors
    thresh_train_95 = np.percentile(train_errors, 95)
    y_val_pred_m1 = (val_errors > thresh_train_95).astype(int)
    print(f"Threshold (95% of train normal): {thresh_train_95:.6f}")
    print(classification_report(y_val_binary, y_val_pred_m1, digits=4))

    print("\n--- Method 2: Optimize Threshold for F1-score on Validation Set ---")
    best_f1 = 0.0
    best_thresh = 0.0
    
    # Grid search over validation error percentiles
    percentiles = np.linspace(1, 99, 200)
    for p in percentiles:
        t = np.percentile(val_errors, p)
        pred = (val_errors > t).astype(int)
        f1 = f1_score(y_val_binary, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    print(f"Optimal Threshold (F1-optimized): {best_thresh:.6f}")
    y_val_pred_m2 = (val_errors > best_thresh).astype(int)
    print(classification_report(y_val_binary, y_val_pred_m2, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val_binary, y_val_pred_m2))

    # Evaluate on Test Set
    print("\n--- Test Set Evaluation (using F1-optimized Threshold) ---")
    y_test_pred = (test_errors > best_thresh).astype(int)
    print(classification_report(y_test_binary, y_test_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_binary, y_test_pred))

if __name__ == "__main__":
    main()
