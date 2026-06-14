from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow import keras
from tensorflow.keras import layers

from prepare_nsl_kdd_autoencoder import prepare_autoencoder_data


RANDOM_STATE = 42
EPOCHS = 10
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.1
PLOTS_DIR = Path("logs/plots")


def to_dense(matrix):
    """Convert sparse matrices to dense arrays for Keras.

    The NSL-KDD preprocessing step can produce sparse matrices after one-hot
    encoding. Keras autoencoders operate on dense tensors, so we materialize the
    arrays here before fitting and scoring the model.
    """

    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def build_autoencoder(input_dim: int) -> keras.Model:
    """Build the baseline autoencoder architecture.

    The reconstruction error will later be used as an anomaly score: patterns
    that are hard to reconstruct are treated as non-self traffic.
    """

    inputs = keras.Input(shape=(input_dim,), name="input_layer")
    encoded = layers.Dense(32, activation="relu", name="encoder_32")(inputs)
    bottleneck = layers.Dense(16, activation="relu", name="bottleneck_16")(encoded)
    decoded = layers.Dense(32, activation="relu", name="decoder_32")(bottleneck)
    outputs = layers.Dense(input_dim, activation="linear", name="reconstruction")(decoded)

    model = keras.Model(inputs=inputs, outputs=outputs, name="nsl_kdd_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def reconstruction_errors(model: keras.Model, X) -> np.ndarray:
    """Return per-sample mean squared reconstruction errors."""

    X_dense = to_dense(X)
    reconstructed = model.predict(X_dense, verbose=0)
    return np.mean(np.square(X_dense - reconstructed), axis=1)


def save_confusion_matrix_plot(confusion: np.ndarray, threshold: float) -> Path:
    """Save the validation confusion matrix as a PNG image."""

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_file = PLOTS_DIR / "autoencoder_confusion_matrix.png"

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    plt.title(f"Autoencoder Confusion Matrix (threshold={threshold:.6f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    return plot_file


def save_reconstruction_error_plot(val_errors: np.ndarray, threshold: float) -> Path:
    """Save a histogram of validation reconstruction errors with the decision threshold."""

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_file = PLOTS_DIR / "autoencoder_reconstruction_errors.png"

    plt.figure(figsize=(10, 5))
    sns.histplot(val_errors, bins=60, kde=True, color="#2a6f97")
    plt.axvline(threshold, color="#d00000", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.6f}")
    plt.title("Validation Reconstruction Error Distribution")
    plt.xlabel("Mean Squared Reconstruction Error")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    return plot_file


def save_training_history_plot(history) -> Path:
    """Save the autoencoder training and validation loss curve."""

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_file = PLOTS_DIR / "autoencoder_training_history.png"

    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Train loss", linewidth=2)
    plt.plot(history.history["val_loss"], label="Validation loss", linewidth=2)
    plt.title("Autoencoder Training History")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    return plot_file


def train_and_evaluate_autoencoder(
    X_train_scaled,
    X_val_scaled,
    y_val_binary,
):
    """Train the autoencoder on normal traffic and evaluate on validation data."""

    X_train_dense = to_dense(X_train_scaled)
    X_val_dense = to_dense(X_val_scaled)

    input_dim = X_train_dense.shape[1]
    autoencoder = build_autoencoder(input_dim)

    history = autoencoder.fit(
        X_train_dense,
        X_train_dense,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        shuffle=True,
        verbose=1,
    )

    val_errors = reconstruction_errors(autoencoder, X_val_dense)
    
    # Optimize threshold on validation set (maximize F1-score)
    print("Finding optimal threshold ...")
    best_f1 = 0
    best_threshold = 0
    percentile_candidates = np.percentile(val_errors, np.linspace(50, 99.9, 500))
    for thresh in percentile_candidates:
        y_pred = (val_errors > thresh).astype(int)
        f1 = f1_score(y_val_binary, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    threshold = best_threshold
    y_val_pred = (val_errors > threshold).astype(int)

    report = classification_report(y_val_binary, y_val_pred, digits=4)
    confusion = confusion_matrix(y_val_binary, y_val_pred)

    return autoencoder, history, val_errors, threshold, y_val_pred, report, confusion


def main() -> None:
    train_path = Path("data/raw/KDDTrain+.txt")
    test_path = Path("data/raw/KDDTest+.txt")

    X_train_scaled, X_val_scaled, y_val_binary, X_test_scaled, y_test_binary, scaler, combined_features = prepare_autoencoder_data(
        train_path=train_path,
        test_path=test_path,
    )

    autoencoder, history, val_errors, threshold, y_val_pred, report, confusion = train_and_evaluate_autoencoder(
        X_train_scaled=X_train_scaled,
        X_val_scaled=X_val_scaled,
        y_val_binary=y_val_binary,
    )

    history_plot = save_training_history_plot(history)
    error_plot = save_reconstruction_error_plot(val_errors, threshold)
    confusion_plot = save_confusion_matrix_plot(confusion, threshold)

    print("Autoencoder training complete")
    print(f"Input dimension: {autoencoder.input_shape[-1]}")
    print(f"Final feature count: {X_train_scaled.shape[1]}")
    print(f"Validation reconstruction error shape: {val_errors.shape}")
    print(f"95th percentile threshold: {threshold:.6f}")
    print(f"Training history plot saved: {history_plot}")
    print(f"Reconstruction error plot saved: {error_plot}")
    print(f"Confusion matrix plot saved: {confusion_plot}")
    print("\nClassification report (validation):")
    print(report)
    print("Confusion matrix (validation):")
    print(confusion)
    print(f"X_train_scaled shape: {to_dense(X_train_scaled).shape}")
    print(f"X_val_scaled shape:   {to_dense(X_val_scaled).shape}")
    print(f"y_val_binary shape:   {y_val_binary.shape}")
    print(f"X_test_scaled shape:  {to_dense(X_test_scaled).shape}")
    print(f"y_test_binary shape:  {y_test_binary.shape}")

    # Save the model, scaler and threshold config
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Keras Model
    model_path = models_dir / "autoencoder.keras"
    autoencoder.save(model_path)
    print(f"\n[OK] Saved model to {model_path}")

    # 2. Save Scaler to JSON (secure alternative to pickle)
    scaler_path = models_dir / "scaler.json"
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "var": scaler.var_.tolist(),
        "scale": scaler.scale_.tolist(),
        "n_samples_seen": scaler.n_samples_seen_.tolist() if isinstance(scaler.n_samples_seen_, np.ndarray) else int(scaler.n_samples_seen_)
    }
    with open(scaler_path, "w") as f:
        json.dump(scaler_data, f, indent=4)
    print(f"[OK] Saved scaler to {scaler_path}")

    # 3. Save Config (Threshold & Feature Names)
    config_path = models_dir / "model_config.json"
    config_data = {
        "threshold": float(threshold),
        "feature_names": combined_features.columns.tolist()
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)
    print(f"[OK] Saved configuration to {config_path}")


if __name__ == "__main__":
    main()