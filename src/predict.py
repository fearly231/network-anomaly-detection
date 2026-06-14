import json
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# Constants for preprocessing (aligned with training)
CAT_COLUMNS = ["protocol_type", "service", "flag"]
SKEWED_COLUMNS = ["duration", "src_bytes", "dst_bytes"]

def load_scaler(scaler_path: Path) -> StandardScaler:
    """Reconstruct StandardScaler securely from saved JSON parameters."""
    with open(scaler_path, "r") as f:
        scaler_data = json.load(f)
        
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_data["mean"])
    scaler.var_ = np.array(scaler_data["var"])
    scaler.scale_ = np.array(scaler_data["scale"])
    scaler.n_samples_seen_ = np.array(scaler_data["n_samples_seen"])
    return scaler

def preprocess_data(df: pd.DataFrame, expected_features: list, scaler: StandardScaler) -> np.ndarray:
    """Preprocess raw DataFrame to match the input shape of the Autoencoder."""
    df_clean = df.copy()
    
    # 1. Drop unused columns if present
    for col in ["difficulty_level", "label", "target"]:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])
            
    # 2. Log-transform skewed features
    for col in SKEWED_COLUMNS:
        if col in df_clean.columns:
            df_clean[col] = np.log1p(df_clean[col].astype("float64"))
            
    # 3. Set categorical data types
    for col in CAT_COLUMNS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype("object")
            
    # 4. Convert numerical features to numeric types
    for col in df_clean.columns:
        if col not in CAT_COLUMNS:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0.0)
            
    # 5. One-hot encoding
    df_processed = pd.get_dummies(df_clean, columns=CAT_COLUMNS)
    
    # 6. Align columns with expected features from training config
    # Add missing columns with 0
    missing_cols = set(expected_features) - set(df_processed.columns)
    for col in missing_cols:
        df_processed[col] = 0.0
        
    # Keep only the expected columns in the correct training order
    df_aligned = df_processed[expected_features].astype("float64")
    
    # 7. Scale using the loaded scaler
    X_scaled = scaler.transform(df_aligned.to_numpy())
    return X_scaled

class AnomalyDetector:
    """Classifier wrapper using the trained Keras Autoencoder."""
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        
        # Load config (threshold & features)
        config_path = self.models_dir / "model_config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.threshold = self.config["threshold"]
        self.feature_names = self.config["feature_names"]
        
        # Load scaler
        self.scaler = load_scaler(self.models_dir / "scaler.json")
        
        # Load neural network model
        self.model = keras.models.load_model(self.models_dir / "autoencoder.keras")
        
    def predict(self, df: pd.DataFrame):
        """Predict anomalies for a raw DataFrame.
        
        Returns:
            predictions (np.ndarray): 1 for anomalies, 0 for normal.
            errors (np.ndarray): raw reconstruction errors (MSE).
        """
        X_scaled = preprocess_data(df, self.feature_names, self.scaler)
        
        # Run prediction through Autoencoder
        reconstructed = self.model.predict(X_scaled, verbose=0)
        
        # Calculate Mean Squared Error (reconstruction error) per sample
        errors = np.mean(np.square(X_scaled - reconstructed), axis=1)
        
        # Classify based on optimal threshold
        predictions = (errors > self.threshold).astype(int)
        return predictions, errors

def save_confusion_matrix_plot(confusion: np.ndarray, threshold: float, output_path: Path):
    """Save the confusion matrix as a PNG image."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    plt.title(f"Autoencoder Test Confusion Matrix (threshold={threshold:.6f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] Saved confusion matrix plot to: {output_path}")

def main():
    from data_loader import COLUMN_NAMES, add_binary_target
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("Initializing detector and loading saved model components...")
    detector = AnomalyDetector(models_dir="models")
    print(f"Optimal threshold loaded: {detector.threshold:.6f}")
    
    # Load raw unseen test data
    test_path = Path("data/raw/KDDTest+.txt")
    print(f"Loading test dataset from: {test_path}...")
    test_df = add_binary_target(pd.read_csv(test_path, header=None, names=COLUMN_NAMES))
    
    y_test = test_df["target"].to_numpy()
    
    print("Running anomaly detection on the test set...")
    predictions, errors = detector.predict(test_df)
    
    print("\nTest Set Evaluation:")
    print(classification_report(y_test, predictions, digits=4))
    
    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(cm)
    
    # Save the plot
    plot_path = Path("logs/plots/autoencoder_test_confusion_matrix.png")
    save_confusion_matrix_plot(cm, detector.threshold, plot_path)

if __name__ == "__main__":
    main()
