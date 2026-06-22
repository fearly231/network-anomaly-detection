import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from data_loader import COLUMN_NAMES, add_binary_target
from predict import load_scaler, preprocess_data
from preprocess import DROP_COLUMNS


class HybridAnomalyDetector:
    """Hybrid Intrusion Detection System combining Autoencoder and Random Forest.

    Implements a Sequential Cascade:
    1. Run inputs through Keras Autoencoder.
    2. If Reconstruction Error (MSE) > threshold_high: classify as ANOMALY (Zero-Day).
    3. If Reconstruction Error (MSE) < threshold_low: classify as NORMAL (Safe).
    4. For borderline cases (threshold_low <= MSE <= threshold_high): pass to Random Forest.
    """

    def __init__(
        self,
        models_dir: str = "models",
        threshold_low: float = 0.05,
        threshold_high: float = 0.5,
        rf_threshold: float = 0.01,
    ):
        self.models_dir = Path(models_dir)

        # Load Autoencoder config (scaler, feature names, threshold)
        config_path = self.models_dir / "model_config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.ae_threshold = self.config["threshold"]
        self.feature_names = self.config["feature_names"]

        # Load scaler for Autoencoder
        self.scaler = load_scaler(self.models_dir / "scaler.json")

        # Load Autoencoder model
        self.ae_model = keras.models.load_model(self.models_dir / "autoencoder.keras")

        # Load Random Forest pipeline (preprocessor + classifier)
        rf_path = self.models_dir / "random_forest.joblib"
        if not rf_path.exists():
            raise FileNotFoundError(
                f"Random Forest model not found at {rf_path}. Please run `python src/train.py` first."
            )
        self.rf_pipeline = joblib.load(rf_path)

        # Custom hybrid thresholds
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.rf_threshold = rf_threshold

    def predict(self, df: pd.DataFrame):
        """Predict anomalies using the Sequential Cascade hybrid approach.

        Returns:
            predictions (np.ndarray): 1 for anomalies, 0 for normal.
            ae_errors (np.ndarray): raw reconstruction errors (MSE).
            decisions_source (np.ndarray): string codes indicating which model made the decision
                                          ('AE_high', 'AE_low', 'RF_anomaly', 'RF_normal').
        """
        # Preprocess for Autoencoder
        X_ae_scaled = preprocess_data(df, self.feature_names, self.scaler)

        # Get Autoencoder reconstruction error
        reconstructed = self.ae_model.predict(X_ae_scaled, verbose=0)
        ae_errors = np.mean(np.square(X_ae_scaled - reconstructed), axis=1)

        n_samples = len(df)
        predictions = np.zeros(n_samples, dtype=int)
        decisions_source = np.array([""] * n_samples, dtype=object)

        # Identify paths based on threshold boundaries
        ae_high_idx = np.where(ae_errors > self.threshold_high)[0]
        ae_low_idx = np.where(ae_errors < self.threshold_low)[0]
        rf_idx = np.where((ae_errors >= self.threshold_low) & (ae_errors <= self.threshold_high))[0]

        # 1. High AE Error -> Automatic Anomaly (1)
        predictions[ae_high_idx] = 1
        decisions_source[ae_high_idx] = "AE_high"

        # 2. Low AE Error -> Automatic Normal (0)
        predictions[ae_low_idx] = 0
        decisions_source[ae_low_idx] = "AE_low"

        # 3. Borderline Error -> Evaluate via Random Forest
        if len(rf_idx) > 0:
            df_rf = df.iloc[rf_idx]
            # Ensure target/label/difficulty_level columns are dropped to match train schema
            cols_to_drop = [col for col in DROP_COLUMNS if col in df_rf.columns]
            x_rf = df_rf.drop(columns=cols_to_drop)

            # Get Random Forest predicted probabilities
            rf_probs = self.rf_pipeline.predict_proba(x_rf)[:, 1]
            rf_preds = (rf_probs >= self.rf_threshold).astype(int)

            predictions[rf_idx] = rf_preds
            decisions_source[rf_idx] = np.where(rf_preds == 1, "RF_anomaly", "RF_normal")

        return predictions, ae_errors, decisions_source


def save_hybrid_confusion_matrix_plot(confusion: np.ndarray, t_low: float, t_high: float, rf_t: float, output_path: Path):
    """Save the hybrid model confusion matrix plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    plt.title(f"Hybrid Cascade CM\n(t_low={t_low:.3f}, t_high={t_high:.3f}, rf_t={rf_t:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] Saved hybrid confusion matrix plot to: {output_path}")


def save_decisions_distribution_plot(decisions: np.ndarray, output_path: Path):
    """Save a bar plot showing the proportion of decisions made by each model component."""
    unique, counts = np.unique(decisions, return_counts=True)
    dist = dict(zip(unique, counts))

    # Order keys for display
    order = ["AE_low", "AE_high", "RF_normal", "RF_anomaly"]
    ordered_labels = []
    ordered_values = []
    for key in order:
        if key in dist:
            ordered_labels.append(key)
            ordered_values.append(dist[key])

    plt.figure(figsize=(8, 5))
    colors = ["#2a9d8f", "#e76f51", "#a8dadc", "#457b9d"]
    bars = plt.bar(ordered_labels, ordered_values, color=colors[:len(ordered_labels)], edgecolor="black", alpha=0.85)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max(ordered_values)*0.01), f"{yval:,} ({yval/sum(ordered_values):.1%})", ha="center", va="bottom", fontweight="bold")

    plt.title("Distribution of Decision Sources in Hybrid IDS")
    plt.xlabel("Decision Path")
    plt.ylabel("Sample Count")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] Saved decision distribution plot to: {output_path}")


def main():
    # Load test dataset
    test_path = Path("data/raw/KDDTest+.txt")
    print(f"Loading test dataset from: {test_path}...")
    test_df = add_binary_target(pd.read_csv(test_path, header=None, names=COLUMN_NAMES))
    y_test = test_df["target"].to_numpy()

    # Initialize hybrid detector
    # We will use the autoencoder threshold (0.25) as a reference point.
    # threshold_low: below this, we are sure it's normal (e.g. 0.05)
    # threshold_high: above this, we are sure it's an anomaly (e.g. 0.50)
    # rf_threshold: 0.01 (baseline tuned threshold)
    print("\nInitializing Hybrid Anomaly Detector...")
    t_low = 0.05
    t_high = 0.50
    rf_t = 0.01
    
    detector = HybridAnomalyDetector(
        models_dir="models",
        threshold_low=t_low,
        threshold_high=t_high,
        rf_threshold=rf_t
    )
    print(f"Loaded Autoencoder base threshold: {detector.ae_threshold:.4f}")
    print(f"Using Hybrid Cascade limits: Low={t_low:.4f}, High={t_high:.4f}")
    print(f"Using Random Forest decision threshold: {rf_t:.2f}")

    print("\nRunning inference with Hybrid Model on the test set...")
    predictions, ae_errors, decisions = detector.predict(test_df)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\n================== HYBRID CASCADE EVALUATION ==================")
    print(f"Accuracy:  {accuracy:.4%}")
    print(f"Precision: {precision:.4%}")
    print(f"Recall:    {recall:.4%}")
    print(f"F1-Score:  {f1:.4%}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions, digits=4))

    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(cm)

    # Display decision statistics
    unique, counts = np.unique(decisions, return_counts=True)
    print("\nDecision Sources breakdown:")
    for label, count in zip(unique, counts):
        print(f"  - {label}: {count} ({count/len(decisions):.2%})")

    # Plots
    plot_cm_path = Path("logs/plots/hybrid_confusion_matrix.png")
    plot_dist_path = Path("logs/plots/hybrid_decisions_distribution.png")
    save_hybrid_confusion_matrix_plot(cm, t_low, t_high, rf_t, plot_cm_path)
    save_decisions_distribution_plot(decisions, plot_dist_path)

    # Save log of the experiment
    logs_dir = Path("logs/experiments")
    logs_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = logs_dir / f"hybrid_cascade_{timestamp}.json"
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "hybrid_cascade",
        "parameters": {
            "ae_threshold_base": detector.ae_threshold,
            "threshold_low": t_low,
            "threshold_high": t_high,
            "rf_threshold": rf_t
        },
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        },
        "confusion_matrix": cm.tolist(),
        "decision_sources": dict(zip(unique, [int(c) for c in counts]))
    }
    
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"\n[OK] Experiment log saved: {log_file}")


if __name__ == "__main__":
    main()
