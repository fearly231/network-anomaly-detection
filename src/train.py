import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from data_loader import add_binary_target, load_nsl_kdd
from modeling import evaluate_binary_classifier, train_baseline_model
from preprocess import prepare_features


def print_dataset_summary(name: str, df) -> None:
    counts = df["target"].value_counts().sort_index()
    total = len(df)
    normal = counts.get(0, 0)
    anomaly = counts.get(1, 0)

    print(f"\n{name}")
    print(f"Rows: {total}")
    print(f"Normal (0):  {normal} ({normal / total:.2%})")
    print(f"Anomaly (1): {anomaly} ({anomaly / total:.2%})")


def save_experiment_log(
    metrics: dict, confusion: object, model_params: dict, experiment_name: str = "baseline"
) -> Path:
    """Zapisuje metryki eksperymentu do JSON z timestampem."""
    logs_dir = Path("logs/experiments")
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = logs_dir / f"{experiment_name}_{timestamp}.json"

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "experiment": experiment_name,
        "model_params": model_params,
        "metrics": metrics,
        "confusion_matrix": confusion.tolist(),
    }

    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"\n✓ Log saved: {log_file}")
    return log_file


def plot_confusion_matrix(confusion: object, experiment_name: str = "baseline") -> Path:
    """Rysuje i zapisuje heatmapę confusion matrix."""
    plots_dir = Path("logs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_file = plots_dir / f"confusion_matrix_{experiment_name}_{timestamp}.png"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    plt.title(f"Confusion Matrix - {experiment_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=100)
    plt.close()

    print(f"✓ Plot saved: {plot_file}")
    return plot_file


def plot_roc_pr_curves(
    y_test, y_proba, experiment_name: str = "baseline"
) -> Path:
    """Rysuje i zapisuje krzywe ROC i PR."""
    plots_dir = Path("logs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_file = plots_dir / f"roc_pr_curves_{experiment_name}_{timestamp}.png"

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC curve
    axes[0].plot(fpr, tpr, label="ROC curve", linewidth=2)
    axes[0].plot([0, 1], [0, 1], "k--", label="Random classifier")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve - {experiment_name}")
    axes[0].legend()
    axes[0].grid()

    # PR curve
    axes[1].plot(recall, precision, label="PR curve", linewidth=2)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve - {experiment_name}")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.savefig(plot_file, dpi=100)
    plt.close()

    print(f"✓ Plot saved: {plot_file}")
    return plot_file


def main() -> None:
    train_path = Path("data/raw/KDDTrain+.txt")
    test_path = Path("data/raw/KDDTest+.txt")

    train_df = add_binary_target(load_nsl_kdd(train_path))
    test_df = add_binary_target(load_nsl_kdd(test_path))

    print_dataset_summary("TRAIN", train_df)
    print_dataset_summary("TEST", test_df)

    x_train, y_train, x_test, y_test, _ = prepare_features(train_df, test_df)
    print("\nPREPROCESSING")
    print(f"X_train shape: {x_train.shape}")
    print(f"X_test shape:  {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")

    model = train_baseline_model(x_train, y_train)
    metrics, confusion = evaluate_binary_classifier(model, x_test, y_test)
    
    # Pobierz prawdopodobieństwa dla krzywych ROC/PR
    y_proba = model.predict_proba(x_test)[:, 1]

    print("\nBASELINE: RandomForestClassifier")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nConfusion matrix [ [TN, FP], [FN, TP] ]")
    print(confusion)

    # Zapisz wyniki
    model_params = {
        "model": "RandomForestClassifier",
        "n_estimators": 300,
        "class_weight": "balanced_subsample",
        "random_state": 42,
    }
    save_experiment_log(metrics, confusion, model_params, experiment_name="baseline")
    plot_confusion_matrix(confusion, experiment_name="baseline")
    plot_roc_pr_curves(y_test, y_proba, experiment_name="baseline")


if __name__ == "__main__":
    main()
