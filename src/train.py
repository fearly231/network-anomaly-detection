import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from data_loader import add_binary_target, load_nsl_kdd
from modeling import evaluate_binary_classifier
from preprocess import build_preprocessor, DROP_COLUMNS


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

    # Use raw DataFrames here; the preprocessor is part of the pipeline
    x_train = train_df.drop(columns=DROP_COLUMNS)
    y_train = train_df["target"]
    x_test = test_df.drop(columns=DROP_COLUMNS)
    y_test = test_df["target"]

    print("\nPREPROCESSING (pipeline will handle transforms)")
    print(f"X_train shape: {x_train.shape}")
    print(f"X_test shape:  {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")

    preprocessor = build_preprocessor()
    # lazy imports
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
    from imblearn.pipeline import Pipeline as ImbPipeline
   

    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ])

    param_dist = {
        "clf__n_estimators": [100, 200, 300, 400],
        "clf__max_depth": [None, 10, 20, 50],
        "clf__class_weight": [None, "balanced", "balanced_subsample"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=20,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )

    print("\nRunning RandomizedSearchCV ...")
    search.fit(x_train, y_train)

    print("\nBest params:")
    print(search.best_params_)

    best = search.best_estimator_

    # Treshold
    chosen_threshold = 0.01
    metrics, confusion, y_proba = evaluate_binary_classifier(best, x_test, y_test, threshold=chosen_threshold)

    print("\nBEST MODEL")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nConfusion matrix [ [TN, FP], [FN, TP] ]")
    print(confusion)

    model_params = {**search.best_params_, "model": "RandomForestClassifier"}
    save_experiment_log(metrics, confusion, model_params, experiment_name="rf_random_search")
    plot_confusion_matrix(confusion, experiment_name="rf_random_search")
    plot_roc_pr_curves(y_test, y_proba, experiment_name="rf_random_search")


if __name__ == "__main__":
    main()
