from pathlib import Path

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

    print("\nBASELINE: RandomForestClassifier")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nConfusion matrix [ [TN, FP], [FN, TP] ]")
    print(confusion)


if __name__ == "__main__":
    main()
