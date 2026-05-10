from pathlib import Path

from data_loader import add_binary_target, load_nsl_kdd


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


if __name__ == "__main__":
    main()
