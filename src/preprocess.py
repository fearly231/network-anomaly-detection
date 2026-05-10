from typing import Tuple

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
DROP_COLUMNS = ["label", "difficulty_level", "target"]


def build_preprocessor() -> ColumnTransformer:
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numeric_transformer = StandardScaler()

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, CATEGORICAL_COLUMNS),
            ("numeric", numeric_transformer, lambda df: [c for c in df.columns if c not in CATEGORICAL_COLUMNS]),
        ],
        sparse_threshold=0,
    )


def prepare_features(train_df, test_df) -> Tuple:
    x_train = train_df.drop(columns=DROP_COLUMNS)
    y_train = train_df["target"]
    x_test = test_df.drop(columns=DROP_COLUMNS)
    y_test = test_df["target"]

    preprocessor = build_preprocessor()
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)

    return x_train_processed, y_train, x_test_processed, y_test, preprocessor
