# Network Anomaly Detection (NSL-KDD)

## Etap 1 (start)
Cel: poprawnie wczytać dane i przygotować etykietę binarną:
- `0` = normal
- `1` = anomaly

## Etap 2 (preprocessing)
Cel: zamienić surowe dane na wejście dla modelu:
- `protocol_type`, `service`, `flag` -> One-Hot Encoding
- cechy numeryczne -> StandardScaler
- wyjście: gotowe `X_train`, `X_test`, `y_train`, `y_test`

## Etap 3 (baseline model)
Cel: wytrenować pierwszy model klasyfikacji binarnej:
- model: `RandomForestClassifier(class_weight="balanced_subsample")`
- metryki: accuracy, precision, recall, f1, ROC-AUC, PR-AUC
- dodatkowo: confusion matrix

### Uruchomienie
```bash
pip install -r requirements.txt
python src/train.py
```

Skrypt wypisze:
- rozkład klas w zbiorze treningowym i testowym
- rozmiary przetworzonych macierzy cech
- metryki baseline modelu
