# Network Anomaly Detection (NSL-KDD)

This repository implements machine learning models to detect network intrusion anomalies using the benchmark **NSL-KDD** dataset. It features both a supervised classifier (**Random Forest**) and an unsupervised reconstruction-based model (**Autoencoder**).

---

## Project Structure

```text
├── data/
│   └── raw/                   # Raw NSL-KDD datasets (KDDTrain+.txt, KDDTest+.txt)
├── logs/
│   ├── experiments/           # JSON files logging training parameters and metrics
│   └── plots/                 # Training curves and confusion matrix heatmaps
├── models/                    # Saved production models, scalers, and configs
│   ├── autoencoder.keras      # Saved Autoencoder neural network
│   ├── scaler.json            # StandardScaler parameters
│   └── model_config.json      # Decision threshold and expected features
├── notebooks/                 # Exploratory Data Analysis & experiments
├── src/
│   ├── data_loader.py         # Parses NSL-KDD files and formats binary targets
│   ├── preprocess.py          # Data preprocessing pipeline for supervised models
│   ├── modeling.py            # Baseline model definitions and evaluation helpers
│   ├── train.py               # Supervised Random Forest training script
│   ├── prepare_nsl_kdd_autoencoder.py # Autoencoder-specific preprocessing (Log1p & StandardScaler)
│   ├── train_autoencoder.py   # Autoencoder training and threshold optimizer script
│   └── predict.py             # Inference pipeline for predicting anomalies on new data
└── requirements.txt           # Python dependencies
```

---

## Installation & Setup

1. **Clone the repository** and navigate to the root directory:
   ```bash
   cd network-anomaly-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Models and Training

### 1. Supervised Classifier (Random Forest)
Trains a Random Forest classifier using cross-validation and hyperparameter search.
* **To run training**:
  ```bash
  python src/train.py
  ```
* **Output**: Logs the best model parameters to `logs/experiments/` and generates evaluation curves in `logs/plots/`.

### 2. Unsupervised Classifier (Autoencoder)
Trains a deep reconstruction network only on normal traffic. Anomalies are detected by measuring the reconstruction error (MSE) against an optimized decision boundary.
* **Preprocessing highlights**:
  * Applies $\log(x + 1)$ scaling on highly skewed features (`duration`, `src_bytes`, `dst_bytes`) to prevent extreme outliers from dominating the MSE loss.
  * Standardizes features using a centered `StandardScaler` (`with_mean=True`).
* **Threshold Optimization**:
  Instead of utilizing a hardcoded percentile threshold, the script runs a search loop to find the exact classification threshold that maximizes the validation **F1-score**.
* **To run training**:
  ```bash
  python src/train_autoencoder.py
  ```
* **Output**: Saves the trained Keras model, scaling parameters, and optimal threshold configuration to the `models/` directory.

---

## Running Inference (Prediction)

You can run predictions on new, raw data using the saved Autoencoder model. 

### 1. Run Test Set Prediction
A verification script is provided to load the trained model and evaluate it against the unseen test set (`KDDTest+.txt`):
```bash
python src/predict.py
```
* **Output**: Prints the classification report in the terminal and saves a confusion matrix plot to `logs/plots/autoencoder_test_confusion_matrix.png`.

### 2. Integrate into Python Code
To predict anomalies on raw pandas DataFrames in your own codebase, import and use the `AnomalyDetector` class:

```python
import pandas as pd
from src.predict import AnomalyDetector

# 1. Initialize detector (loads autoencoder.keras, scaler.json, and model_config.json)
detector = AnomalyDetector(models_dir="models")

# 2. Load raw network flow data (formatted as NSL-KDD columns)
new_data = pd.read_csv("path/to/network_traffic.csv")

# 3. Perform prediction
# predictions: 1 for anomalies, 0 for normal traffic
# errors: raw Mean Squared Error reconstruction errors
predictions, errors = detector.predict(new_data)
```

---

## Secure Serialization

This project implements **secure model serialization**:
* Model weights and architecture are saved using Keras native `.keras` format (a secure zip archive of JSON configuration and weights).
* Scaler coefficients (`mean`, `variance`, `scale`, `n_samples_seen`) are serialized to **`scaler.json`**. This completely avoids using Python `pickle` (`.pkl`), removing any threat of insecure deserialization/arbitrary code execution vulnerabilities in production environments.
