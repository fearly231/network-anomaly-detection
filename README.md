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
│   ├── random_forest.joblib   # Saved Random Forest classifier pipeline
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
│   ├── hybrid_predict.py      # Decision cascade integrating both Autoencoder and Random Forest
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
* **Output**: Saves the trained pipeline to `models/random_forest.joblib`, logs the best model parameters to `logs/experiments/`, and generates evaluation curves in `logs/plots/`.

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

### 3. Hybrid Classifier (Sequential Cascade)
Integrates both the supervised and unsupervised models into a two-stage hybrid IDS.
* **How it works**:
  1. The incoming traffic is first evaluated by the **Autoencoder**.
  2. If the reconstruction error is extremely high (above `threshold_high = 0.50`), it is classified as an **Anomaly** immediately (fast path for Zero-Day attacks / obvious outliers).
  3. If the reconstruction error is very low (below `threshold_low = 0.05`), it is classified as **Normal** immediately (fast path for safe traffic).
  4. Borderline cases (`0.05 <= error <= 0.50`) are forwarded to the **Random Forest** pipeline for a precise classification.
* **Benefits**: Saves substantial computation by resolving **81.3%** of traffic instantly through the lightweight Autoencoder, while achieving higher overall test metrics than either model individually:
  * **Accuracy**: **93.23%** (vs. RF 93.14%, AE 92.93%)
  * **Precision**: **90.51%** (vs. RF 90.48%, AE 90.42%)
  * **Recall**: **98.43%** (vs. RF 98.29%, AE 98.25%)
  * **F1-Score**: **94.30%** (vs. RF 94.22%, AE 94.17%)

---

## Running Inference (Prediction)

You can run predictions on new, raw data using the saved models. 

### 1. Run Test Set Prediction (Autoencoder Baseline)
A verification script is provided to load the trained model and evaluate it against the unseen test set (`KDDTest+.txt`):
```bash
python src/predict.py
```
* **Output**: Prints the classification report in the terminal and saves a confusion matrix plot to `logs/plots/autoencoder_test_confusion_matrix.png`.

### 2. Run Test Set Prediction (Hybrid Cascade)
You can evaluate the hybrid sequential cascade model on the unseen test set (`KDDTest+.txt`):
```bash
python src/hybrid_predict.py
```
* **Output**: Prints the hybrid cascade metrics, breakdown of decision paths, and saves plots to `logs/plots/hybrid_confusion_matrix.png` and `logs/plots/hybrid_decisions_distribution.png`.

### 3. Integrate into Python Code
To predict anomalies using either detector in your own codebase, import and use their respective classes:

#### Using Autoencoder only:
```python
import pandas as pd
from src.predict import AnomalyDetector

detector = AnomalyDetector(models_dir="models")
new_data = pd.read_csv("path/to/network_traffic.csv")
predictions, errors = detector.predict(new_data)
```

#### Using Hybrid Cascade (Recommended):
```python
import pandas as pd
from src.hybrid_predict import HybridAnomalyDetector

# Initialize the hybrid cascade detector
detector = HybridAnomalyDetector(
    models_dir="models",
    threshold_low=0.05,
    threshold_high=0.50,
    rf_threshold=0.01
)

new_data = pd.read_csv("path/to/network_traffic.csv")

# predictions: 1 for anomalies, 0 for normal traffic
# errors: raw Mean Squared Error reconstruction errors
# sources: array indicating which model made the decision ('AE_low', 'AE_high', 'RF_normal', 'RF_anomaly')
predictions, errors, sources = detector.predict(new_data)
```

---

## Secure Serialization

This project implements **secure model serialization**:
* Model weights and architecture are saved using Keras native `.keras` format (a secure zip archive of JSON configuration and weights).
* Scaler coefficients (`mean`, `variance`, `scale`, `n_samples_seen`) are serialized to **`scaler.json`**. This completely avoids using Python `pickle` (`.pkl`) for the Autoencoder, removing any threat of insecure deserialization/arbitrary code execution vulnerabilities in production environments.
* The Random Forest pipeline is serialized to **`random_forest.joblib`** in the `models/` directory (excluded from git tracking via `.gitignore`).
