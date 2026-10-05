import os
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split, cross_val_score

# Load environment variables from .env (local) or from GitHub Secrets (CI)
load_dotenv()
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
VERSION_FILE_NAME = os.getenv("VERSION_FILE_NAME")


# ----------------- Data ----------------- #
# Breast Cancer Wisconsin dataset: 569 samples, 30 numeric features, binary target
# (0 = malignant, 1 = benign)
def download_data():
    data = load_breast_cancer()
    features = pd.DataFrame(data.data, columns=data.feature_names)
    target = pd.Series(data.target, name="target")
    return features, target


# Descriptive statistics of the dataset before training
def dataset_statistics(X, y):
    class_counts = y.value_counts().sort_index()
    top_means = X.mean().sort_values(ascending=False).head(5)
    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "missing_values": int(X.isnull().sum().sum()),
        "class_distribution": {int(k): int(v) for k, v in class_counts.items()},
        "class_balance_ratio": round(float(class_counts.min() / class_counts.max()), 4),
        "top_feature_means": {str(k): round(float(v), 4) for k, v in top_means.items()},
    }


# Stratified split keeps the malignant/benign ratio the same in train and test
def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# ----------------- Model ----------------- #
def train_model(X_train, y_train):
    model = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42
    )
    model.fit(X_train, y_train)
    return model


# Full evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
# and (optionally) 5-fold cross-validated F1 on the training set
def evaluate_model(model, X_test, y_test, X_train=None, y_train=None):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_proba)), 4)
    if X_train is not None and y_train is not None:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
        metrics["cv_f1_mean"] = round(float(cv_scores.mean()), 4)
        metrics["cv_f1_std"] = round(float(cv_scores.std()), 4)
    return metrics


# ----------------- GCS: versioning ----------------- #
def get_model_version(bucket_name, version_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(version_file_name)
    if blob.exists():
        version = int(blob.download_as_text())
    else:
        version = 0
    return version


def update_model_version(bucket_name, version_file_name, version):
    if not isinstance(version, int):
        raise ValueError("Version must be an integer")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(version_file_name)
        blob.upload_from_string(str(version))
        return True
    except Exception as e:
        print(f"Error updating model version: {e}")
        return False


# ----------------- GCS: artifacts ----------------- #
def ensure_folder_exists(bucket, folder_name):
    blob = bucket.blob(f"{folder_name}/")
    if not blob.exists():
        blob.upload_from_string("")
        print(f"Created folder: {folder_name}")


def save_model_to_gcs(model, bucket_name, blob_name):
    joblib.dump(model, "model.joblib")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    ensure_folder_exists(bucket, "trained_models")
    blob = bucket.blob(blob_name)
    blob.upload_from_filename("model.joblib")


# Upload the metrics + dataset stats as a JSON file next to the model
def save_metrics_to_gcs(metrics, bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(metrics, indent=2), content_type="application/json")


# ----------------- Pipeline ----------------- #
def main():
    current_version = get_model_version(BUCKET_NAME, VERSION_FILE_NAME)
    new_version = current_version + 1

    X, y = download_data()
    stats = dataset_statistics(X, y)
    print("Dataset statistics:", json.dumps(stats, indent=2))

    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test, X_train, y_train)
    print("Evaluation metrics:", json.dumps(metrics, indent=2))
    print(classification_report(y_test, model.predict(X_test),
                                target_names=["malignant", "benign"]))

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_blob = f"trained_models/model_v{new_version}_{timestamp}.joblib"
    metrics_blob = f"trained_models/metrics_v{new_version}_{timestamp}.json"

    save_model_to_gcs(model, BUCKET_NAME, model_blob)
    print(f"Model saved to gs://{BUCKET_NAME}/{model_blob}")

    save_metrics_to_gcs({"version": new_version, "dataset": stats, "metrics": metrics},
                        BUCKET_NAME, metrics_blob)
    print(f"Metrics saved to gs://{BUCKET_NAME}/{metrics_blob}")

    if update_model_version(BUCKET_NAME, VERSION_FILE_NAME, new_version):
        print(f"Model version updated to {new_version}")
        print(f"MODEL_VERSION_OUTPUT: {new_version}")  # read by the GitHub Actions workflow
    else:
        print("Failed to update model version")


if __name__ == "__main__":
    main()
