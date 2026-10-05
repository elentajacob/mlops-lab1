import json
import pytest
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from unittest.mock import patch, MagicMock
from src.train_and_save_model import download_data, dataset_statistics, preprocess_data
from src.train_and_save_model import train_model, evaluate_model
from src.train_and_save_model import get_model_version, update_model_version
from src.train_and_save_model import ensure_folder_exists, save_model_to_gcs, save_metrics_to_gcs


# ----------------- Test Download ----------------- #
def test_download_data():
    X, y = download_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (569, 30)            # Breast Cancer dataset shape
    assert set(y.unique()) == {0, 1}       # binary target
    assert X.shape[0] == y.shape[0]


# ----------------- Test Dataset Statistics ----------------- #
def test_dataset_statistics():
    X, y = download_data()
    stats = dataset_statistics(X, y)
    assert stats["n_samples"] == 569
    assert stats["n_features"] == 30
    assert stats["missing_values"] == 0
    assert sum(stats["class_distribution"].values()) == 569
    assert 0 < stats["class_balance_ratio"] <= 1
    assert len(stats["top_feature_means"]) == 5
    json.dumps(stats)                      # must be JSON-serializable for GCS upload


# ----------------- Test Preprocess ----------------- #
def test_preprocess_data():
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]
    assert X_train.shape[1] == X.shape[1]
    # Stratification: class ratio in train and test should be close
    assert abs(y_train.mean() - y_test.mean()) < 0.02


# ----------------- Test Train model ----------------- #
def test_train_model():
    X, y = download_data()
    X_train, _, y_train, _ = preprocess_data(X, y)
    model = train_model(X_train, y_train)
    assert isinstance(model, GradientBoostingClassifier)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


# ----------------- Test Evaluate model ----------------- #
def test_evaluate_model():
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix"]:
        assert key in metrics
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        assert 0.0 <= metrics[key] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == len(y_test)
    assert metrics["accuracy"] > 0.85      # sanity floor for this dataset
    assert "cv_f1_mean" not in metrics     # CV only runs when train data is passed

    metrics_cv = evaluate_model(model, X_test, y_test, X_train, y_train)
    assert 0.0 <= metrics_cv["cv_f1_mean"] <= 1.0
    assert metrics_cv["cv_f1_std"] >= 0.0


# ----------------- Test Model versioning ----------------- #
def test_get_model_version():
    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        bucket_name = "bucket-test"
        version_file_name = "version.txt"

        # Version file exists (return value set BEFORE the call)
        mock_blob.exists.return_value = True
        mock_blob.download_as_text.return_value = '3'
        version = get_model_version(bucket_name, version_file_name)

        assert version == 3
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.download_as_text.assert_called_once()

        mock_storage_client.reset_mock()
        mock_bucket.reset_mock()
        mock_blob.reset_mock()

        # Version file does not exist
        mock_blob.exists.return_value = False
        version = get_model_version(bucket_name, version_file_name)

        assert version == 0
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.download_as_text.assert_not_called()


# ----------------- Test Update Model version ----------------- #
def test_update_model_version():
    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        bucket_name = 'bucket-test'
        version_file_name = 'version.txt'
        new_version = 2

        result = update_model_version(bucket_name, version_file_name, new_version)
        assert result is True
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.upload_from_string.assert_called_once_with(str(new_version))

        mock_storage_client.reset_mock()
        mock_bucket.reset_mock()
        mock_blob.reset_mock()

        with pytest.raises(ValueError):
            update_model_version(bucket_name, version_file_name, 'invalid_version')

        mock_blob.upload_from_string.side_effect = Exception("Upload failed")
        result = update_model_version(bucket_name, version_file_name, new_version)
        assert result is False
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.upload_from_string.assert_called_once_with(str(new_version))


# ----------------- Test Ensure Folder Exists ----------------- #
def test_ensure_folder_exists():
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    folder_name = "trained_models"

    mock_blob.exists.return_value = False
    ensure_folder_exists(mock_bucket, folder_name)
    mock_bucket.blob.assert_called_with(f"{folder_name}/")
    mock_blob.upload_from_string.assert_called_once_with('')

    mock_blob.reset_mock()

    mock_blob.exists.return_value = True
    ensure_folder_exists(mock_bucket, folder_name)
    mock_bucket.blob.assert_called_with(f"{folder_name}/")
    mock_blob.upload_from_string.assert_not_called()


# ----------------- Test Save model to GCS ----------------- #
def test_save_model_to_gcs():
    model = GradientBoostingClassifier()
    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = False

        save_model_to_gcs(model, 'bucket-test', 'blob-test')

        mock_storage_client.assert_called_once()
        mock_storage_client.return_value.bucket.assert_called_once_with('bucket-test')
        assert mock_bucket.blob.call_count == 2
        mock_bucket.blob.assert_any_call('trained_models/')
        mock_bucket.blob.assert_any_call('blob-test')
        mock_blob.upload_from_filename.assert_called_once_with('model.joblib')


# ----------------- Test Save metrics to GCS ----------------- #
def test_save_metrics_to_gcs():
    metrics = {"accuracy": 0.95, "f1_score": 0.96}
    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        save_metrics_to_gcs(metrics, 'bucket-test', 'trained_models/metrics.json')

        mock_storage_client.return_value.bucket.assert_called_once_with('bucket-test')
        mock_bucket.blob.assert_called_once_with('trained_models/metrics.json')
        args, kwargs = mock_blob.upload_from_string.call_args
        assert json.loads(args[0]) == metrics
        assert kwargs["content_type"] == "application/json"
