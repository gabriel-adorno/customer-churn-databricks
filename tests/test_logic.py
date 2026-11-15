import pandas as pd
import pytest
from src.churn_project.logic import make_feature_frame, make_target, validate_columns, REQUIRED_COLUMNS

def sample():
    data = {c: [1, 0] for c in REQUIRED_COLUMNS}
    data["churn"] = [0, 1]
    data["status"] = [1, 0]
    return pd.DataFrame(data)

def test_validate_columns_accepts_required_columns():
    validate_columns(sample().columns)

def test_validate_columns_reports_missing_column():
    with pytest.raises(ValueError):
        validate_columns(["churn"])

def test_target_is_integer():
    assert make_target(sample()).tolist() == [0, 1]

def test_features_exclude_target_and_status():
    features = make_feature_frame(sample())
    assert "churn" not in features.columns
    assert "status" not in features.columns
