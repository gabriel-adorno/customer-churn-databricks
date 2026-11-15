import os
import re
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_project.logic import make_feature_frame, make_target

UCI_DATA_URL = os.getenv(
    "UCI_DATA_URL",
    "https://archive.ics.uci.edu/static/public/563/data.csv"
)
CATALOG = os.getenv("CHURN_CATALOG", "workspace")
SCHEMA = os.getenv("CHURN_SCHEMA", "churn_demo")

def normalize_columns(columns):
    return [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in columns]

def main():
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()

    pdf = pd.read_csv(UCI_DATA_URL)
    pdf.columns = normalize_columns(pdf.columns)
    pdf["customer_id"] = [f"uci_{i:05d}" for i in range(1, len(pdf) + 1)]

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
    spark.createDataFrame(pdf).write.mode("overwrite").format("delta").saveAsTable(
        f"{CATALOG}.{SCHEMA}.bronze_customers"
    )

    X = make_feature_frame(pdf)
    y = make_target(pdf)
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    mlflow.set_experiment(f"/Shared/customer-churn-prediction")
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced"
        ),
    }

    best_model, best_f1 = None, -1
    for name, estimator in candidates.items():
        model = Pipeline([("preprocess", preprocess), ("model", estimator)])
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            metrics = {
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, proba),
            }
            mlflow.log_params({"model": name, "test_size": 0.25, "random_state": 42})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")
            if metrics["f1"] > best_f1:
                best_model, best_f1 = model, metrics["f1"]

    output = pdf[["customer_id"]].copy()
    output["churn_prediction"] = best_model.predict(X)
    output["churn_probability"] = best_model.predict_proba(X)[:, 1]
    output["risk_level"] = pd.cut(
        output["churn_probability"],
        bins=[-0.001, 0.35, 0.65, 1.0],
        labels=["low", "medium", "high"]
    ).astype(str)

    spark.createDataFrame(output).write.mode("overwrite").format("delta").saveAsTable(
        f"{CATALOG}.{SCHEMA}.gold_churn_predictions"
    )
    print(f"Pipeline concluído. Melhor F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()
