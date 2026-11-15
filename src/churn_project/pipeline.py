
import os
import re
import sys
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Permite localizar o pacote churn_project dentro de src/
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from churn_project.logic import make_feature_frame, make_target


# --------------------------------------------------
# CONFIGURAÇÕES
# --------------------------------------------------

UCI_DATA_URL = os.getenv(
    "UCI_DATA_URL",
    "https://archive.ics.uci.edu/static/public/563/data.csv",
)

CATALOG = os.getenv("CHURN_CATALOG", "churn_demo")
SCHEMA = os.getenv("CHURN_SCHEMA", "churn_demo")

EXPERIMENT_NAME = "/Shared/customer-churn-prediction"


# --------------------------------------------------
# FUNÇÕES AUXILIARES
# --------------------------------------------------

def normalize_columns(columns):
    return [
        re.sub(
            r"[^a-z0-9]+",
            "_",
            str(column).strip().lower()
        ).strip("_")
        for column in columns
    ]


def validate_data(pdf):
    if pdf.empty:
        raise ValueError("O dataset está vazio.")

    if "churn" not in pdf.columns:
        raise ValueError(
            "A coluna alvo 'churn' não foi encontrada no dataset."
        )

    if pdf["churn"].nunique() < 2:
        raise ValueError(
            "A variável churn precisa possuir pelo menos duas classes."
        )


# --------------------------------------------------
# PIPELINE PRINCIPAL
# --------------------------------------------------

def main():

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    print("Iniciando pipeline de Customer Churn...")

    # 1. EXTRAÇÃO DOS DADOS

    print("Baixando dataset da UCI...")

    pdf = pd.read_csv(UCI_DATA_URL)

    pdf.columns = normalize_columns(pdf.columns)

    validate_data(pdf)

    pdf["customer_id"] = [
        f"uci_{i:05d}"
        for i in range(1, len(pdf) + 1)
    ]

    print(f"Dataset carregado: {len(pdf)} registros.")

    # 2. CAMADA BRONZE

    print("Gravando dados na camada Bronze...")

    spark.sql(
        f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}"
    )

    spark.createDataFrame(pdf).write \
        .mode("overwrite") \
        .format("delta") \
        .saveAsTable(
            f"{CATALOG}.{SCHEMA}.bronze_customers"
        )

    # 3. PREPARAÇÃO DOS DADOS

    print("Preparando features e variável alvo...")

    X = make_feature_frame(pdf)
    y = make_target(pdf)

    numeric = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical = [
        column
        for column in X.columns
        if column not in numeric
    ]

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

    # 4. DIVISÃO TREINO E TESTE

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # 5. EXPERIMENTOS COM MLflow

    print("Iniciando experimentos com MLflow...")

    mlflow.set_experiment(EXPERIMENT_NAME)

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced",
        ),
    }

    best_model = None
    best_f1 = -1
    best_model_name = None

    for name, estimator in candidates.items():

        print(f"Treinando modelo: {name}")

        model = Pipeline([
            ("preprocess", preprocess),
            ("model", estimator),
        ])

        with mlflow.start_run(run_name=name):

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            probabilities = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(
                    y_test, predictions
                ),
                "precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "f1": f1_score(
                    y_test, predictions, zero_division=0
                ),
                "roc_auc": roc_auc_score(
                    y_test, probabilities
                ),
            }

            mlflow.log_params({
                "model": name,
                "test_size": 0.25,
                "random_state": 42,
            })

            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
            )

            print(f"Métricas de {name}: {metrics}")

            if metrics["f1"] > best_f1:

                best_model = model
                best_f1 = metrics["f1"]
                best_model_name = name

    if best_model is None:
        raise RuntimeError("Nenhum modelo foi treinado com sucesso.")

    print(f"Melhor modelo: {best_model_name}")
    print(f"Melhor F1: {best_f1:.4f}")

    # 6. GERAÇÃO DAS PREDIÇÕES

    print("Gerando previsões para os clientes...")

    output = pdf[["customer_id"]].copy()

    output["churn_prediction"] = best_model.predict(X)

    output["churn_probability"] = (
        best_model.predict_proba(X)[:, 1]
    )

    output["risk_level"] = pd.cut(
        output["churn_probability"],
        bins=[-0.001, 0.35, 0.65, 1.0],
        labels=["low", "medium", "high"],
    ).astype(str)

    # 7. CAMADA GOLD

    print("Gravando previsões na camada Gold...")

    spark.createDataFrame(output).write \
        .mode("overwrite") \
        .format("delta") \
        .saveAsTable(
            f"{CATALOG}.{SCHEMA}.gold_churn_predictions"
        )

    # 8. RESULTADO FINAL

    print("------------------------------------")
    print("PIPELINE CONCLUÍDO COM SUCESSO")
    print("------------------------------------")
    print(f"Registros processados: {len(pdf)}")
    print(f"Melhor modelo: {best_model_name}")
    print(f"F1-score: {best_f1:.4f}")
    print(f"Tabela Bronze: {CATALOG}.{SCHEMA}.bronze_customers")
    print(f"Tabela Gold: {CATALOG}.{SCHEMA}.gold_churn_predictions")


if __name__ == "__main__":
    main()
