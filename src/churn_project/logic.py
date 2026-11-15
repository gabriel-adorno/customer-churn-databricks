REQUIRED_COLUMNS = [
    "call_failure", "complains", "subscription_length", "charge_amount",
    "seconds_of_use", "frequency_of_use", "frequency_of_sms",
    "distinct_called_numbers", "age_group", "tariff_plan", "age",
    "customer_value", "churn",
]

def validate_columns(columns):
    missing = sorted(set(REQUIRED_COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {missing}")

def make_feature_frame(df):
    validate_columns(df.columns)
    # `status` é deliberadamente excluída para reduzir risco de target leakage.
    return df.drop(columns=["churn", "status"], errors="ignore")

def make_target(df):
    validate_columns(df.columns)
    return df["churn"].astype(int)
