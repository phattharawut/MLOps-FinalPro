# =============================================
# 01_data_validation.py  — Dry Beans
# =============================================
import json
import os
import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest



def _basic_stats(df: pd.DataFrame, label_col: str):
    stats = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "label_distribution": df[label_col].value_counts().to_dict() if label_col in df.columns else {},
    }
    # numeric column summary
    num_cols = df.select_dtypes(include=[np.number]).columns
    desc = df[num_cols].describe().T
    stats["numeric_summary"] = desc.to_dict()
    return stats


def _infer_schema(df: pd.DataFrame):
    schema = []
    for col in df.columns:
        col_schema = {
            "name": col,
            "dtype": str(df[col].dtype),
            "nullable": bool(df[col].isnull().any()),
        }
        if np.issubdtype(df[col].dtype, np.number):
            col_schema.update({
                "min": float(df[col].min(skipna=True)),
                "max": float(df[col].max(skipna=True)),
                "mean": float(df[col].mean(skipna=True)),
                "std": float(df[col].std(skipna=True)),
            })
        schema.append(col_schema)
    return {"fields": schema}


def _detect_anomalies(df: pd.DataFrame, label_col: str, random_state: int = 42):
    """
    Use IsolationForest on numeric features to flag anomalies.
    Returns anomaly indices and anomaly score per row.
    """
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.empty:
        return pd.DataFrame(index=df.index)

    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=random_state)
    preds = iso.fit_predict(num_df)
    scores = iso.decision_function(num_df)
    out = pd.DataFrame({"anomaly": (preds == -1).astype(int), "anomaly_score": scores}, index=df.index)
    if label_col in df.columns:
        out[label_col] = df[label_col]
    return out


def validate_data(
    data_path: str = "Dry_Bean_Dataset.xlsx",
    sheet_name: str = None,
    label_col: str = "Class",
    experiment_name: str = "DryBeans - Data Validation",
):
    """Load Dry Beans dataset from Excel, compute stats, schema, detect anomalies, and log to MLflow."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("ml.step", "data_validation")
        mlflow.log_param("data_path", os.path.abspath(data_path))
        if sheet_name is not None:
            mlflow.log_param("sheet_name", sheet_name)

        # 1) Load
        xls = pd.ExcelFile(data_path)
        print("Available sheets:", xls.sheet_names)
        df = pd.read_excel(xls, sheet_name=sheet_name or xls.sheet_names[0])
        # 2) Basic checks
        num_rows, num_cols = df.shape
        missing_values = int(df.isnull().sum().sum())
        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("num_cols", num_cols)
        mlflow.log_metric("missing_values", missing_values)

        # 3) Stats & schema
        stats = _basic_stats(df, label_col)
        schema = _infer_schema(df)

        os.makedirs("validation_artifacts", exist_ok=True)
        with open("validation_artifacts/data_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        with open("validation_artifacts/data_schema.json", "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)

        # 4) Anomaly detection
        anomalies = _detect_anomalies(df, label_col)
        anomalies_path = "validation_artifacts/anomalies.csv"
        anomalies.to_csv(anomalies_path, index=True)
        mlflow.log_metric("anomaly_count", int(anomalies["anomaly"].sum()) if "anomaly" in anomalies.columns else 0)

        # 5) Validation status
        validation_status = "Success" if missing_values == 0 else "HasMissing"
        mlflow.log_param("validation_status", validation_status)

        # 6) Log artifacts
        mlflow.log_artifacts("validation_artifacts", artifact_path="data_validation")

        print("Validation completed. Run ID:", run_id)
        if os.getenv("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"validation_run_id={run_id}", file=f)


if __name__ == "__main__":
    validate_data()
