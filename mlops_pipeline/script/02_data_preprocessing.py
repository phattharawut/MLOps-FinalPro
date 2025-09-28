import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import mlflow


def preprocess_data(
    data_path: str = "Dry_Bean_Dataset.xlsx",
    sheet_name: str = None,
    label_col: str = "Class",
    test_size: float = 0.25,
    random_state: int = 42,
    experiment_name: str = "DryBeans - Data Preprocessing",
):
    """Split train/test; fit transformers on train; persist transformers + splits as MLflow artifacts.
    Ensures same transformation is applicable to new/serving data to avoid training-serving skew.
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("ml.step", "data_preprocessing")
        mlflow.log_param("data_path", os.path.abspath(data_path))
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        xls = pd.ExcelFile(data_path)
        print("Available sheets:", xls.sheet_names)
        df = pd.read_excel(xls, sheet_name=sheet_name or xls.sheet_names[0])

        # Split X, y
        X = df.drop(columns=[label_col])
        y = df[label_col]

        # Identify numeric columns
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        mlflow.log_param("num_feature_count", len(num_cols))

        # Train/test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Fit transformers on TRAIN only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[num_cols])
        X_test_scaled = scaler.transform(X_test[num_cols])

        # Encode labels (fit on train only)
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # Save processed splits
        os.makedirs("processed_data", exist_ok=True)
        train_proc = pd.DataFrame(X_train_scaled, columns=num_cols)
        train_proc[label_col] = y_train_enc
        test_proc = pd.DataFrame(X_test_scaled, columns=num_cols)
        test_proc[label_col] = y_test_enc
        train_proc.to_csv("processed_data/train.csv", index=False)
        test_proc.to_csv("processed_data/test.csv", index=False)

        # Persist transformers for serving (avoid skew)
        os.makedirs("transformers", exist_ok=True)
        joblib.dump({"scaler": scaler, "num_cols": num_cols}, "transformers/feature_transformer.pkl")
        joblib.dump({"label_encoder": le, "classes_": le.classes_.tolist(), "label_col": label_col}, "transformers/label_encoder.pkl")

        # Log artifacts
        mlflow.log_artifacts("processed_data", artifact_path="processed_data")
        mlflow.log_artifacts("transformers", artifact_path="transformers")

        # Log simple metrics
        mlflow.log_metric("training_set_rows", int(len(X_train)))
        mlflow.log_metric("test_set_rows", int(len(X_test)))

        print("Preprocessing completed. Run ID:", run_id)
        print("Label classes order:", list(le.classes_))
        if os.getenv("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"preprocessing_run_id={run_id}", file=f)


if __name__ == "__main__":
    preprocess_data()