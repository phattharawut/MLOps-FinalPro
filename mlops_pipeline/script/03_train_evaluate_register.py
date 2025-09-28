import json
import os
import sys
from typing import Any, Dict

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


MODELS_AND_GRIDS: Dict[str, Any] = {
    "svm_rbf": (SVC(probability=True), {"C": [0.5, 1, 2], "gamma": ["scale", "auto"]}),
    "logreg": (LogisticRegression(max_iter=2000), {"C": [0.5, 1, 2]}),
    "rf": (RandomForestClassifier(random_state=42), {"n_estimators": [200, 400], "max_depth": [None, 10, 20]}),
    "gbm": (GradientBoostingClassifier(random_state=42), {"n_estimators": [150, 300], "learning_rate": [0.05, 0.1]}),
    "knn": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 11]})
}


DEF_EXPERIMENT = "DryBeans - Model Training"


def _load_artifacts_from_preprocessing_run(run_id: str):
    local_proc = download_artifacts(run_id=run_id, artifact_path="processed_data")
    local_trans = download_artifacts(run_id=run_id, artifact_path="transformers")

    train_df = pd.read_csv(os.path.join(local_proc, "train.csv"))
    test_df = pd.read_csv(os.path.join(local_proc, "test.csv"))

    feature_transformer = joblib.load(os.path.join(local_trans, "feature_transformer.pkl"))
    label_encoder_obj = joblib.load(os.path.join(local_trans, "label_encoder.pkl"))
    return train_df, test_df, feature_transformer, label_encoder_obj


def _plot_and_log_confusion(cm: np.ndarray, classes: list, artifact_dir="eval_artifacts"):
    os.makedirs(artifact_dir, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    path = os.path.join(artifact_dir, "confusion_matrix.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

    mlflow.log_artifacts(artifact_dir, artifact_path="evaluation")


def train_evaluate_register(preprocessing_run_id: str, model_registry_name: str = "DryBeans-Classifier"):
    mlflow.set_experiment(DEF_EXPERIMENT)

    with mlflow.start_run(run_name=f"gridsearch_models_from_{preprocessing_run_id}"):
        mlflow.set_tag("ml.step", "model_training_evaluation")
        mlflow.log_param("preprocessing_run_id", preprocessing_run_id)

        train_df, test_df, feature_transformer, label_encoder_obj = _load_artifacts_from_preprocessing_run(preprocessing_run_id)

        label_col = label_encoder_obj.get("label_col", "Class")
        classes_order = label_encoder_obj.get("classes_", [])

        X_train = train_df.drop(columns=[label_col])
        y_train = train_df[label_col]
        X_test = test_df.drop(columns=[label_col])
        y_test = test_df[label_col]

        # Grid search across candidate models with CV=5 using weighted F1
        best_model_name, best_estimator, best_score, best_params = None, None, -np.inf, None
        cv_results_summary = {}

        for name, (estimator, grid) in MODELS_AND_GRIDS.items():
            clf = GridSearchCV(estimator, grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            clf.fit(X_train, y_train)
            cv_results_summary[name] = {
                "best_score": float(clf.best_score_),
                "best_params": clf.best_params_
            }
            mlflow.log_param(f"{name}_best_params", json.dumps(clf.best_params_))
            mlflow.log_metric(f"{name}_cv_best_f1_weighted", float(clf.best_score_))
            if clf.best_score_ > best_score:
                best_score = clf.best_score_
                best_estimator = clf.best_estimator_
                best_model_name = name
                best_params = clf.best_params_

        # Fit best on full train
        best_estimator.fit(X_train, y_train)

        # Evaluate on test
        y_pred = best_estimator.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1w = f1_score(y_test, y_pred, average='weighted')
        precw = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recw = recall_score(y_test, y_pred, average='weighted')

        mlflow.log_metric("test_accuracy", float(acc))
        mlflow.log_metric("test_f1_weighted", float(f1w))
        mlflow.log_metric("test_precision_weighted", float(precw))
        mlflow.log_metric("test_recall_weighted", float(recw))

        # Save detailed report
        report_txt = classification_report(y_test, y_pred, target_names=[str(c) for c in classes_order])
        os.makedirs("eval_artifacts", exist_ok=True)
        with open("eval_artifacts/classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report_txt)

        cm = confusion_matrix(y_test, y_pred)
        _plot_and_log_confusion(cm, classes=[str(c) for c in classes_order])

        # Persist the whole serving bundle: model + transformers
        serving_bundle = {
            "model": best_estimator,
            "feature_transformer": feature_transformer,
            "label_encoder": label_encoder_obj,
        }
        os.makedirs("model_bundle", exist_ok=True)
        joblib.dump(serving_bundle, "model_bundle/serving_bundle.pkl")
        mlflow.log_artifacts("model_bundle", artifact_path="model_bundle")

        # Log model with signature
        signature = infer_signature(X_train, best_estimator.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=best_estimator,
            artifact_path="drybeans_model",
            signature=signature,
            registered_model_name=model_registry_name,
        )

        mlflow.log_param("selected_model", best_model_name)
        mlflow.log_param("selected_model_best_params", json.dumps(best_params))

        print("Training + evaluation complete. Selected:", best_model_name)
        print("CV summary:", json.dumps(cv_results_summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 03_train_evaluate_register.py <preprocessing_run_id> [registry_name]")
        sys.exit(1)
    run_id = sys.argv[1]
    registry_name = sys.argv[2] if len(sys.argv) > 2 else "DryBeans-Classifier"
    train_evaluate_register(run_id, registry_name)