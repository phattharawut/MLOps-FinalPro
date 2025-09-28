import importlib.util
import os
from pathlib import Path

import pandas as pd

CANDIDATE_PATHS = [
    "02_data_preprocessing.py",
    "mlops_pipeline/02_data_preprocessing.py",
    "script/02_data_preprocessing.py",
]

def resolve_preprocess_path() -> str:
    # Prefer GitHub workspace when present; otherwise use repo root heuristic.
    repo_root = os.getenv("GITHUB_WORKSPACE", os.getcwd())
    for rel in CANDIDATE_PATHS:
        p = Path(repo_root) / rel
        if p.exists():
            return str(p.resolve())
    # fallback: search recursively (last resort)
    for p in Path(repo_root).rglob("02_data_preprocessing.py"):
        return str(p.resolve())
    raise FileNotFoundError("Cannot locate 02_data_preprocessing.py in repo. Checked: " + ", ".join(CANDIDATE_PATHS))

def load_module_func(py_path: str, func_name: str):
    spec = importlib.util.spec_from_file_location("preprocess_module", py_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "Invalid module spec for %s" % py_path
    spec.loader.exec_module(module)
    fn = getattr(module, func_name)
    return fn

def test_preprocess_creates_artifacts(tmp_path):
    # Arrange: create a tiny Excel dataset with numeric features + 'Class' label
    df = pd.DataFrame({
        "Feature1": [1.0, 2.0, 3.0, 4.0],
        "Feature2": [0.5, 0.1, 0.3, 0.2],
        "Class": ["A", "A", "B", "B"],
    })
    xlsx_path = tmp_path / "drybeans_dummy.xlsx"
    df.to_excel(xlsx_path, index=False)

    # Resolve path to the preprocessing script in the repo
    preproc_path = resolve_preprocess_path()
    preprocess_data = load_module_func(preproc_path, "preprocess_data")

    # Act within tmp_path so outputs don't pollute repo
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        preprocess_data(
            data_path=str(xlsx_path),
            sheet_name=0,
            label_col="Class",
            test_size=0.5,
            random_state=42,
            experiment_name="CI Test Preprocessing"
        )
        # Assert: artifacts exist
        assert (tmp_path / "processed_data" / "train.csv").exists()
        assert (tmp_path / "processed_data" / "test.csv").exists()
        assert (tmp_path / "transformers" / "feature_transformer.pkl").exists()
        assert (tmp_path / "transformers" / "label_encoder.pkl").exists()

        # Basic content checks
        train_df = pd.read_csv(tmp_path / "processed_data" / "train.csv")
        assert "Class" in train_df.columns
        # Encoded labels should be integers 0..K-1
        assert pd.api.types.is_integer_dtype(train_df["Class"])
    finally:
        os.chdir(cwd)
