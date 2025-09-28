import os
import sys
import types
import pandas as pd
from importlib.machinery import SourceFileLoader

def load_module_func(py_path: str, func_name: str):
    mod_name = os.path.splitext(os.path.basename(py_path))[0].replace("-", "_")
    module = SourceFileLoader(mod_name, py_path).load_module()
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

    # Change working directory so outputs land under tmp_path
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        preprocess_data = load_module_func(os.path.abspath("02_data_preprocessing.py"), "preprocess_data")
        # Act
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
