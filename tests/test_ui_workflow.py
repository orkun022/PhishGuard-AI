"""Exercise the interface against temporary model artifacts, not a user's files."""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.ui import csv_bytes, empty_state


@pytest.fixture
def workspace(tmp_path):
    source = (ROOT / "app/app.py").read_text()
    source = source.replace(
        "PROJECT_ROOT = Path(__file__).resolve().parents[1]",
        f"PROJECT_ROOT = Path({str(tmp_path)!r})",
        1,
    )
    return tmp_path, source


def install_test_model(folder):
    model = DummyClassifier(strategy="constant", constant=1).fit(
        np.zeros((4, 16)), np.array([0, 1, 0, 1])
    )
    metrics = {
        "accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1": 2 / 3,
        "cv_mean": 2 / 3, "cv_std": 0.0,
        "confusion_matrix": [[0, 2], [0, 2]],
    }
    metadata = {
        "selected_model": "Test model", "dataset_kind": "synthetic_demo",
        "train_samples": 4, "test_samples": 4,
        "results": {"Test model": metrics},
    }
    (folder / "models").mkdir()
    joblib.dump({"model": model, "metadata": metadata}, folder / "models/model_bundle.joblib")


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_empty_workspace_does_not_train_or_crash(workspace):
    folder, source = workspace
    app = AppTest.from_string(source).run(timeout=15)
    assert not app.exception
    assert button(app, "Analyze address").disabled
    assert not (folder / "models").exists()
    for page in ["Batch review", "Model performance", "Scan history"]:
        app.radio[0].set_value(page).run(timeout=15)
        assert not app.exception


def test_scan_navigation_theme_and_clear_history(workspace):
    folder, source = workspace
    install_test_model(folder)
    app = AppTest.from_string(source).run(timeout=15)
    assert not app.exception
    app.text_input[0].set_value("https://example.com")
    button(app, "Analyze address").click().run(timeout=15)
    assert not app.exception
    assert app.session_state["scan_count"] == 1
    assert app.session_state["flagged_count"] == 1
    assert len(app.session_state["history"]) == 1
    assert app.session_state["last_result"]["url"] == "https://example.com"
    for page in ["Model performance", "Batch review", "Scan history"]:
        app.radio[0].set_value(page).run(timeout=15)
        assert not app.exception
    app.toggle[0].set_value(True).run(timeout=15)
    assert not app.exception
    assert len(app.session_state["history"]) == 1
    button(app, "Clear session history").click().run(timeout=15)
    assert not app.exception
    assert "history" not in app.session_state


def test_invalid_url_does_not_create_scan(workspace):
    folder, source = workspace
    install_test_model(folder)
    app = AppTest.from_string(source).run(timeout=15)
    app.text_input[0].set_value("javascript:alert(1)")
    button(app, "Analyze address").click().run(timeout=15)
    assert not app.exception
    assert len(app.error) == 1
    assert "last_result" not in app.session_state
    assert "history" not in app.session_state


def test_downloads_neutralize_formula_prefixes_and_cards_escape_text():
    exported = csv_bytes(pd.DataFrame({"url": ["\t=1+1", "https://example.com"]})).decode("utf-8-sig")
    assert "'\t=1+1" in exported
    assert "https://example.com" in exported
    card = empty_state('<img src=x onerror="bad">', "<script>bad</script>")
    assert "<img" not in card
    assert "<script>" not in card
