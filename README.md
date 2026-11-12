# PhishGuard-AI

A Python workspace for phishing URL classification: 16 structural features, reproducible model comparison, a hostname-separated holdout, and a Streamlit interface for single and batch predictions.

## Quick start

Python 3.10 or later:

```sh
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.generate_dataset
python -m src.train --cv-folds 3 --dataset-kind synthetic_demo
streamlit run app/app.py
```

The generator creates **synthetic demonstration data**, not a verified real-world corpus. You can also launch the app without a model and explicitly choose **Train a synthetic demo**. Startup does not silently train or overwrite your raw dataset.

## Train with your own data

Provide a CSV with `url,label`. Accepted labels: 0/1, legitimate/phishing, benign/malicious, or safe/phishing. Invalid/missing labels and conflicting labels for the same normalized URL are rejected. Duplicate URLs are removed.

```sh
python -m src.train --data path/to/reviewed_urls.csv --dataset-kind user_supplied
python -m src.predict --url https://example.com
python -m src.predict --model models/random_forest.pkl --url https://example.com
```

Do not treat URL generators, heuristic rules, or a reputation-service no-match as ground-truth benign labels.

## Training and evaluation

- Compare Logistic Regression, Random Forest, SVM and XGBoost (when installed).
- Split by normalized hostname (including normalization of a leading `www.`) so the same hostname does not appear in training and holdout.
- Fit scaling inside each model pipeline and each cross-validation fold.
- Use grouped training CV F1 to select the model. Holdout metrics do **not** select it.
- Save the complete prediction pipeline and evaluation metadata in `models/model_bundle.joblib`, plus `models/best_model.pkl` and individual models.
- Write measured accuracy, precision, recall, F1, ROC-AUC and confusion matrices to `reports/metrics.json`.

The hostname grouping is not a full public-suffix/registered-domain split: related subdomains or repeated generator templates can remain correlated. The random split is not a future-time evaluation. Use independent reviewed sources, registered-domain or temporal holdouts, and calibration before making real-world claims.

## Interface

The Streamlit dashboard uses a sidebar to separate **URL analysis**, **Batch review**, **Model performance**, and **Scan history**. It includes light/dark appearance, session summary cards, URL feature badges, an explicit model score, and model comparison charts.

Single-URL reviews include all 16 features. Batch review supports up to 1,000 rows / 2 MB, per-row validation errors, a CSV template, progress feedback, and CSV export. Session history holds the last 20 single-URL reviews and can be exported or cleared. Model performance includes holdout F1/recall charts, detailed metrics, a confusion matrix, and evaluation JSON download. Model scores are explicitly uncalibrated and do not prove safety.

Predictions do not open the submitted address. An **optional, explicit** Google Safe Browsing lookup sends the selected URL to Google using your API key. A no-match or lookup failure remains unknown. Automatic feedback retraining is disabled; use reviewed labeled data for explicit training.

The separate [browser-only PhishGuard app](https://github.com/orkun022/phishguard) uses local rules and does not run these models.

## Reproduced demo results

`reports/demo-metrics.json` records one synthetic demonstration run, including split sizes, model selection, seed and measured results. It is an example of the workflow, **not a real-world detection benchmark or an accuracy guarantee**. Reruns with different package versions can differ.

## Checks

```sh
pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q src app
```

Tests cover feature extraction, hostname boundaries, IPv6/credential hosts, input validation, label validation, disjoint holdout hostnames and non-authoritative reputation results. GitHub Actions runs the checks.

## Repository map

- `src/`: feature extraction, validated input, training, prediction and optional reputation checks.
- `app/app.py`: Streamlit workspace.
- `tests/`: regression checks.
- `models/`: locally generated artifacts, excluded from source control.
- `reports/`: measured JSON evaluations.

Only load model artifacts you created or otherwise trust; joblib files are executable Python serialization.
