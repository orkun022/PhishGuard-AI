"""Reproducible model comparison: CV selects the model; holdout only reports."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing import prepare_data
from src.feature_extraction import FEATURE_NAMES

def get_models():
    estimators = {
        'Logistic Regression': LogisticRegression(max_iter=1500, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight='balanced'),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
    }
    try:
        from xgboost import XGBClassifier
        estimators['XGBoost'] = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=2, eval_metric='logloss')
    except ImportError:
        pass
    return {name: Pipeline([('scale', StandardScaler()), ('classifier', model)]) for name, model in estimators.items()}

def train_and_evaluate(X_train, X_test, y_train, y_test, models=None, cv_folds=5, groups_train=None, dataset_kind='user_supplied'):
    models = get_models() if models is None else models
    if not models:
        raise ValueError('At least one model is required.')
    if cv_folds < 2:
        raise ValueError('Cross-validation requires at least two folds.')
    counts = np.bincount(y_train, minlength=2)
    folds = min(cv_folds, int(counts.min()))
    if groups_train is not None:
        folds = min(folds, *(len(set(groups_train[y_train == c])) for c in (0, 1)))
    if folds < 2:
        raise ValueError('Need at least two training groups per class for cross-validation.')
    cv = StratifiedGroupKFold(folds, shuffle=True, random_state=42) if groups_train is not None else StratifiedKFold(folds, shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, groups=groups_train, cv=cv, scoring='f1', error_score='raise')
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
        results[name] = {
            'accuracy': float(accuracy_score(y_test, predicted)),
            'precision': float(precision_score(y_test, predicted, zero_division=0)),
            'recall': float(recall_score(y_test, predicted, zero_division=0)),
            'f1': float(f1_score(y_test, predicted, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, probability)),
            'cv_mean': float(scores.mean()), 'cv_std': float(scores.std()),
            'confusion_matrix': confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
        }
    selected = max(results, key=lambda name: results[name]['cv_mean'])
    metadata = {'selected_model': selected, 'selection_metric': 'training CV F1', 'holdout_split': 'hostname groups', 'dataset_kind': dataset_kind, 'train_samples': len(y_train), 'test_samples': len(y_test), 'cv_folds': folds, 'random_state': 42, 'trained_at': datetime.now(timezone.utc).isoformat(), 'results': results}
    models_dir = PROJECT_ROOT / 'models'
    models_dir.mkdir(exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, models_dir / (name.lower().replace(' ', '_') + '.pkl'))
    joblib.dump(models[selected], models_dir / 'best_model.pkl')
    joblib.dump({'model': models[selected], 'feature_names': FEATURE_NAMES, 'metadata': metadata}, models_dir / 'model_bundle.joblib')
    # Scaling lives inside each model pipeline, including during each CV fold.
    (models_dir / 'scaler.pkl').unlink(missing_ok=True)
    report = PROJECT_ROOT / 'reports/metrics.json'
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    return results

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data')
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--dataset-kind', choices=['synthetic_demo', 'user_supplied'], default='user_supplied')
    args = parser.parse_args(argv)
    X_train, X_test, y_train, y_test, _, groups = prepare_data(args.data, return_groups=True)
    return train_and_evaluate(X_train, X_test, y_train, y_test, cv_folds=args.cv_folds, groups_train=groups, dataset_kind=args.dataset_kind)

if __name__ == '__main__':
    main()
