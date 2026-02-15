import os
import sys
import time
import warnings
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import prepare_data
from src.feature_extraction import FEATURE_NAMES
from src.utils import (
    plot_confusion_matrix, plot_roc_curve,
    plot_feature_importance, plot_model_comparison,
    print_classification_report
)


def get_models():
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1,
        ),
        'SVM': SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42),
    }

    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            use_label_encoder=False, eval_metric='logloss',
        )
    except ImportError:
        print("[!] XGBoost yuklu degil, atlaniyor.")

    return models


def train_and_evaluate(X_train, X_test, y_train, y_test, models=None, cv_folds=5):
    if models is None:
        models = get_models()

    results = {}
    best_model = None
    best_f1 = 0
    best_model_name = ''

    print("\n" + "=" * 60)
    print("MODEL EGITIMI")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n--- {name} ---")

        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='f1')

        results[name] = {
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
            'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
            'train_time': train_time,
        }

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  CV F1:     {cv_scores.mean():.4f}")

        print_classification_report(y_test, y_pred, name)
        plot_confusion_matrix(y_test, y_pred, name)
        if y_proba is not None:
            plot_roc_curve(y_test, y_proba, name)
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(model.feature_importances_, FEATURE_NAMES, name)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    plot_model_comparison(results)

    print(f"\nEn iyi model: {best_model_name} (F1: {best_f1:.4f})")

    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(best_model, os.path.join(models_dir, 'best_model.pkl'))

    for name, model in models.items():
        model_filename = name.lower().replace(' ', '_') + '.pkl'
        joblib.dump(model, os.path.join(models_dir, model_filename))

    print(f"[+] Modeller kaydedildi: {models_dir}")
    return results


def main():
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    print("\nEgitim tamamlandi!")
    return results


if __name__ == '__main__':
    main()
