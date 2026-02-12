"""
Model Training Module
=====================
Birden fazla ML modelini eğitir, karşılaştırır ve en iyi modeli kaydeder.
Modeller: Logistic Regression, Random Forest, XGBoost, SVM
"""

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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

warnings.filterwarnings('ignore')

# Proje kök dizini
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import prepare_data
from src.feature_extraction import FEATURE_NAMES
from src.utils import (
    plot_confusion_matrix, plot_roc_curve,
    plot_feature_importance, plot_model_comparison,
    print_classification_report
)


def get_models() -> dict:
    """
    Eğitilecek modellerin sözlüğünü döndürür.

    Returns
    -------
    dict
        {model_name: model_instance}
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0,
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        'SVM': SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            probability=True,
            random_state=42,
        ),
    }

    # XGBoost opsiyonel
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
        )
    except ImportError:
        print("[!] XGBoost yüklü değil, atlanıyor. Yüklemek için: pip install xgboost")

    return models


def train_and_evaluate(
    X_train, X_test, y_train, y_test,
    models: dict = None,
    cv_folds: int = 5,
) -> dict:
    """
    Tüm modelleri eğitir ve değerlendirir.

    Parameters
    ----------
    X_train, X_test : ndarray
        Eğitim ve test öznitelikleri.
    y_train, y_test : ndarray
        Eğitim ve test etiketleri.
    models : dict, optional
        Model sözlüğü.
    cv_folds : int
        Cross-validation fold sayısı.

    Returns
    -------
    dict
        Her model için performans metrikleri.
    """
    if models is None:
        models = get_models()

    results = {}
    best_model = None
    best_f1 = 0
    best_model_name = ''

    print("\n" + "=" * 60)
    print("MODEL EĞİTİMİ VE DEĞERLENDİRME")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n{'─' * 50}")
        print(f"🔧 {name}")
        print('─' * 50)

        # Eğitim
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        # Tahmin
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]

        # Metrikler
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='f1')

        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_time': train_time,
        }

        # Sonuçları yazdır
        print(f"  Accuracy:    {acc:.4f}")
        print(f"  Precision:   {prec:.4f}")
        print(f"  Recall:      {rec:.4f}")
        print(f"  F1-Score:    {f1:.4f}")
        print(f"  CV F1:       {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"  Eğitim süresi: {train_time:.2f}s")

        # Classification report
        print_classification_report(y_test, y_pred, name)

        # Grafikler
        plot_confusion_matrix(y_test, y_pred, name)
        if y_proba is not None:
            plot_roc_curve(y_test, y_proba, name)

        # Feature importance (sadece tree-based modeller)
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(
                model.feature_importances_,
                FEATURE_NAMES,
                name
            )

        # En iyi model kontrolü
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    # Model karşılaştırma grafiği
    print(f"\n{'=' * 60}")
    print("MODEL KARŞILAŞTIRMASI")
    print('=' * 60)
    plot_model_comparison(results)

    # Sonuç tablosu
    print(f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'CV F1':>12}")
    print('─' * 80)
    for name, r in results.items():
        marker = ' ★' if name == best_model_name else ''
        print(f"{name:<25} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>10.4f} {r['f1']:>10.4f} {r['cv_mean']:>6.4f}±{r['cv_std']:.4f}{marker}")

    # En iyi modeli kaydet
    print(f"\n🏆 En İyi Model: {best_model_name} (F1: {best_f1:.4f})")

    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)

    best_path = os.path.join(models_dir, 'best_model.pkl')
    joblib.dump(best_model, best_path)
    print(f"[✓] En iyi model kaydedildi: {best_path}")

    # Tüm modelleri kaydet
    for name, model in models.items():
        model_filename = name.lower().replace(' ', '_') + '.pkl'
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(model, model_path)

    print(f"[✓] Tüm modeller '{models_dir}' klasörüne kaydedildi.")

    return results


def main():
    """Ana eğitim akışı."""
    # Veri hazırla
    X_train, X_test, y_train, y_test, scaler = prepare_data()

    # Modelleri eğit ve değerlendir
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 60)
    print("✅ EĞİTİM TAMAMLANDI!")
    print("=" * 60)
    print("\n📂 Çıktılar:")
    print("   models/          → Kaydedilmiş modeller (.pkl)")
    print("   reports/figures/  → Performans grafikleri (.png)")

    return results


if __name__ == '__main__':
    main()
