"""
Model Pipeline Testleri
========================
Model eğitim ve tahmin pipeline'ının çalışabilirliğini doğrular.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction import extract_features, FEATURE_NAMES


def test_feature_extraction_returns_correct_features():
    """Feature extraction doğru sayıda öznitelik döndürüyor mu?"""
    url = "https://www.example.com/path?q=test"
    features = extract_features(url)

    assert len(features) == len(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        assert name in features


def test_features_are_numeric():
    """Tüm öznitelikler sayısal mı?"""
    urls = [
        "https://www.google.com",
        "http://192.168.1.1/login",
        "http://phishing.tk/claim",
    ]

    for url in urls:
        features = extract_features(url)
        feature_values = [features[f] for f in FEATURE_NAMES]
        arr = np.array(feature_values)
        assert arr.dtype in [np.int64, np.float64, np.int32, np.float32], \
            f"Öznitelikler numpy array'e dönüştürülebilmeli, url: {url}"


def test_feature_vector_shape():
    """Feature vektörünün boyutu doğru mu?"""
    url = "https://www.example.com"
    features = extract_features(url)

    X = np.array([[features[f] for f in FEATURE_NAMES]])
    assert X.shape == (1, len(FEATURE_NAMES)), \
        f"Beklenen shape: (1, {len(FEATURE_NAMES)}), alınan: {X.shape}"


def test_phishing_vs_legitimate_features_differ():
    """Phishing ve legitimate URL'lerin öznitelikleri farklı mı?"""
    legit_url = "https://www.google.com/search?q=python"
    phishing_url = "http://192.168.1.1@paypal-secure.tk/login/verify"

    legit_features = extract_features(legit_url)
    phishing_features = extract_features(phishing_url)

    # En azından bazı öznitelikler farklı olmalı
    differences = sum(
        1 for f in FEATURE_NAMES
        if legit_features[f] != phishing_features[f]
    )

    assert differences > 3, \
        f"Phishing ve legitimate URL'ler arasında yeterli fark olmalı (fark: {differences})"


def test_sklearn_model_can_fit():
    """Basit bir sklearn modeli özniteliklerle eğitilebiliyor mu?"""
    from sklearn.ensemble import RandomForestClassifier

    # Mini veri seti oluştur
    urls = [
        ("https://www.google.com", 0),
        ("https://github.com/features", 0),
        ("https://www.python.org/docs", 0),
        ("http://192.168.1.1/login", 1),
        ("http://phishing.tk/claim", 1),
        ("http://paypal-secure.ml/verify", 1),
    ]

    X = []
    y = []
    for url, label in urls:
        features = extract_features(url)
        X.append([features[f] for f in FEATURE_NAMES])
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Model eğit
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Tahmin yap
    predictions = model.predict(X)
    assert len(predictions) == len(y), "Tahmin sayısı etiket sayısına eşit olmalı"
    assert all(p in [0, 1] for p in predictions), "Tahminler 0 veya 1 olmalı"


def test_model_predict_proba():
    """Model olasılık tahmini yapabiliyor mu?"""
    from sklearn.ensemble import RandomForestClassifier

    urls = [
        ("https://www.google.com", 0),
        ("https://github.com", 0),
        ("http://phishing.tk/claim", 1),
        ("http://192.168.1.1/admin", 1),
    ]

    X = []
    y = []
    for url, label in urls:
        features = extract_features(url)
        X.append([features[f] for f in FEATURE_NAMES])
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2), "Olasılık matrisi (n, 2) boyutunda olmalı"
    assert np.allclose(proba.sum(axis=1), 1.0), "Olasılıklar toplamı 1 olmalı"
