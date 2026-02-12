"""
Prediction Module
=================
Kaydedilmiş modeli kullanarak tek URL veya URL listesi üzerinde tahmin yapar.
"""

import os
import sys
import argparse
import joblib
import numpy as np

# Proje kök dizini
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.feature_extraction import extract_features, FEATURE_NAMES


def load_model(model_path: str = None):
    """
    Kaydedilmiş modeli yükler.

    Parameters
    ----------
    model_path : str, optional
        Model dosya yolu. Belirtilmezse en iyi model yüklenir.

    Returns
    -------
    tuple
        (model, scaler)
    """
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, 'models', 'best_model.pkl')

    scaler_path = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {model_path}\n"
            "Önce 'python src/train.py' ile model eğitin."
        )

    model = joblib.load(model_path)

    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    return model, scaler


def predict_url(url: str, model=None, scaler=None) -> dict:
    """
    Tek bir URL için tahmin yapar.

    Parameters
    ----------
    url : str
        Değerlendirilecek URL.
    model : optional
        ML modeli. None ise kaydedilmiş model yüklenir.
    scaler : optional
        StandardScaler. None ise kaydedilmiş scaler yüklenir.

    Returns
    -------
    dict
        Tahmin sonuçları:
        - prediction: 0 (Legitimate) veya 1 (Phishing)
        - label: 'Legitimate' veya 'Phishing'
        - confidence: Tahmin güveni (%)
        - features: Çıkarılan öznitelikler
    """
    if model is None:
        model, scaler = load_model()

    # Feature extraction
    features = extract_features(url)
    X = np.array([[features[f] for f in FEATURE_NAMES]])

    # Ölçeklendirme
    if scaler is not None:
        X = scaler.transform(X)

    # Tahmin
    prediction = model.predict(X)[0]

    # Güven skoru
    confidence = 0.0
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
        confidence = max(proba) * 100

    label = 'Phishing' if prediction == 1 else 'Legitimate'

    return {
        'url': url,
        'prediction': int(prediction),
        'label': label,
        'confidence': round(confidence, 2),
        'features': features,
    }


def predict_urls(urls: list, model=None, scaler=None) -> list:
    """
    Birden fazla URL için tahmin yapar.

    Parameters
    ----------
    urls : list of str
        URL listesi.

    Returns
    -------
    list of dict
        Her URL için tahmin sonuçları.
    """
    if model is None:
        model, scaler = load_model()

    return [predict_url(url, model, scaler) for url in urls]


def main():
    """CLI arayüzü."""
    parser = argparse.ArgumentParser(
        description='Phishing URL Tespiti — Tahmin Modülü'
    )
    parser.add_argument(
        '--url', type=str,
        help='Değerlendirilecek URL'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='Model dosya yolu (varsayılan: models/best_model.pkl)'
    )

    args = parser.parse_args()

    if args.url:
        result = predict_url(args.url)

        print("\n" + "=" * 50)
        print("PHISHING URL TESPİTİ")
        print("=" * 50)
        print(f"\n🔗 URL: {result['url']}")
        print(f"\n{'🚨 PHISHING!' if result['prediction'] == 1 else '✅ LEGITIMATE'}")
        print(f"   Sonuç:   {result['label']}")
        print(f"   Güven:   {result['confidence']:.1f}%")

        print(f"\n📊 Çıkarılan Öznitelikler:")
        for key, val in result['features'].items():
            print(f"   {key:<22} : {val}")
    else:
        # İnteraktif mod
        print("\n🛡️ Phishing URL Tespiti — İnteraktif Mod")
        print("Çıkmak için 'q' yazın.\n")

        model, scaler = load_model()

        while True:
            url = input("URL girin: ").strip()
            if url.lower() in ('q', 'quit', 'exit'):
                print("Çıkılıyor...")
                break
            if not url:
                continue

            result = predict_url(url, model, scaler)
            emoji = '🚨' if result['prediction'] == 1 else '✅'
            print(f"  {emoji} {result['label']} (Güven: {result['confidence']:.1f}%)\n")


if __name__ == '__main__':
    main()
