import os
import sys
import argparse
import joblib
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.feature_extraction import extract_features, FEATURE_NAMES


def load_model(model_path=None):
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, 'models', 'best_model.pkl')

    scaler_path = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model bulunamadi: {model_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler


def predict_url(url, model=None, scaler=None):
    if model is None:
        model, scaler = load_model()

    features = extract_features(url)
    X = np.array([[features[f] for f in FEATURE_NAMES]])

    if scaler is not None:
        X = scaler.transform(X)

    prediction = model.predict(X)[0]

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


def predict_urls(urls, model=None, scaler=None):
    if model is None:
        model, scaler = load_model()
    return [predict_url(url, model, scaler) for url in urls]


def main():
    parser = argparse.ArgumentParser(description='Phishing URL Tespiti')
    parser.add_argument('--url', type=str, help='Degerlendirilecek URL')
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    if args.url:
        result = predict_url(args.url)
        print(f"\nURL: {result['url']}")
        print(f"Sonuc: {result['label']}")
        print(f"Guven: {result['confidence']:.1f}%")
        print(f"\nFeatures:")
        for key, val in result['features'].items():
            print(f"   {key:<22} : {val}")
    else:
        print("\nPhishing URL Tespiti - Interaktif Mod")
        print("Cikmak icin 'q' yazin.\n")
        model, scaler = load_model()
        while True:
            url = input("URL girin: ").strip()
            if url.lower() in ('q', 'quit', 'exit'):
                break
            if not url:
                continue
            result = predict_url(url, model, scaler)
            print(f"  {result['label']} (Guven: {result['confidence']:.1f}%)\n")


if __name__ == '__main__':
    main()
