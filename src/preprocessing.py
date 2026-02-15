import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.feature_extraction import extract_features, FEATURE_NAMES


def load_data(filepath=None):
    if filepath is None:
        filepath = os.path.join(PROJECT_ROOT, 'data', 'raw', 'phishing_urls.csv')

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Veri seti bulunamadi: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()

    if 'url' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV dosyasinda 'url' ve 'label' sutunlari olmali.")

    return df


def apply_feature_extraction(df):
    print("[*] Feature extraction baslatiliyor...")
    features_list = []
    total = len(df)

    for i, url in enumerate(df['url']):
        if (i + 1) % 200 == 0 or (i + 1) == total:
            print(f"    Islenen: {i + 1}/{total}")
        features_list.append(extract_features(str(url)))

    features_df = pd.DataFrame(features_list)
    result = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    print(f"[+] {len(FEATURE_NAMES)} feature cikarildi.")
    return result


def prepare_data(filepath=None, test_size=0.2, random_state=42, scale=True, save_processed=True):
    print("=" * 50)
    print("VERI HAZIRLAMA")
    print("=" * 50)

    df = load_data(filepath)
    print(f"[+] Veri yuklendi: {len(df)} satir")

    df = apply_feature_extraction(df)

    if save_processed:
        processed_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'features.csv')
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)

    X = df[FEATURE_NAMES].values
    y = df['label'].values

    if y.dtype == object:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[+] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        scaler_path = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return X_train, X_test, y_train, y_test, scaler


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
