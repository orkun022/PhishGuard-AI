"""
Preprocessing Module
====================
Veri yükleme, feature extraction uygulama ve train/test split işlemleri.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Proje kök dizinini belirle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Modül import
import sys
sys.path.insert(0, PROJECT_ROOT)
from src.feature_extraction import extract_features, FEATURE_NAMES


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    CSV dosyasından URL verisini yükler.

    Parameters
    ----------
    filepath : str, optional
        CSV dosya yolu. Belirtilmezse varsayılan demo veri seti kullanılır.

    Returns
    -------
    pd.DataFrame
        'url' ve 'label' sütunlarını içeren DataFrame.
    """
    if filepath is None:
        filepath = os.path.join(PROJECT_ROOT, 'data', 'raw', 'phishing_urls.csv')

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Veri seti bulunamadı: {filepath}\n"
            "Lütfen 'data/raw/' klasörüne bir CSV dosyası koyun."
        )

    df = pd.read_csv(filepath)

    # Sütun isimlerini standartlaştır
    df.columns = df.columns.str.strip().str.lower()

    # 'url' ve 'label' sütunlarının varlığını kontrol et
    if 'url' not in df.columns:
        raise ValueError("CSV dosyasında 'url' sütunu bulunamadı.")
    if 'label' not in df.columns:
        raise ValueError("CSV dosyasında 'label' sütunu bulunamadı.")

    return df


def apply_feature_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame'deki URL'lerden öznitelik çıkarır.

    Parameters
    ----------
    df : pd.DataFrame
        'url' sütununu içeren DataFrame.

    Returns
    -------
    pd.DataFrame
        Özniteliklerin eklenmiş olduğu DataFrame.
    """
    print("[*] Feature extraction başlatılıyor...")

    features_list = []
    total = len(df)

    for i, url in enumerate(df['url']):
        if (i + 1) % 200 == 0 or (i + 1) == total:
            print(f"    İşlenen: {i + 1}/{total}")
        features_list.append(extract_features(str(url)))

    features_df = pd.DataFrame(features_list)
    result = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    print(f"[✓] {len(FEATURE_NAMES)} öznitelik çıkarıldı.")
    return result


def prepare_data(
    filepath: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
    save_processed: bool = True,
) -> tuple:
    """
    Veriyi yükler, öznitelik çıkarır, train/test split yapar.

    Parameters
    ----------
    filepath : str, optional
        CSV dosya yolu.
    test_size : float
        Test seti oranı (varsayılan: 0.2)
    random_state : int
        Rastgelelik tohumu.
    scale : bool
        StandardScaler uygulansın mı.
    save_processed : bool
        İşlenmiş veriyi kaydet.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, scaler)
    """
    # 1. Veriyi yükle
    print("=" * 50)
    print("VERI HAZIRLAMA")
    print("=" * 50)

    df = load_data(filepath)
    print(f"[✓] Veri yüklendi: {len(df)} satır")
    print(f"    Sınıf dağılımı:\n{df['label'].value_counts().to_string()}\n")

    # 2. Feature extraction
    df = apply_feature_extraction(df)

    # 3. İşlenmiş veriyi kaydet
    if save_processed:
        processed_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'features.csv')
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"[✓] İşlenmiş veri kaydedildi: {processed_path}")

    # 4. X ve y ayır
    X = df[FEATURE_NAMES].values
    y = df['label'].values

    # Label encoding (eğer string ise)
    if y.dtype == object:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"[✓] Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # 5. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\n[✓] Train/Test split ({1-test_size:.0%}/{test_size:.0%}):")
    print(f"    Train: {X_train.shape[0]} örnek")
    print(f"    Test:  {X_test.shape[0]} örnek")

    # 6. Ölçeklendirme
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Scaler'ı kaydet
        scaler_path = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"[✓] StandardScaler kaydedildi: {scaler_path}")

    print("=" * 50)
    return X_train, X_test, y_train, y_test, scaler


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
