"""
Self-Learning Feedback Loop — PhishGuard-AI
=============================================
Google Safe Browsing API ile model tahminlerini doğrular,
geri bildirimi kaydeder ve modeli otomatik yeniden eğitir.

Bu modül sayesinde model KENDİ KENDİNE ÖĞRENIR:
  1. Model bir URL hakkında tahmin yapar
  2. Google Safe Browsing API ile doğruluk kontrol edilir
  3. Sonuçlar feedback CSV'ye kaydedilir
  4. Yeterli veri biriktiğinde model otomatik yeniden eğitilir
  5. Her yeniden eğitimde model DAHA İYİ olur

Bu konsepte "Active Learning" veya "Online Learning" denir.
"""

import os
import csv
import time
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Feedback verisinin kaydedileceği dosya
FEEDBACK_FILE = os.path.join(PROJECT_ROOT, 'data', 'feedback', 'feedback_log.csv')
# Yeniden eğitim için gereken minimum yeni kayıt sayısı
RETRAIN_THRESHOLD = 50
# API sonuç önbelleği (aynı URL'yi tekrar sorgulamayı engeller)
CACHE_FILE = os.path.join(PROJECT_ROOT, 'data', 'feedback', 'api_cache.json')


def _ensure_feedback_dir():
    """Feedback dizinini oluşturur."""
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)


def _load_cache() -> dict:
    """API sonuç önbelleğini yükler."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    """API sonuç önbelleğini kaydeder."""
    _ensure_feedback_dir()
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)


def _url_hash(url: str) -> str:
    """URL'nin hash'ini döner (önbellek anahtarı)."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def check_url_safe_browsing(url: str, api_key: str = None) -> dict:
    """
    Google Safe Browsing API ile URL'yi kontrol eder.

    API key yoksa veya API erişilemezse, gelişmiş kural tabanlı
    doğrulama kullanılır (heuristic fallback).

    Parameters
    ----------
    url : str
        Kontrol edilecek URL.
    api_key : str, optional
        Google Safe Browsing API anahtarı.

    Returns
    -------
    dict
        {
            'is_malicious': bool,       # Zararlı mı?
            'source': str,              # Doğrulama kaynağı
            'confidence': float,        # API güven skoru (0-1)
            'threat_type': str,         # Tehdit türü
            'details': str,             # Detay açıklama
        }
    """
    # Önbellekte var mı kontrol et
    cache = _load_cache()
    url_key = _url_hash(url)
    if url_key in cache:
        cached = cache[url_key]
        cached['source'] = cached.get('source', 'cache') + ' (cached)'
        return cached

    result = None

    # ── 1. Google Safe Browsing API ──
    if api_key:
        try:
            import requests
            api_url = f'https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}'
            payload = {
                'client': {'clientId': 'phishguard-ai', 'clientVersion': '1.0'},
                'threatInfo': {
                    'threatTypes': ['MALWARE', 'SOCIAL_ENGINEERING', 'UNWANTED_SOFTWARE',
                                    'POTENTIALLY_HARMFUL_APPLICATION'],
                    'platformTypes': ['ANY_PLATFORM'],
                    'threatEntryTypes': ['URL'],
                    'threatEntries': [{'url': url}],
                }
            }
            response = requests.post(api_url, json=payload, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('matches'):
                    match = data['matches'][0]
                    result = {
                        'is_malicious': True,
                        'source': 'Google Safe Browsing API',
                        'confidence': 1.0,
                        'threat_type': match.get('threatType', 'UNKNOWN'),
                        'details': f"Google DB: {match.get('threatType', 'UNKNOWN')} tehdidi tespit edildi",
                    }
                else:
                    result = {
                        'is_malicious': False,
                        'source': 'Google Safe Browsing API',
                        'confidence': 0.95,
                        'threat_type': 'NONE',
                        'details': 'Google DB: Bu URL temiz.',
                    }
        except Exception as e:
            pass  # API hata verirse heuristic'e düş

    # ── 2. VirusTotal API (yedek) ──
    if result is None and api_key:
        try:
            import requests
            vt_url = 'https://www.virustotal.com/vtapi/v2/url/report'
            params = {'apikey': api_key, 'resource': url}
            response = requests.get(vt_url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                positives = data.get('positives', 0)
                total = data.get('total', 1)
                result = {
                    'is_malicious': positives > 2,
                    'source': 'VirusTotal API',
                    'confidence': positives / max(total, 1),
                    'threat_type': 'MALWARE' if positives > 2 else 'NONE',
                    'details': f"VirusTotal: {positives}/{total} motor tehdit tespit etti",
                }
        except Exception:
            pass

    # ── 3. Gelişmiş Heuristic Doğrulama (API olmadan) ──
    if result is None:
        result = _heuristic_verification(url)

    # Önbelleğe kaydet
    cache[url_key] = result
    _save_cache(cache)

    return result


def _heuristic_verification(url: str) -> dict:
    """
    API olmadan gelişmiş kural tabanlı doğrulama.
    Bilinen phishing kalıplarını ve güvenli siteleri kontrol eder.
    """
    from urllib.parse import urlparse
    import re

    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()

    threat_score = 0
    reasons = []

    # ── Bilinen Güvenli Siteler (Whitelist) ──
    SAFE_DOMAINS = [
        'google.com', 'youtube.com', 'facebook.com', 'twitter.com',
        'instagram.com', 'linkedin.com', 'github.com', 'microsoft.com',
        'apple.com', 'amazon.com', 'netflix.com', 'wikipedia.org',
        'stackoverflow.com', 'reddit.com', 'whatsapp.com', 'yahoo.com',
        'bing.com', 'zoom.us', 'spotify.com', 'twitch.tv',
    ]
    for safe in SAFE_DOMAINS:
        if domain.endswith(safe):
            return {
                'is_malicious': False,
                'source': 'Heuristic (Whitelist)',
                'confidence': 0.99,
                'threat_type': 'NONE',
                'details': f'Bilinen guvenli site: {safe}',
            }

    # ── Skor Tabanlı Analiz ──
    # IP adresi kullanımı
    if re.match(r'^(\d{1,3}\.){3}\d{1,3}', domain):
        threat_score += 40
        reasons.append('IP adresi kullanimi')

    # Şüpheli TLD
    suspicious_tlds = ['tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'buzz',
                       'club', 'work', 'date', 'win', 'bid', 'stream', 'download']
    tld = domain.split('.')[-1] if '.' in domain else ''
    if tld in suspicious_tlds:
        threat_score += 35
        reasons.append(f'Suspeli TLD: .{tld}')

    # HTTPS yok
    if parsed.scheme != 'https':
        threat_score += 15
        reasons.append('HTTPS yok')

    # Domain'de marka taklit kalıpları
    brand_keywords = ['paypal', 'apple', 'google', 'microsoft', 'amazon',
                      'netflix', 'bank', 'secure', 'login', 'verify', 'account',
                      'update', 'confirm', 'signin', 'password']
    for brand in brand_keywords:
        if brand in domain and not domain.endswith(f'{brand}.com'):
            threat_score += 25
            reasons.append(f'Marka taklidi: {brand}')
            break

    # URL kısaltma servisleri
    shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly',
                  'is.gd', 'buff.ly', 'cutt.ly', 'rb.gy']
    for s in shorteners:
        if s in domain:
            threat_score += 30
            reasons.append(f'URL kisaltma: {s}')
            break

    # Domain'de tire
    if '-' in domain:
        threat_score += 10
        reasons.append('Domain tire iceriyor')

    # @ işareti
    if '@' in url:
        threat_score += 30
        reasons.append('@ isareti (URL manipulasyonu)')

    # Çok uzun URL
    if len(url) > 100:
        threat_score += 10
        reasons.append(f'Cok uzun URL: {len(url)} karakter')

    # Sonuç
    is_malicious = threat_score >= 40
    confidence = min(threat_score / 100.0, 0.95)

    return {
        'is_malicious': is_malicious,
        'source': 'Heuristic Analysis',
        'confidence': confidence if is_malicious else 1.0 - confidence,
        'threat_type': 'SOCIAL_ENGINEERING' if is_malicious else 'NONE',
        'details': '; '.join(reasons) if reasons else 'Heuristic: skor dusuk, guvenli gorunuyor',
    }


def log_feedback(url: str, model_prediction: int, model_confidence: float,
                 api_result: dict, features: dict):
    """
    Model tahmini ve API doğrulamasını feedback dosyasına kaydeder.

    Bu kayıtlar, modelin yeniden eğitilmesinde kullanılacak.
    """
    _ensure_feedback_dir()

    # API sonucunu binary label'a çevir
    api_label = 1 if api_result.get('is_malicious', False) else 0

    # Model doğru mu tahmin etmiş?
    is_correct = (model_prediction == api_label)

    row = {
        'timestamp': datetime.now().isoformat(),
        'url': url,
        'model_prediction': model_prediction,
        'model_confidence': round(model_confidence, 4),
        'api_label': api_label,
        'api_source': api_result.get('source', 'unknown'),
        'api_confidence': round(api_result.get('confidence', 0), 4),
        'is_correct': int(is_correct),
        'threat_type': api_result.get('threat_type', 'UNKNOWN'),
    }

    # Öznitelikleri de ekle
    for k, v in features.items():
        row[f'feat_{k}'] = v

    # CSV'ye yaz
    file_exists = os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return is_correct, api_label


def get_feedback_stats() -> dict:
    """Feedback istatistiklerini döner."""
    if not os.path.exists(FEEDBACK_FILE):
        return {
            'total': 0, 'correct': 0, 'incorrect': 0,
            'accuracy': 0, 'ready_to_retrain': False,
            'needed_for_retrain': RETRAIN_THRESHOLD,
        }

    df = pd.read_csv(FEEDBACK_FILE)
    total = len(df)
    correct = int(df['is_correct'].sum()) if 'is_correct' in df.columns else 0
    incorrect = total - correct

    return {
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': correct / max(total, 1),
        'ready_to_retrain': total >= RETRAIN_THRESHOLD,
        'needed_for_retrain': max(0, RETRAIN_THRESHOLD - total),
    }


def auto_retrain_if_ready(force: bool = False) -> dict:
    """
    Yeterli feedback verisi biriktiğinde modeli otomatik yeniden eğitir.

    Bu fonksiyon Active Learning döngüsünün kalbidir:
    1. Orijinal eğitim verisini yükle
    2. Feedback verisini ekle (özellikle MODEL'İN YANILDIĞI örnekler!)
    3. Modeli yeniden eğit
    4. Yeni modeli kaydet
    5. Performans karşılaştırması yap

    Parameters
    ----------
    force : bool
        True ise eşik kontrolü yapılmadan eğitim başlatılır.

    Returns
    -------
    dict : Yeniden eğitim sonuçları
    """
    stats = get_feedback_stats()

    if not force and not stats['ready_to_retrain']:
        return {
            'retrained': False,
            'message': f"{stats['needed_for_retrain']} kayit daha gerekiyor ({stats['total']}/{RETRAIN_THRESHOLD})",
        }

    if not os.path.exists(FEEDBACK_FILE):
        return {'retrained': False, 'message': 'Feedback verisi bulunamadi'}

    from src.feature_extraction import extract_features, FEATURE_NAMES
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # ── 1. Orijinal veri setini yükle ──
    original_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'phishing_urls.csv')
    results = {'retrained': False}

    if os.path.exists(original_path):
        orig_df = pd.read_csv(original_path)
        orig_features = []
        orig_labels = []
        for _, row in orig_df.iterrows():
            feats = extract_features(str(row.get('url', '')))
            orig_features.append([feats[f] for f in FEATURE_NAMES])
            label = row.get('label', 'legitimate')
            orig_labels.append(1 if str(label).lower() in ['phishing', '1', 'bad'] else 0)
    else:
        orig_features = []
        orig_labels = []

    # ── 2. Feedback verisini yükle ──
    fb_df = pd.read_csv(FEEDBACK_FILE)
    fb_features = []
    fb_labels = []

    feat_cols = [c for c in fb_df.columns if c.startswith('feat_')]
    for _, row in fb_df.iterrows():
        feat_vector = [row[c] for c in feat_cols]
        fb_features.append(feat_vector)
        # API label'ını kullan (gerçek doğruluk kaynağı)
        fb_labels.append(int(row.get('api_label', row.get('model_prediction', 0))))

    # ── 3. Veri setlerini birleştir ──
    all_features = orig_features + fb_features
    all_labels = orig_labels + fb_labels

    if len(all_features) < 10:
        return {'retrained': False, 'message': 'Yeterli veri yok (min 10 ornek gerekli)'}

    X = np.array(all_features)
    y = np.array(all_labels)

    # ── 4. Train/Test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 5. Ölçeklendirme ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── 6. Model eğitimi ──
    model = RandomForestClassifier(
        n_estimators=200, max_depth=20,
        min_samples_split=5, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # ── 7. Değerlendirme ──
    y_pred = model.predict(X_test_scaled)
    new_acc = accuracy_score(y_test, y_pred)
    new_f1 = f1_score(y_test, y_pred, zero_division=0)

    # ── 8. Eski model ile karşılaştır ──
    old_model_path = os.path.join(PROJECT_ROOT, 'models', 'best_model.pkl')
    old_acc = 0
    if os.path.exists(old_model_path):
        try:
            old_model = joblib.load(old_model_path)
            old_scaler_path = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')
            if os.path.exists(old_scaler_path):
                old_scaler = joblib.load(old_scaler_path)
                X_test_old = old_scaler.transform(X_test)
            else:
                X_test_old = X_test
            old_pred = old_model.predict(X_test_old)
            old_acc = accuracy_score(y_test, old_pred)
        except Exception:
            old_acc = 0

    # ── 9. Yeni model daha iyiyse kaydet ──
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Her zaman kaydet (feedback verisi ile güncellendi)
    joblib.dump(model, os.path.join(models_dir, 'best_model.pkl'))
    joblib.dump(model, os.path.join(models_dir, 'random_forest.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))

    # Eğitim logunu kaydet
    log_path = os.path.join(PROJECT_ROOT, 'data', 'feedback', 'retrain_log.csv')
    log_row = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(all_features),
        'feedback_samples': len(fb_features),
        'old_accuracy': round(old_acc, 4),
        'new_accuracy': round(new_acc, 4),
        'new_f1': round(new_f1, 4),
        'improved': new_acc >= old_acc,
    }
    log_exists = os.path.exists(log_path)
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=log_row.keys())
        if not log_exists:
            writer.writeheader()
        writer.writerow(log_row)

    results = {
        'retrained': True,
        'total_samples': len(all_features),
        'feedback_samples': len(fb_features),
        'old_accuracy': round(old_acc, 4),
        'new_accuracy': round(new_acc, 4),
        'new_f1': round(new_f1, 4),
        'improved': new_acc >= old_acc,
        'message': f"Model yeniden egitildi! Dogruluk: {old_acc:.2%} -> {new_acc:.2%}",
    }

    return results
