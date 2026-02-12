"""
Feature Extraction Testleri
============================
URL öznitelik çıkarımı fonksiyonlarının doğruluğunu test eder.
"""

import sys
import os

# Proje kök dizinini ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction import extract_features, extract_features_batch, FEATURE_NAMES


class TestFeatureExtraction:
    """Feature extraction fonksiyonlarının test sınıfı."""

    def test_legitimate_google(self):
        """Google URL'si için temel öznitelik kontrolü."""
        url = "https://www.google.com/search?q=python"
        features = extract_features(url)

        assert features['has_https'] == 1, "Google HTTPS kullanmalı"
        assert features['has_ip'] == 0, "Google IP adresi kullanmamalı"
        assert features['has_at_sign'] == 0, "Google URL'de @ olmamalı"
        assert features['has_shortener'] == 0, "Google kısaltma servisi değil"
        assert features['suspicious_tld'] == 0, "Google şüpheli TLD kullanmamalı"

    def test_phishing_ip_url(self):
        """IP adresi kullanan phishing URL testi."""
        url = "http://192.168.1.1/admin/login.php"
        features = extract_features(url)

        assert features['has_ip'] == 1, "IP adresi tespit edilmeli"
        assert features['has_https'] == 0, "HTTP kullanılıyor"

    def test_phishing_at_sign(self):
        """@ işareti içeren phishing URL testi."""
        url = "http://www.paypal.com@hack.tk/login"
        features = extract_features(url)

        assert features['has_at_sign'] == 1, "@ işareti tespit edilmeli"

    def test_url_with_dash(self):
        """Domain'de dash içeren URL testi."""
        url = "http://paypal-secure-login.com/update"
        features = extract_features(url)

        assert features['has_dash'] == 1, "Dash tespit edilmeli"

    def test_shortener_detection(self):
        """URL kısaltma servisi testi."""
        url = "http://bit.ly/abc123"
        features = extract_features(url)

        assert features['has_shortener'] == 1, "Kısaltma servisi tespit edilmeli"

    def test_suspicious_tld(self):
        """Şüpheli TLD testi."""
        url = "http://free-iphone.tk/claim"
        features = extract_features(url)

        assert features['suspicious_tld'] == 1, "Şüpheli TLD tespit edilmeli"

    def test_double_slash(self):
        """Çift // yönlendirmesi testi."""
        url = "http://example.com//http://malicious.com/steal"
        features = extract_features(url)

        assert features['has_double_slash'] == 1, "Çift // tespit edilmeli"

    def test_feature_count(self):
        """Toplam öznitelik sayısı testi."""
        url = "https://www.example.com"
        features = extract_features(url)

        assert len(features) == len(FEATURE_NAMES), \
            f"Beklenen {len(FEATURE_NAMES)} öznitelik, alınan {len(features)}"

    def test_all_feature_names_present(self):
        """Tüm öznitelik isimlerinin mevcut olduğunu doğrula."""
        url = "https://www.example.com/path?q=test"
        features = extract_features(url)

        for name in FEATURE_NAMES:
            assert name in features, f"'{name}' özniteliği eksik"

    def test_batch_extraction(self):
        """Batch feature extraction testi."""
        urls = [
            "https://www.google.com",
            "http://phishing.tk/login",
            "http://192.168.1.1/admin",
        ]
        results = extract_features_batch(urls)

        assert len(results) == 3, "3 URL için 3 sonuç olmalı"
        assert all(isinstance(r, dict) for r in results), "Sonuçlar dict olmalı"

    def test_url_length(self):
        """URL uzunluğu testi."""
        short_url = "http://a.co"
        long_url = "http://this-is-a-very-long-suspicious-domain-name.com/with/a/very/long/path/structure"

        short_features = extract_features(short_url)
        long_features = extract_features(long_url)

        assert long_features['url_length'] > short_features['url_length']

    def test_num_digits(self):
        """Rakam sayısı testi."""
        url = "http://123.456.789.0/path123/file456"
        features = extract_features(url)

        assert features['num_digits'] > 0, "Rakamlar sayılmalı"

    def test_subdomain_count(self):
        """Alt domain sayısı testi."""
        url = "http://sub1.sub2.sub3.example.com/path"
        features = extract_features(url)

        assert features['subdomain_count'] >= 2, "Birden fazla alt domain olmalı"

    def test_num_params(self):
        """Sorgu parametresi sayısı testi."""
        url = "http://example.com/page?id=1&name=test&token=abc"
        features = extract_features(url)

        assert features['num_params'] == 3, "3 parametre olmalı"

    def test_empty_url(self):
        """Boş URL testi — hata vermemeli."""
        features = extract_features("")
        assert isinstance(features, dict), "Boş URL için de dict dönmeli"

    def test_numeric_values(self):
        """Tüm özniteliklerin sayısal olduğunu doğrula."""
        url = "https://www.google.com/search?q=test"
        features = extract_features(url)

        for key, val in features.items():
            assert isinstance(val, (int, float)), \
                f"'{key}' sayısal olmalı, type: {type(val)}"


# pytest ile çalıştırılabilir testler
def test_legitimate_google():
    TestFeatureExtraction().test_legitimate_google()

def test_phishing_ip_url():
    TestFeatureExtraction().test_phishing_ip_url()

def test_phishing_at_sign():
    TestFeatureExtraction().test_phishing_at_sign()

def test_url_with_dash():
    TestFeatureExtraction().test_url_with_dash()

def test_shortener_detection():
    TestFeatureExtraction().test_shortener_detection()

def test_suspicious_tld():
    TestFeatureExtraction().test_suspicious_tld()

def test_double_slash():
    TestFeatureExtraction().test_double_slash()

def test_feature_count():
    TestFeatureExtraction().test_feature_count()

def test_all_feature_names_present():
    TestFeatureExtraction().test_all_feature_names_present()

def test_batch_extraction():
    TestFeatureExtraction().test_batch_extraction()

def test_url_length():
    TestFeatureExtraction().test_url_length()

def test_num_digits():
    TestFeatureExtraction().test_num_digits()

def test_subdomain_count():
    TestFeatureExtraction().test_subdomain_count()

def test_num_params():
    TestFeatureExtraction().test_num_params()

def test_empty_url():
    TestFeatureExtraction().test_empty_url()

def test_numeric_values():
    TestFeatureExtraction().test_numeric_values()
