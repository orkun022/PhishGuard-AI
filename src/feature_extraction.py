"""
Feature Extraction Module — PhishGuard-AI
==========================================
URL'lerden (Uniform Resource Locator) makine öğrenmesi modeline
beslenecek sayısal öznitelikler (features) çıkarır.

Toplam 16 adet öznitelik çıkarılır:
  - URL yapısal özellikleri (uzunluk, path, parametre)
  - Domain analizi (IP, subdomain, TLD)
  - Güvenlik göstergeleri (HTTPS, @, //, shortener)

────────────────────────────────────────────────────────────
BIG-O ZAMAN KARMAŞIKLIĞI ANALİZİ
────────────────────────────────────────────────────────────

extract_features(url) fonksiyonunun zaman karmaşıklığı:

  n = URL string uzunluğu
  d = domain parça sayısı (split('.') sonucu)
  s = SHORTENING_SERVICES listesi uzunluğu (sabit, ~15)

  1. urlparse(url)           → O(n)   : URL string'ini bir kez tarar
  2. len(url)                → O(1)   : Python str nesnesi uzunluğu sabit zamanda döner
  3. len(domain)             → O(1)   : Aynı şekilde sabit zaman
  4. re.match (IP pattern)   → O(d)   : Domain uzunluğu kadar taranır (d << n)
  5. '@' in url              → O(n)   : String'de arama, en kötü durumda O(n)
  6. url.count('//')         → O(n)   : String'de sayma, O(n)
  7. '-' in domain           → O(d)   : Domain uzunluğu kadar, O(d)
  8. scheme == 'https'       → O(1)   : String karşılaştırma (sabit uzunluk)
  9. url.count('.')          → O(n)   : String'de sayma, O(n)
  10. sum(isdigit)           → O(n)   : Her karakter kontrol edilir
  11. re.findall(special)    → O(n)   : Regex ile tek tarama
  12. domain.split('.')      → O(d)   : Domain'i böl (d << n)
  13. len(path)              → O(1)   : Sabit zaman
  14. parse_qs(query)        → O(q)   : q = query string uzunluğu (q ≤ n)
  15. any(s in domain)       → O(s*d) : Her ser vis için domain'de arama (s ve d sabit)
  16. split('.')[-1]         → O(d)   : TLD çıkarma

  TOPLAM: O(n) + O(n) + O(n) + O(n) + O(n) + O(s*d)
        = O(n)  (s ve d pratikte sabit değerler olduğu için)

  Sonuç: Tek bir URL için O(n) — lineer zaman karmaşıklığı.
  
  extract_features_batch(urls) fonksiyonu:
    m = URL listesi uzunluğu
    Toplam: O(m * n_avg) — her URL için O(n) uygulanır
    n_avg = ortalama URL uzunluğu

  Uzay Karmaşıklığı: O(k) — k sabit sayıda öznitelik (16 adet)
  Her URL için sabit boyutlu bir dict döner.
────────────────────────────────────────────────────────────
"""

import re                           # Düzenli ifadeler (Regular Expression) kütüphanesi
from urllib.parse import urlparse, parse_qs  # URL'yi bileşenlerine ayırmak için standart kütüphane


# ──────────────────────────────────────────────
# SABİT LİSTELER (Lookup tabloları)
# ──────────────────────────────────────────────

# Bilinen URL kısaltma servisleri — phishing saldırılarında
# gerçek URL'yi gizlemek için sıklıkla kullanılır.
SHORTENING_SERVICES = [
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly',
    'is.gd', 'buff.ly', 'adf.ly', 'cutt.ly', 'rb.gy',
    'short.io', 'tiny.cc', 'lnkd.in', 'db.tt', 'qr.ae',
]

# Şüpheli TLD'ler (Top-Level Domain)
# Bu TLD'ler ücretsiz sunulduğu için phishing sitelerinde
# diğer TLD'lere kıyasla çok daha yüksek oranda kullanılır.
SUSPICIOUS_TLDS = [
    'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'buzz',
    'club', 'work', 'date', 'racing', 'win', 'bid',
    'stream', 'download', 'cricket', 'party', 'science',
]


def extract_features(url: str) -> dict:
    """
    Verilen bir URL string'inden 16 adet sayısal öznitelik çıkarır.
    Bu öznitelikler, ML modeline girdi olarak kullanılır.

    Parameters
    ----------
    url : str
        Analiz edilecek URL string'i (örn: "https://www.google.com/search?q=ai")

    Returns
    -------
    dict
        16 özniteliği içeren sözlük. Tüm değerler int tipindedir (0/1 veya sayım).

    Time Complexity: O(n), n = URL uzunluğu
    Space Complexity: O(1), sabit sayıda öznitelik döner
    """

    features = {}  # Çıkarılan özniteliklerin tutulacağı sözlük

    # ──── URL'yi bileşenlerine ayır (parse et) ────
    # urlparse() fonksiyonu URL'yi şu parçalara böler:
    #   scheme://netloc/path;params?query#fragment
    # Örnek: https://www.google.com/search?q=ai
    #   scheme = "https"
    #   netloc = "www.google.com"  (domain)
    #   path   = "/search"
    #   query  = "q=ai"
    try:
        parsed = urlparse(url)       # URL'yi parçalarına ayır — O(n)
    except Exception:
        parsed = urlparse('')        # Hatalı URL varsa boş parse objesi oluştur

    domain = parsed.netloc or ''     # Domain (netloc) kısmını al; yoksa boş string
    path = parsed.path or ''         # Path kısmını al; yoksa boş string
    query = parsed.query or ''       # Query string kısmını al; yoksa boş string

    # ═══════════════════════════════════════════════
    # ÖZNİTELİK 1: URL Uzunluğu
    # ═══════════════════════════════════════════════
    # Phishing URL'leri genellikle normalden daha uzundur çünkü
    # gerçek domain'i gizlemek için ekstra path ve parametre eklerler.
    # Araştırmalara göre 75 karakterden uzun URL'ler şüphelidir.
    # Zaman: O(1) — Python'da len() sabit zamanlıdır.
    features['url_length'] = len(url)

    # ═══════════════════════════════════════════════
    # ÖZNİTELİK 2: Domain Uzunluğu
    # ═══════════════════════════════════════════════
    # Uzun domain isimleri genellikle meşru sitelerde görülmez.
    # "paypal-secure-account-verification-update.com" gibi uzun domain'ler
    # phishing belirtisi olabilir.
    # Zaman: O(1)
    features['domain_length'] = len(domain)

    # ═══════════════════════════════════════════════
    # ÖZNİTELİK 3: IP Adresi Kullanımı
    # ═══════════════════════════════════════════════
    # Meşru siteler neredeyse hiçbir zaman IP adresi kullanmaz.
    # "http://192.168.1.1/login" gibi URL'ler büyük olasılıkla phishing'dir.
    # Regex ile IPv4 formatını kontrol ediyoruz: X.X.X.X (0-255 arası sayılar)
    # Zaman: O(d), d = domain uzunluğu
    ip_pattern = re.compile(
        r'^(\d{1,3}\.){3}\d{1,3}$'   # IPv4 pattern: 3 adet "rakam." + son rakam
    )
    features['has_ip'] = 1 if ip_pattern.match(domain) else 0  # 1=IP var, 0=yok

    # ═══════════════════════════════════════════════
    # ÖZNİTELİK 4: '@' İşareti Varlığı
    # ═══════════════════════════════════════════════
    # Web tarayıcıları URL'deki '@' işaretinden ÖNCEKI kısmı yok sayar!
    # Örnek: http://www.paypal.com@hacker.com/login
    # → Tarayıcı aslında "hacker.com/login" adresine gider!
    # Bu, kullanıcıyı yanıltmak için kullanılan klasik bir phishing tekniğidir.
    # Zaman: O(n) — 'in' operatörü string'i baştan sona tarar
    features['has_at_sign'] = 1 if '@' in url else 0

    # ═══════════════════════════════════════════════
    # ÖZNİTELİK 5: Çift Slash (//) Yönlendirmesi
    # ═══════════════════════════════════════════════
    # Normal bir URL'de '//' sadece protokolden sonra (https://) bulunur.
    # URL'de birden fazla '//' varsa, yönlendirme manipülasyonu yapılıyor olabilir.
    # Örnek: http://example.com//http://malicious.com/steal
    # Zaman: O(n) — count() fonksiyonu tüm string'i tarar
    features['has_double_slash'] = 1 if url.count('//') > 1 else 0

    # ═══════════════════════════════════════════════
    # ÖZNİTELİK 6: Domain'de Tire (-) İşareti
    # ═══════════════════════════════════════════════
    # Meşru siteler nadiren domain'de '-' kullanır.
    # Phishing siteleri "-" ile marka adlarını taklit eder:
    # "paypal-secure-login.com", "google-account-verify.com"
    # Zaman: O(d) — domain string'inde arama
    features['has_dash'] = 1 if '-' in domain else 0

    # ═══════════════════════════════════════════════
    # ÖZNİTELİK 7: HTTPS Protokolü
    # ═══════════════════════════════════════════════
    # HTTPS, SSL/TLS sertifikası ile şifreli bağlantı sağlar.
    # Meşru siteler genellikle HTTPS kullanır.
    # HTTP kullanan siteler daha risklidir (veriler şifresiz iletilir).
    # Not: Bazı phishing siteleri de HTTPS kullanabilir, bu yüzden
    # tek başına güvenlik garantisi değildir.
    # Zaman: O(1) — sabit uzunlukta string karşılaştırma
    features['has_https'] = 1 if parsed.scheme == 'https' else 0

    # ═══════════════════════════════════════════════
    # ÖZNİTELİK 8: Nokta (.) Sayısı
    # ═══════════════════════════════════════════════
    # Normal URL'lerde 2-3 nokta bulunur (www.example.com).
    # Çok fazla nokta = çok fazla subdomain = şüpheli
    # Örnek: "login.paypal.com.secure.verify.hacker.tk" (6 nokta!)
    # Zaman: O(n) — tüm string'de '.' sayımı
    features['num_dots'] = url.count('.')

    # ═══════════════════════════════════════════════
    # ÖZNİTELİK 9: Rakam Sayısı
    # ═══════════════════════════════════════════════
    # Phishing URL'lerinde rastgele rakamlar sıkça görülür:
    # "http://secure123.login456.verify789.com"
    # Meşru URL'lerde genellikle az sayıda rakam bulunur.
    # Zaman: O(n) — her karakter isdigit() ile kontrol edilir
    # Generator expression kullanıldı (memory-efficient)
    features['num_digits'] = sum(c.isdigit() for c in url)

    # ═══════════════════════════════════════════════
    # ÖZNİTELİK 10: Özel Karakter Sayısı
    # ═══════════════════════════════════════════════
    # URL'deki ?, &, =, %, #, @, !, ~, *, +, ^ gibi özel karakterler.
    # Fazla özel karakter = URL manipülasyonu belirtisi olabilir.
    # Regex ile tek seferde sayılır.
    # Zaman: O(n) — regex tüm string'i tek geçişte tarar
    features['num_special_chars'] = len(re.findall(r'[?&=%#@!~\*\+\^]', url))

    # ═══════════════════════════════════════════════
    # ÖZNİTELİK 11: Alt Domain (Subdomain) Sayısı
    # ═══════════════════════════════════════════════
    # Subdomain derinliği: domain'deki noktalara göre hesaplanır.
    # "www.google.com" → subdomain=0 (normal)
    # "login.secure.paypal.com.fake.tk" → subdomain=4 (çok şüpheli!)
    # "www" öneki hesaplamadan çıkarılır (yaygın ama anlamsız).
    # Zaman: O(d) — d = domain parça sayısı
    domain_parts = domain.split('.')          # Domain'i noktalara göre böl
    if domain_parts and domain_parts[0] == 'www':  # "www" varsa çıkar
        domain_parts = domain_parts[1:]
    # Toplam parça sayısından TLD ve ana domain çıkarılır (2 çıkar)
    features['subdomain_count'] = max(0, len(domain_parts) - 2)

    # ═══════════════════════════════════════════════
    # ÖZNİTELİK 12: Path Uzunluğu
    # ═══════════════════════════════════════════════
    # URL'nin path kısmının uzunluğu (/login/secure/verify/account).
    # Uzun path'ler, phishing sayfalarının derin klasör yapılarında
    # gizlendiğini gösterebilir.
    # Zaman: O(1)
    features['path_length'] = len(path)

    # ═══════════════════════════════════════════════
    # ÖZNİTELİK 13: Sorgu Parametresi Sayısı
    # ═══════════════════════════════════════════════
    # URL'deki ?key=value&key2=value2 formatındaki parametrelerin sayısı.
    # Aşırı parametre = veri toplama / izleme amaçlı olabilir.
    # parse_qs() fonksiyonu query string'i dict'e dönüştürür.
    # Zaman: O(q), q = query string uzunluğu
    features['num_params'] = len(parse_qs(query))

    # ═══════════════════════════════════════════════
    # ÖZNİTELİK 14: URL Kısaltma Servisi Kontrolü
    # ═══════════════════════════════════════════════
    # bit.ly, tinyurl.com gibi servisler gerçek URL'yi gizler.
    # Saldırganlar phishing linkini kısaltarak kullanıcının
    # linkin nereye gittiğini görmesini engeller.
    # Zaman: O(s * d) — s = servis listesi boyutu (sabit ~15),
    #                   d = domain uzunluğu → pratikte O(1)
    features['has_shortener'] = 1 if any(
        s in domain.lower() for s in SHORTENING_SERVICES
    ) else 0

    # ═══════════════════════════════════════════════
    # ÖZNİTELİK 15: TLD Uzunluğu
    # ═══════════════════════════════════════════════
    # TLD (Top-Level Domain) = domain'in son kısmı (.com, .org, .tk).
    # Meşru TLD'ler genellikle 2-3 karakter (com, org, net).
    # Uzun veya sıradışı TLD'ler (.download, .cricket) phishing belirtisi.
    # Zaman: O(d) — split ve son eleman erişimi
    tld = domain.split('.')[-1] if '.' in domain else ''  # Son parça = TLD
    features['tld_length'] = len(tld)

    # ═══════════════════════════════════════════════
    # ÖZNİTELİK 16: Şüpheli TLD Kontrolü
    # ═══════════════════════════════════════════════
    # Bazı TLD'ler ücretsiz sunulur (.tk, .ml, .ga, .cf, .gq).
    # Bu TLD'lerin phishing/malware'de kötüye kullanım oranı %90+'dır.
    # SUSPICIOUS_TLDS listesindeki TLD'lerle karşılaştırılır.
    # Zaman: O(t) — t = şüpheli TLD listesi boyutu (sabit ~19) → O(1)
    features['suspicious_tld'] = 1 if tld.lower() in SUSPICIOUS_TLDS else 0

    return features  # 16 öznitelik içeren dict döndür


def extract_features_batch(urls: list) -> list:
    """
    Birden fazla URL için toplu (batch) feature extraction.

    Parameters
    ----------
    urls : list of str
        URL listesi.

    Returns
    -------
    list of dict
        Her URL için bir öznitelik sözlüğü.

    Time Complexity: O(m * n_avg)
        m = URL sayısı, n_avg = ortalama URL uzunluğu
    Space Complexity: O(m * k)
        k = öznitelik sayısı (16)
    """
    # Her URL için extract_features() çağrılır — list comprehension
    return [extract_features(url) for url in urls]


# ──────────────────────────────────────────────
# Feature isimleri (sıralı liste)
# ──────────────────────────────────────────────
# Model eğitim/tahmin sırasında özniteliklerin
# tutarlı sırada olmasını garanti eder.
FEATURE_NAMES = [
    'url_length',          # 1.  URL toplam uzunluğu
    'domain_length',       # 2.  Domain adı uzunluğu
    'has_ip',              # 3.  IP adresi var mı (0/1)
    'has_at_sign',         # 4.  '@' işareti var mı (0/1)
    'has_double_slash',    # 5.  Çift '//' var mı (0/1)
    'has_dash',            # 6.  Domain'de '-' var mı (0/1)
    'has_https',           # 7.  HTTPS kullanılıyor mu (0/1)
    'num_dots',            # 8.  Nokta sayısı
    'num_digits',          # 9.  Rakam sayısı
    'num_special_chars',   # 10. Özel karakter sayısı
    'subdomain_count',     # 11. Alt domain sayısı
    'path_length',         # 12. Path uzunluğu
    'num_params',          # 13. Sorgu parametre sayısı
    'has_shortener',       # 14. URL kısaltma servisi mi (0/1)
    'tld_length',          # 15. TLD uzunluğu
    'suspicious_tld',      # 16. Şüpheli TLD mi (0/1)
]


# ──────────────────────────────────────────────
# Doğrudan çalıştırma testi
# ──────────────────────────────────────────────
if __name__ == '__main__':
    # Test URL'leri — hem meşru hem phishing örnekleri
    test_urls = [
        'https://www.google.com/search?q=python',                        # Meşru
        'http://192.168.1.1/admin/login.php',                            # IP adresi
        'http://paypal-secure-login.com/update@info.html',               # @ + dash
        'https://bit.ly/3xYz123',                                       # Kısaltma servisi
        'http://free-iphone.tk/win-now/claim?id=12345&ref=abc',          # Şüpheli TLD
    ]

    # Her URL'yi analiz et ve sonuçları yazdır
    for url in test_urls:
        feats = extract_features(url)  # Öznitelikleri çıkar
        print(f"\nURL: {url}")
        for k, v in feats.items():     # Her özniteliği yazdır
            print(f"  {k}: {v}")
