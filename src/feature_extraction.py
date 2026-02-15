import re
from urllib.parse import urlparse, parse_qs

SHORTENING_SERVICES = [
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly',
    'is.gd', 'buff.ly', 'adf.ly', 'cutt.ly', 'rb.gy',
    'short.io', 'tiny.cc', 'lnkd.in', 'db.tt', 'qr.ae',
]

SUSPICIOUS_TLDS = [
    'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'buzz',
    'club', 'work', 'date', 'racing', 'win', 'bid',
    'stream', 'download', 'cricket', 'party', 'science',
]


def extract_features(url):
    features = {}

    try:
        parsed = urlparse(url)
    except Exception:
        parsed = urlparse('')

    domain = parsed.netloc or ''
    path = parsed.path or ''
    query = parsed.query or ''

    features['url_length'] = len(url)
    features['domain_length'] = len(domain)

    ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
    features['has_ip'] = 1 if ip_pattern.match(domain) else 0

    features['has_at_sign'] = 1 if '@' in url else 0
    features['has_double_slash'] = 1 if url.count('//') > 1 else 0
    features['has_dash'] = 1 if '-' in domain else 0
    features['has_https'] = 1 if parsed.scheme == 'https' else 0
    features['num_dots'] = url.count('.')
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_special_chars'] = len(re.findall(r'[?&=%#@!~\*\+\^]', url))

    domain_parts = domain.split('.')
    if domain_parts and domain_parts[0] == 'www':
        domain_parts = domain_parts[1:]
    features['subdomain_count'] = max(0, len(domain_parts) - 2)

    features['path_length'] = len(path)
    features['num_params'] = len(parse_qs(query))

    features['has_shortener'] = 1 if any(
        s in domain.lower() for s in SHORTENING_SERVICES
    ) else 0

    tld = domain.split('.')[-1] if '.' in domain else ''
    features['tld_length'] = len(tld)
    features['suspicious_tld'] = 1 if tld.lower() in SUSPICIOUS_TLDS else 0

    return features


def extract_features_batch(urls):
    return [extract_features(url) for url in urls]


FEATURE_NAMES = [
    'url_length', 'domain_length', 'has_ip', 'has_at_sign', 'has_double_slash',
    'has_dash', 'has_https', 'num_dots', 'num_digits', 'num_special_chars',
    'subdomain_count', 'path_length', 'num_params', 'has_shortener',
    'tld_length', 'suspicious_tld'
]


if __name__ == '__main__':
    test_urls = [
        'https://www.google.com/search?q=python',
        'http://192.168.1.1/admin/login.php',
        'http://paypal-secure-login.com/update@info.html',
        'https://bit.ly/3xYz123',
        'http://free-iphone.tk/win-now/claim?id=12345&ref=abc',
    ]

    for url in test_urls:
        feats = extract_features(url)
        print(f"\nURL: {url}")
        for k, v in feats.items():
            print(f"  {k}: {v}")
