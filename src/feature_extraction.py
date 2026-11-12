import ipaddress
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

    domain = (parsed.hostname or '').lower().rstrip('.')
    path = parsed.path or ''
    query = parsed.query or ''
    features['url_length'] = len(url)
    features['domain_length'] = len(domain)
    try:
        ipaddress.ip_address(domain)
        features['has_ip'] = 1
    except ValueError:
        features['has_ip'] = 0

    features['has_at_sign'] = 1 if '@' in url else 0
    features['has_double_slash'] = 1 if url.count('//') > 1 else 0
    features['has_dash'] = 1 if '-' in domain else 0
    features['has_https'] = 1 if parsed.scheme == 'https' else 0
    features['num_dots'] = url.count('.')
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_special_chars'] = len(re.findall(r'[?&=%#@!~\*\+\^]', url))

    domain_parts = [] if features['has_ip'] else domain.split('.')
    if domain_parts and domain_parts[0] == 'www':
        domain_parts = domain_parts[1:]
    features['subdomain_count'] = max(0, len(domain_parts) - 2)

    features['path_length'] = len(path)
    features['num_params'] = len(parse_qs(query))

    features['has_shortener'] = 1 if any(
        domain == s or domain.endswith('.' + s) for s in SHORTENING_SERVICES
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



def normalize_url(value):
    """Validate a web address without visiting it. Add HTTPS for bare hostnames."""
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError('Enter a non-empty web address (maximum 4096 characters).')
    value = value.strip()
    if re.search(r'[\s\\]', value):
        raise ValueError('Whitespace and backslashes are not allowed in a web address.')
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', value) or re.match(r'^[^/:]+:\d+(?:/|$)', value):
        value = 'https://' + value
    try:
        parsed = urlparse(value)
        host = parsed.hostname
        _ = parsed.port
        if parsed.scheme.lower() not in ('http', 'https') or not host:
            raise ValueError()
        ascii_host = host.encode('idna').decode('ascii')
        try:
            ipaddress.ip_address(ascii_host)
        except ValueError:
            if len(ascii_host) > 253 or '.' not in ascii_host or not all(
                re.fullmatch(r'[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?', label)
                for label in ascii_host.rstrip('.').split('.')
            ):
                raise ValueError()
    except (ValueError, UnicodeError):
        raise ValueError('Use a valid HTTP or HTTPS address with a hostname.') from None
    return value
