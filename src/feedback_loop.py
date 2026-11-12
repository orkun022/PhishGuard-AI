"""Optional reputation lookup; heuristic results never become training labels."""
from src.feature_extraction import normalize_url, extract_features

def _heuristic_verification(url):
    url = normalize_url(url)
    features = extract_features(url)
    reasons = []
    for key, description in [('has_ip', 'IP address host'), ('has_at_sign', 'Address contains @'), ('has_shortener', 'Shortened URL'), ('suspicious_tld', 'TLD risk signal')]:
        if features[key]:
            reasons.append(description)
    return {'is_malicious': None, 'verified': False, 'source': 'Local structural rules', 'confidence': None, 'threat_type': 'UNKNOWN', 'details': '; '.join(reasons) if reasons else 'No listed structural indicators; reputation remains unknown.'}

def check_url_safe_browsing(url, api_key=None):
    url = normalize_url(url)
    if not api_key:
        return _heuristic_verification(url)
    import requests
    payload = {'client': {'clientId': 'phishguard-ai', 'clientVersion': '2.0'}, 'threatInfo': {'threatTypes': ['MALWARE', 'SOCIAL_ENGINEERING', 'UNWANTED_SOFTWARE', 'POTENTIALLY_HARMFUL_APPLICATION'], 'platformTypes': ['ANY_PLATFORM'], 'threatEntryTypes': ['URL'], 'threatEntries': [{'url': url}]}}
    try:
        response = requests.post('https://safebrowsing.googleapis.com/v4/threatMatches:find', params={'key': api_key}, json=payload, timeout=8)
        response.raise_for_status()
        matches = response.json().get('matches', [])
        if matches:
            return {'is_malicious': True, 'verified': True, 'source': 'Google Safe Browsing API', 'confidence': None, 'threat_type': matches[0].get('threatType', 'UNKNOWN'), 'details': 'Known threat match: ' + matches[0].get('threatType', 'UNKNOWN')}
        return {'is_malicious': None, 'verified': False, 'source': 'Google Safe Browsing API', 'confidence': None, 'threat_type': 'UNKNOWN', 'details': 'No known threat match. This does not establish that the URL is safe.'}
    except (requests.RequestException, ValueError):
        return {'is_malicious': None, 'verified': False, 'source': 'Unavailable', 'confidence': None, 'threat_type': 'UNKNOWN', 'details': 'Reputation lookup unavailable. No training label was created.'}

def auto_retrain_if_ready(force=False):
    return {'retrained': False, 'message': 'Automatic feedback retraining is disabled. Train explicitly using a reviewed, labeled CSV.'}
