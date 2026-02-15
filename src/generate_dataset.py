import csv
import os
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEGITIMATE_DOMAINS = [
    'google.com', 'youtube.com', 'facebook.com', 'amazon.com',
    'wikipedia.org', 'twitter.com', 'instagram.com', 'linkedin.com',
    'reddit.com', 'netflix.com', 'github.com', 'stackoverflow.com',
    'microsoft.com', 'apple.com', 'medium.com', 'bbc.co.uk',
    'nytimes.com', 'cnn.com', 'spotify.com', 'dropbox.com',
    'slack.com', 'zoom.us', 'adobe.com', 'salesforce.com',
    'shopify.com', 'wordpress.org', 'mozilla.org', 'python.org',
    'coursera.org', 'udemy.com', 'kaggle.com', 'openai.com',
    'cloudflare.com', 'digitalocean.com', 'heroku.com',
    'paypal.com', 'stripe.com', 'twitch.tv', 'pinterest.com',
    'tumblr.com', 'quora.com', 'ebay.com', 'walmart.com',
    'target.com', 'bestbuy.com', 'etsy.com', 'airbnb.com',
    'booking.com', 'tripadvisor.com', 'yelp.com',
]

LEGITIMATE_PATHS = [
    '', '/', '/about', '/contact', '/help', '/support',
    '/products', '/services', '/blog', '/news', '/login',
    '/signup', '/pricing', '/features', '/docs', '/api',
    '/search', '/settings', '/profile', '/dashboard',
]

LEGITIMATE_PARAMS = [
    '', '?q=python', '?page=1', '?lang=en', '?ref=homepage',
    '?utm_source=google', '?id=12345', '?category=tech',
]

PHISHING_PATTERNS = [
    'http://192.168.{a}.{b}/login/secure',
    'http://10.0.{a}.{b}/paypal/signin',
    'http://172.16.{a}.{b}/bank/verify',
    'http://45.{a}.{b}.{c}/account-update',
    'http://185.{a}.{b}.{c}/secure-login.php',
    'http://paypal-secure-login.{tld}/update',
    'http://apple-id-verify.{tld}/confirm',
    'http://microsoft-account.{tld}/signin',
    'http://google-security.{tld}/alert',
    'http://amazon-delivery.{tld}/track',
    'http://netflix-billing.{tld}/payment',
    'http://facebook-security.{tld}/verify',
    'http://suspicious-{word1}-{word2}-{word3}.{tld}/login/secure/update/verify/account',
    'http://free-{word1}.{tld}/win-{word2}/claim-now?id={num1}&ref={num2}&token={num3}',
    'http://login.paypal.com.{random}.{tld}/signin',
    'http://secure.apple.com.{random}.{tld}/verify',
    'http://www.paypal.com@{random}.{tld}/login',
    'http://secure.bank.com@{random}.{tld}/transfer',
    'http://bit.ly/{short}',
    'http://tinyurl.com/{short}',
    'http://legitimate-site.com//http://{random}.{tld}/phishing',
    'http://{num1}{num2}{num3}.{tld}/verify-account/{num4}',
    'http://free-iphone.{suspicious_tld}/claim-now',
    'http://login-update.{suspicious_tld}/secure',
    'http://account-verify.{suspicious_tld}/confirm',
    'http://prize-winner.{suspicious_tld}/redeem',
    'http://security-alert.{suspicious_tld}/action',
]

PHISHING_WORDS = [
    'secure', 'login', 'update', 'verify', 'confirm',
    'account', 'billing', 'payment', 'bank', 'wallet',
    'crypto', 'free', 'prize', 'winner', 'urgent',
]

SUSPICIOUS_TLDS_LIST = ['tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'buzz', 'club', 'work']
NORMAL_TLDS = ['com', 'net', 'org', 'info', 'biz']

RANDOM_STRINGS = [
    'xk9f2m', 'abc123', 'qwerty', 'zz99yy', 'hack3r',
    'temp01', 'test99', 'user42', 'data77', 'web555',
    'srv001', 'node33', 'host88', 'proxy7', 'cdn456',
]


def generate_legitimate_url():
    scheme = random.choice(['https'] * 9 + ['http'])
    domain = random.choice(LEGITIMATE_DOMAINS)
    www = random.choice(['www.', ''] * 2 + [''])
    path = random.choice(LEGITIMATE_PATHS)
    params = random.choice(LEGITIMATE_PARAMS + [''] * 5)
    return f"{scheme}://{www}{domain}{path}{params}"


def generate_phishing_url():
    pattern = random.choice(PHISHING_PATTERNS)
    url = pattern.format(
        a=random.randint(1, 254), b=random.randint(1, 254),
        c=random.randint(1, 254),
        tld=random.choice(NORMAL_TLDS + SUSPICIOUS_TLDS_LIST),
        suspicious_tld=random.choice(SUSPICIOUS_TLDS_LIST),
        random=random.choice(RANDOM_STRINGS),
        word1=random.choice(PHISHING_WORDS), word2=random.choice(PHISHING_WORDS),
        word3=random.choice(PHISHING_WORDS),
        num1=random.randint(100, 999), num2=random.randint(100, 999),
        num3=random.randint(100, 999), num4=random.randint(100, 999),
        short=random.choice(RANDOM_STRINGS),
    )
    return url


def generate_dataset(n_legitimate=500, n_phishing=500, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n_legitimate):
        data.append((generate_legitimate_url(), 0))
    for _ in range(n_phishing):
        data.append((generate_phishing_url(), 1))
    random.shuffle(data)
    return data


def save_dataset(data, filepath=None):
    if filepath is None:
        filepath = os.path.join(PROJECT_ROOT, 'data', 'raw', 'phishing_urls.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'label'])
        writer.writerows(data)
    print(f"[+] Dataset kaydedildi: {filepath} ({len(data)} URL)")


def main():
    data = generate_dataset(n_legitimate=500, n_phishing=500)
    save_dataset(data)


if __name__ == '__main__':
    main()
