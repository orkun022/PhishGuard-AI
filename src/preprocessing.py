"""Validated URL data and hostname-separated holdout preparation."""
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from src.feature_extraction import extract_features, FEATURE_NAMES, normalize_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = {'0': 0, '1': 1, '0.0': 0, '1.0': 1, 'legitimate': 0, 'benign': 0, 'safe': 0, 'phishing': 1, 'malicious': 1}

def load_data(filepath=None):
    path = Path(filepath) if filepath else PROJECT_ROOT / 'data/raw/phishing_urls.csv'
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    if not {'url', 'label'} <= set(df):
        raise ValueError("CSV must contain url and label columns.")
    df = df[['url', 'label']].copy()
    if df.isna().any().any():
        raise ValueError('URL and label values cannot be missing.')
    df['url'] = df['url'].map(normalize_url)
    labels = df['label'].astype(str).str.strip().str.lower().map(LABELS)
    if labels.isna().any():
        raise ValueError('Labels must be 0/1, legitimate/phishing, benign/malicious, or safe/phishing.')
    df['label'] = labels.astype(int)
    if df.groupby('url')['label'].nunique().gt(1).any():
        raise ValueError('The same URL has conflicting labels.')
    df = df.drop_duplicates('url').reset_index(drop=True)
    if df['label'].nunique() != 2:
        raise ValueError('Both legitimate (0) and phishing (1) examples are required.')
    return df

def apply_feature_extraction(df):
    features = pd.DataFrame([extract_features(u) for u in df['url']], columns=FEATURE_NAMES)
    return pd.concat([df.reset_index(drop=True), features], axis=1)

def split_by_hostname(df, test_size=0.2, random_state=42):
    groups = df['url'].map(lambda u: (urlparse(u).hostname or '').lower().removeprefix('www.'))
    splitter = GroupShuffleSplit(n_splits=30, test_size=test_size, random_state=random_state)
    for train, test in splitter.split(df, df['label'], groups):
        if df.iloc[train]['label'].nunique() == 2 and df.iloc[test]['label'].nunique() == 2:
            return train, test, groups.to_numpy()
    raise ValueError('Need more distinct hostnames in each class for a two-class holdout.')

def prepare_data(filepath=None, test_size=0.2, random_state=42, scale=False, save_processed=True, return_groups=False):
    df = apply_feature_extraction(load_data(filepath))
    train, test, groups = split_by_hostname(df, test_size, random_state)
    if save_processed:
        path = PROJECT_ROOT / 'data/processed/features.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    X, y = df[FEATURE_NAMES].to_numpy(dtype=float), df['label'].to_numpy()
    X_train, X_test = X[train], X[test]
    scaler = None
    if scale:
        scaler = StandardScaler().fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    result = (X_train, X_test, y[train], y[test], scaler)
    return (*result, groups[train]) if return_groups else result
