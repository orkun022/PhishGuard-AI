import numpy as np
import pandas as pd
import pytest
from src.feature_extraction import extract_features, normalize_url
from src.preprocessing import load_data, split_by_hostname
from src.feedback_loop import _heuristic_verification, auto_retrain_if_ready, check_url_safe_browsing

def test_hostname_ignores_credentials_and_port():
    assert extract_features('http://user:pass@192.0.2.5:8080/login')['has_ip'] == 1
    assert extract_features('http://[2001:db8::1]:8080/')['has_ip'] == 1
    assert extract_features('https://notbit.ly/a')['has_shortener'] == 0
    assert extract_features('https://bit.ly.evil.test/a')['has_shortener'] == 0
    assert extract_features('https://BIT.LY/a')['has_shortener'] == 1

@pytest.mark.parametrize('url', ['', 'javascript:alert(1)', 'ftp://example.com', 'https://', 'a b.com', 'https://example.com:bad'])
def test_invalid_web_input_is_rejected(url):
    with pytest.raises(ValueError): normalize_url(url)

def test_labels_are_explicit_and_conflicts_rejected(tmp_path):
    path=tmp_path/'urls.csv'
    pd.DataFrame({'url':['https://a.test','https://b.test'], 'label':['legitimate','unknown']}).to_csv(path,index=False)
    with pytest.raises(ValueError,match='Labels'): load_data(path)
    pd.DataFrame({'url':['https://a.test','https://a.test'], 'label':[0,1]}).to_csv(path,index=False)
    with pytest.raises(ValueError,match='conflicting'): load_data(path)

def test_holdout_has_no_training_hostname_overlap():
    df=pd.DataFrame([{'url':f'https://host{i}.test/path{j}','label':i%2} for i in range(20) for j in range(3)])
    train,test,groups=split_by_hostname(df)
    assert set(groups[train]).isdisjoint(groups[test])
    assert set(df.iloc[test]['label']) == {0,1}

def test_heuristics_and_no_match_are_not_training_truth(monkeypatch):
    result=_heuristic_verification('https://notgoogle.com/login')
    assert result['is_malicious'] is None
    assert result['verified'] is False
    assert not auto_retrain_if_ready(force=True)['retrained']
    class Response:
        def raise_for_status(self): pass
        def json(self): return {}
    monkeypatch.setattr('requests.post',lambda *a,**kw:Response())
    result=check_url_safe_browsing('https://example.com','test-key')
    assert result['is_malicious'] is None
    assert result['verified'] is False


def test_model_selection_uses_cv_not_holdout(tmp_path, monkeypatch):
    import joblib
    from sklearn.dummy import DummyClassifier
    from src import train
    monkeypatch.setattr(train, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(train, 'cross_val_score', lambda model, *a, **kw: np.array([0.9, 0.9]) if model.constant == 0 else np.array([0.2, 0.2]))
    models={'CV winner': DummyClassifier(strategy='constant', constant=0), 'Holdout winner': DummyClassifier(strategy='constant', constant=1)}
    train.train_and_evaluate(np.zeros((8,16)),np.zeros((4,16)),np.array([0,1]*4),np.array([0,1,1,1]),models=models,cv_folds=2)
    bundle=joblib.load(tmp_path/'models/model_bundle.joblib')
    assert bundle['metadata']['selected_model'] == 'CV winner'
    assert bundle['metadata']['results']['Holdout winner']['f1'] > bundle['metadata']['results']['CV winner']['f1']
