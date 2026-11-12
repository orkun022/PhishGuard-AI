"""PhishGuard model workspace. Training is explicit; prediction stays local."""
from pathlib import Path
import sys
import tempfile
import joblib
import numpy as np
import pandas as pd
import streamlit as st
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.feature_extraction import extract_features, normalize_url, FEATURE_NAMES
from src.predict import predict_url
st.set_page_config(page_title='PhishGuard | URL intelligence', page_icon='◈', layout='wide')
st.markdown("""<style>
.block-container{max-width:1180px;padding-top:2.5rem}
.workspace-label{font-size:.72rem;letter-spacing:.18em;color:#7e91a8;font-weight:600;margin-bottom:.7rem}
.workspace-title{font-size:2.7rem;letter-spacing:-.07rem;line-height:1.15;font-weight:700;margin-bottom:1rem}
.workspace-intro{font-size:1.05rem;color:#9daec3;max-width:690px;margin-bottom:2rem}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def read_bundle(path, modified):
    return joblib.load(path)

def load_bundle():
    path = PROJECT_ROOT / 'models/model_bundle.joblib'
    return read_bundle(str(path), path.stat().st_mtime_ns) if path.exists() else None

with st.sidebar:
    st.title('◈ PhishGuard')
    st.caption('URL INTELLIGENCE WORKSPACE')
    st.divider()
    st.subheader('Model setup')
    st.caption('No model trains in the background. Demo data is synthetic and is only for exploring the workflow.')
    if st.button('Train a synthetic demo', width='stretch'):
        from src.generate_dataset import generate_dataset, save_dataset
        from src.preprocessing import prepare_data
        from src.train import train_and_evaluate
        try:
            with st.spinner('Training and evaluating on separate hostname groups…'):
                with tempfile.TemporaryDirectory() as folder:
                    path = str(Path(folder) / 'demo.csv')
                    save_dataset(generate_dataset(), path)
                    Xtr, Xte, ytr, yte, _, groups = prepare_data(path, save_processed=False, return_groups=True)
                    train_and_evaluate(Xtr, Xte, ytr, yte, cv_folds=3, groups_train=groups, dataset_kind='synthetic_demo')
            read_bundle.clear()
            st.success('Demo model ready.')
        except Exception as error:
            st.error(str(error))
    st.divider()
    st.markdown('[Source code](https://github.com/orkun022/PhishGuard-AI)')
    st.caption('Local predictions do not open the submitted addresses.')

st.markdown('<div class="workspace-label">SECURITY / URL ANALYSIS</div><div class="workspace-title">Inspect the address.<br>Understand the prediction.</div><p class="workspace-intro">A workspace for URL feature extraction, model comparison, and transparent evaluation.</p>', unsafe_allow_html=True)
bundle = load_bundle()
if not bundle:
    st.info('Train a demo from the sidebar, or run the documented training command with your labeled CSV.')
    st.code('python -m src.train --data data/raw/phishing_urls.csv --dataset-kind user_supplied')
    st.stop()
metadata = bundle['metadata']
if metadata['dataset_kind'] == 'synthetic_demo':
    st.warning('Synthetic demo model: these results do not measure real-world phishing detection performance.')
cols = st.columns(3)
cols[0].metric('Selected model', metadata['selected_model'])
cols[1].metric('Training samples', metadata['train_samples'])
cols[2].metric('Holdout samples', metadata['test_samples'])
scan, batch, evaluation = st.tabs(['Inspect a URL', 'Batch review', 'Evaluation'])
with scan:
    with st.form('single-url'):
        address = st.text_input('Web address', placeholder='https://example.com', help='HTTP/HTTPS only. Bare hostnames default to HTTPS.')
        submitted = st.form_submit_button('Analyze address', type='primary')
    if submitted:
        try:
            result = predict_url(address, bundle['model'])
            st.session_state['last_result'] = result
            history = st.session_state.setdefault('history', [])
            history.insert(0, {k: result[k] for k in ['url', 'label', 'confidence']})
            del history[20:]
        except ValueError as error:
            st.session_state.pop('last_result', None)
            st.error(str(error))
    result = st.session_state.get('last_result')
    if result:
        st.subheader('Phishing predicted' if result['prediction'] else 'Legitimate predicted')
        st.code(result['url'], language=None)
        st.metric('Predicted class model score', f"{result['confidence']:.1f}%")
        st.caption('This is an uncalibrated model score, not a safety guarantee. HTTPS and a low score cannot establish safety.')
        features = pd.DataFrame({'Feature': FEATURE_NAMES, 'Value': [result['features'][f] for f in FEATURE_NAMES]})
        with st.expander('View all 16 URL features'):
            st.dataframe(features, hide_index=True, width='stretch')
        with st.expander('Optional reputation lookup'):
            st.caption('This sends this URL to Google Safe Browsing. No match means no known match, not proof of safety.')
            key = st.text_input('Google Safe Browsing API key', type='password')
            if st.button('Check reputation', disabled=not key):
                from src.feedback_loop import check_url_safe_browsing
                check = check_url_safe_browsing(result['url'], key)
                st.info(check['details'])
    history = st.session_state.get('history', [])
    if history:
        with st.expander('Session history'):
            st.dataframe(history, hide_index=True, width='stretch')
            if st.button('Clear session history'):
                st.session_state.pop('history', None)
                st.session_state.pop('last_result', None)
                st.rerun()
with batch:
    st.subheader('Review a URL list')
    st.caption('Upload a CSV with a url column. Up to 1,000 rows and 2 MB; invalid rows are reported individually.')
    uploaded = st.file_uploader('URL CSV', type=['csv'])
    if uploaded and st.button('Analyze batch', type='primary'):
        try:
            if uploaded.size > 2_000_000:
                raise ValueError('The CSV must be smaller than 2 MB.')
            frame = pd.read_csv(uploaded)
            if 'url' not in frame or len(frame) > 1000 or frame.empty:
                raise ValueError('Provide 1–1,000 rows with a url column.')
            rows = []
            for value in frame['url']:
                try:
                    r = predict_url(value, bundle['model'])
                    rows.append({'url': r['url'], 'prediction': r['label'], 'model_score': r['confidence'], 'error': ''})
                except ValueError as error:
                    rows.append({'url': str(value), 'prediction': '', 'model_score': None, 'error': str(error)})
            st.session_state['batch_results'] = pd.DataFrame(rows)
        except (ValueError, pd.errors.ParserError) as error:
            st.error(str(error))
    if 'batch_results' in st.session_state:
        frame = st.session_state['batch_results']
        st.dataframe(frame, hide_index=True, width='stretch')
        export = frame.copy()
        export['url'] = export['url'].map(lambda s: "'" + s if s.startswith(('=', '+', '-', '@')) else s)
        st.download_button('Download results CSV', export.to_csv(index=False), 'phishguard-results.csv', 'text/csv')
with evaluation:
    st.subheader('Measured results')
    st.caption('Model selection uses training cross-validation F1. The hostname-separated holdout is used for reporting only. Related subdomains can still be correlated; this is not a registered-domain or time-based benchmark.')
    results = metadata['results']
    st.dataframe(pd.DataFrame({name: {k: v for k, v in scores.items() if k != 'confusion_matrix'} for name, scores in results.items()}).T, width='stretch')
    selected = results[metadata['selected_model']]
    st.caption('Holdout confusion matrix: rows = actual; columns = predicted.')
    st.dataframe(pd.DataFrame(selected['confusion_matrix'], index=['Legitimate', 'Phishing'], columns=['Legitimate', 'Phishing']))
    st.download_button('Download evaluation JSON', __import__('json').dumps(metadata, indent=2), 'phishguard-metrics.json', 'application/json')
