"""PhishGuard: an analysis dashboard with explicit model setup."""
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys
import tempfile
from urllib.parse import urlparse

import altair as alt
import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from app.ui import DARK, LIGHT, csv_bytes, empty_state, hero, panel_header, signals, stylesheet
from src.feature_extraction import FEATURE_NAMES
from src.predict import predict_url

st.set_page_config(
    page_title="PhishGuard AI · URL intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def read_bundle(path, modified):
    return joblib.load(path)


def load_bundle():
    path = PROJECT_ROOT / "models/model_bundle.joblib"
    if not path.exists():
        return None, None
    signature = path.stat().st_mtime_ns
    return read_bundle(str(path), signature), signature


def set_example(url):
    st.session_state["scan_url"] = url


def train_demo():
    from src.generate_dataset import generate_dataset, save_dataset
    from src.preprocessing import prepare_data
    from src.train import train_and_evaluate

    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "demo.csv")
        save_dataset(generate_dataset(), path)
        Xtr, Xte, ytr, yte, _, groups = prepare_data(
            path, save_processed=False, return_groups=True
        )
        train_and_evaluate(
            Xtr, Xte, ytr, yte, cv_folds=3,
            groups_train=groups, dataset_kind="synthetic_demo",
        )
    read_bundle.clear()


with st.sidebar:
    st.markdown(
        '<div class="pg-brand"><span class="pg-brand-mark">P</span>'
        '<div class="pg-brand-name">PhishGuard<small>AI WORKSPACE</small></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="pg-nav-label">WORKSPACE</div>', unsafe_allow_html=True)
    page = st.radio(
        "Workspace navigation",
        ["URL analysis", "Batch review", "Model performance", "Scan history"],
        label_visibility="collapsed", key="workspace_page",
    )
    st.divider()
    dark = st.toggle("Dark appearance", value=False, key="dark_appearance")
    with st.expander("Model settings"):
        st.caption("Demo training uses synthetic data and replaces the local model bundle. Predictions become available once setup finishes.")
        if st.button("Train a synthetic demo", width="stretch", key="train_demo"):
            try:
                with st.spinner("Preparing your workspace…"):
                    train_demo()
                st.success("Demo model ready.")
            except Exception as error:
                st.error(str(error))
        st.caption("For a reviewed dataset, use the training command in the README.")
        st.link_button("Training guide", "https://github.com/orkun022/PhishGuard-AI#train-with-your-own-data", width="stretch")
    st.caption("LOCAL PREDICTIONS")
    st.caption("Submitted addresses are not opened. Session history is kept only for this session.")
    st.link_button("View source code", "https://github.com/orkun022/PhishGuard-AI", width="stretch")

st.markdown(stylesheet(dark), unsafe_allow_html=True)
palette = DARK if dark else LIGHT
try:
    bundle, signature = load_bundle()
except (ValueError, OSError, KeyError):
    bundle, signature = None, None
    st.error("The saved model could not be loaded. Open Model settings to set up a new model.")
if st.session_state.get("model_signature") != signature:
    for key in ["last_result", "batch_results"]:
        st.session_state.pop(key, None)
    st.session_state["model_signature"] = signature
metadata = bundle["metadata"] if bundle else None
status = "Synthetic demo" if metadata and metadata["dataset_kind"] == "synthetic_demo" else "Model ready" if bundle else "Setup needed"
st.markdown(
    '<div class="pg-topline"><span>Workspace <span aria-hidden="true">/</span> <strong>'
    + escape(page) + '</strong></span><span class="pg-status"><i></i>'
    + escape(status) + '</span></div>', unsafe_allow_html=True,
)
if metadata and metadata["dataset_kind"] == "synthetic_demo":
    st.caption("DEMO DATA · This model uses generated examples. Its scores do not measure real-world phishing detection.")

history = st.session_state.get("history", [])
model_name = metadata["selected_model"] if metadata else "Not configured"


def render_result(result):
    with st.container(border=True, key="result-panel"):
        flagged = bool(result["prediction"])
        title = "Review recommended" if flagged else "No phishing pattern predicted"
        subtitle = "The model classified this address as phishing." if flagged else "The model classified this address as legitimate."
        st.markdown(
            '<div class="pg-verdict' + (' flagged' if flagged else '')
            + '"><span class="pg-eyebrow">ANALYSIS RESULT</span><h2>'
            + title + '</h2><p>' + subtitle + '</p></div>', unsafe_allow_html=True,
        )
        st.markdown('<div class="pg-result-url">' + escape(result["url"]) + '</div>', unsafe_allow_html=True)
        left, right = st.columns([2, 1])
        with left:
            st.caption("PREDICTED CLASS MODEL SCORE")
            st.progress(min(1.0, max(0.0, result["confidence"] / 100)))
        with right:
            st.markdown(f"### {result['confidence']:.1f}%")
        st.caption("An uncalibrated model score, not a safety guarantee. Structural signals below describe the URL; they are not model explanations.")
        st.markdown(signals(result["features"]), unsafe_allow_html=True)
        with st.expander("Explore all 16 URL features"):
            features = pd.DataFrame({"Feature": FEATURE_NAMES, "Value": [result["features"][name] for name in FEATURE_NAMES]})
            st.dataframe(features, hide_index=True, width="stretch")
        with st.expander("Optional reputation lookup"):
            st.caption("This explicitly sends the selected URL to Google Safe Browsing. No known match does not establish safety.")
            key = st.text_input("Google Safe Browsing API key", type="password", key="reputation_key")
            if st.button("Check reputation", disabled=not key, key="check_reputation"):
                from src.feedback_loop import check_url_safe_browsing
                st.info(check_url_safe_browsing(result["url"], key)["details"])


if page == "URL analysis":
    st.markdown(hero("A closer look.<br><span>Before the click.</span>", "Inspect a web address, understand its structural signals, and review what the model predicts."), unsafe_allow_html=True)
    metrics = st.columns(4)
    metrics[0].metric("Scans this session", st.session_state.get("scan_count", len(history)))
    metrics[1].metric("Flagged this session", st.session_state.get("flagged_count", sum(row.get("label") == "Phishing" for row in history)))
    metrics[2].metric("URL features", "16")
    metrics[3].metric("Active model", model_name)
    st.markdown('<div class="pg-section-gap"></div>', unsafe_allow_html=True)
    main, aside = st.columns([2.1, 1], gap="large")
    with main:
        with st.container(border=True, key="analysis-panel"):
            st.markdown(panel_header("Inspect an address", "SINGLE URL"), unsafe_allow_html=True)
            with st.form("single-url"):
                address = st.text_input("Web address", placeholder="https://example.com", key="scan_url")
                st.caption("HTTP and HTTPS addresses only. Bare hostnames default to HTTPS.")
                submitted = st.form_submit_button("Analyze address", type="primary", width="stretch", disabled=not bundle)
            a, b = st.columns(2)
            a.button("Try a standard address", on_click=set_example, args=("https://example.com/docs",), width="stretch")
            b.button("Try an IP address", on_click=set_example, args=("http://192.0.2.10/login",), width="stretch")
            if not bundle:
                st.info("Open Model settings in the sidebar and train a demo to enable analysis.")
            if submitted and bundle:
                try:
                    with st.spinner("Extracting features and running the model…"):
                        result = predict_url(address, bundle["model"])
                    result["scanned_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    result["model"] = model_name
                    st.session_state["last_result"] = result
                    history = st.session_state.setdefault("history", [])
                    history.insert(0, {key: result[key] for key in ["url", "label", "confidence", "scanned_at", "model"]})
                    del history[20:]
                    st.session_state["scan_count"] = st.session_state.get("scan_count", 0) + 1
                    st.session_state["flagged_count"] = st.session_state.get("flagged_count", 0) + result["prediction"]
                    st.rerun()
                except ValueError as error:
                    st.session_state.pop("last_result", None)
                    st.error(str(error))
            if not st.session_state.get("last_result"):
                st.markdown(empty_state("Ready when you are", "Paste an address above. Your result and its URL features will appear here."), unsafe_allow_html=True)
        if st.session_state.get("last_result"):
            render_result(st.session_state["last_result"])
    with aside:
        with st.container(border=True, key="explainer-panel"):
            st.markdown(panel_header("Inside the analysis"), unsafe_allow_html=True)
            for number, title, description in [
                ("01", "Read the structure", "Extract 16 features from the address, including hostname, length, and protocol."),
                ("02", "Run the classifier", "Apply the selected model and its fitted preprocessing pipeline."),
                ("03", "Review the details", "Inspect the prediction, model score, and individual URL features."),
            ]:
                st.markdown('<div class="pg-step"><b>' + number + '</b><div><strong>' + title + '</strong><p>' + description + '</p></div></div>', unsafe_allow_html=True)
            st.markdown('<div class="pg-note">Local prediction does not open the submitted link, follow redirects, or inspect page contents.</div>', unsafe_allow_html=True)
        if metadata:
            with st.expander("Current model"):
                st.write(model_name)
                st.caption(f"{metadata['train_samples']:,} training examples · {metadata['test_samples']:,} holdout examples")
                st.caption("Selected using training cross-validation F1.")

elif page == "Batch review":
    st.markdown(hero("One list.<br><span>A clearer overview.</span>", "Review multiple addresses together, find rows that need attention, and take the results with you.", "BATCH ANALYSIS"), unsafe_allow_html=True)
    with st.container(border=True, key="batch-panel"):
        st.markdown(panel_header("Upload your URL list", "CSV · UP TO 1,000 ROWS"), unsafe_allow_html=True)
        st.caption("Include a url column. Maximum file size: 2 MB. Invalid addresses are reported individually.")
        st.download_button("Download CSV template", "url\nhttps://example.com\n", "url-template.csv", "text/csv", key="batch_template")
        uploaded = st.file_uploader("URL CSV", type=["csv"], key="url_csv")
        if st.button("Analyze batch", type="primary", disabled=not uploaded or not bundle, key="analyze_batch"):
            try:
                if uploaded.size > 2_000_000:
                    raise ValueError("The CSV must be smaller than 2 MB.")
                frame = pd.read_csv(uploaded)
                if "url" not in frame or frame.empty or len(frame) > 1000:
                    raise ValueError("Provide 1–1,000 rows with a url column.")
                rows = []
                bar = st.progress(0, text="Reviewing addresses…")
                for index, value in enumerate(frame["url"]):
                    try:
                        result = predict_url(value, bundle["model"])
                        rows.append({"url": result["url"], "prediction": result["label"], "model_score": result["confidence"], "error": ""})
                    except ValueError as error:
                        rows.append({"url": str(value), "prediction": "", "model_score": None, "error": str(error)})
                    bar.progress((index + 1) / len(frame), text=f"Reviewed {index + 1} of {len(frame)} addresses")
                st.session_state["batch_results"] = pd.DataFrame(rows)
                bar.empty()
            except (ValueError, pd.errors.ParserError) as error:
                st.session_state.pop("batch_results", None)
                st.error(str(error))
        if not bundle:
            st.info("Set up a model in the sidebar to enable batch analysis.")
        if "batch_results" not in st.session_state:
            st.markdown(empty_state("Your batch review starts here", "Upload a CSV to see predictions, model scores, and validation errors in one table."), unsafe_allow_html=True)
    if "batch_results" in st.session_state:
        frame = st.session_state["batch_results"]
        a, b, c = st.columns(3)
        a.metric("Addresses reviewed", len(frame))
        b.metric("Flagged predictions", int(frame["prediction"].eq("Phishing").sum()))
        c.metric("Invalid addresses", int(frame["error"].ne("").sum()))
        st.dataframe(frame, hide_index=True, width="stretch")
        st.download_button("Download review CSV", csv_bytes(frame), "phishguard-results.csv", "text/csv", key="batch_export")

elif page == "Model performance":
    st.markdown(hero("Measured results.<br><span>Visible trade-offs.</span>", "Compare models on the same holdout and inspect how the selected model made the cut.", "MODEL EVALUATION"), unsafe_allow_html=True)
    if not bundle:
        st.info("Set up a model to view its measured evaluation results.")
    else:
        results = metadata["results"]
        selected = results[model_name]
        columns = st.columns(4)
        for column, (label, key) in zip(columns, [("Holdout F1", "f1"), ("Holdout recall", "recall"), ("Holdout precision", "precision"), ("Training CV F1", "cv_mean")]):
            column.metric(label, f"{selected[key]:.1%}")
        st.markdown('<div class="pg-section-gap"></div>', unsafe_allow_html=True)
        with st.container(border=True, key="model-panel"):
            st.markdown(panel_header("Model comparison", "HOLDOUT METRICS"), unsafe_allow_html=True)
            scores = pd.DataFrame([{ "Model": name, "Metric": metric, "Score": values[key] } for name, values in results.items() for metric, key in [("F1", "f1"), ("Recall", "recall")]])
            chart = alt.Chart(scores).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("Model:N", axis=alt.Axis(labelAngle=0, labelColor=palette["muted"], title=None)),
                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%", labelColor=palette["muted"], title=None, gridColor=palette["border"])),
                xOffset="Metric:N",
                color=alt.Color("Metric:N", scale=alt.Scale(domain=["F1", "Recall"], range=["#19836e", "#a4d9c9"]), legend=alt.Legend(title=None, orient="top", labelColor=palette["muted"])),
                tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".2%")],
            ).properties(height=280, background=palette["surface"]).configure_view(strokeWidth=0)
            st.altair_chart(chart, width="stretch", theme=None)
            st.caption("The selected model uses training cross-validation F1, not the holdout results shown above.")
        with st.expander("Detailed metrics and confusion matrix", expanded=False):
            summary = pd.DataFrame({name: {key: value for key, value in values.items() if key != "confusion_matrix"} for name, values in results.items()}).T
            st.dataframe(summary, width="stretch")
            st.caption("Selected model confusion matrix · rows: actual, columns: predicted")
            st.dataframe(pd.DataFrame(selected["confusion_matrix"], index=["Legitimate", "Phishing"], columns=["Legitimate", "Phishing"]), width="stretch")
        st.caption("The holdout separates normalized hostnames. Related subdomains and repeated templates can remain correlated; this is not a registered-domain or time-based benchmark.")
        st.download_button("Download evaluation JSON", json.dumps(metadata, indent=2), "phishguard-metrics.json", "application/json", key="evaluation_export")

else:
    st.markdown(hero("Your recent reviews.<br><span>All in one place.</span>", "Revisit the last 20 addresses analyzed in this session, including the model used for each prediction.", "SESSION HISTORY"), unsafe_allow_html=True)
    with st.container(border=True, key="history-panel"):
        st.markdown(panel_header("Recent addresses", "SESSION ONLY"), unsafe_allow_html=True)
        if not history:
            st.markdown(empty_state("No reviews yet", "Analyze an address to start building your session history."), unsafe_allow_html=True)
        else:
            history_frame = pd.DataFrame(history).rename(columns={"confidence": "model_score"})
            st.dataframe(history_frame, hide_index=True, width="stretch")
            a, b = st.columns(2)
            a.download_button("Export session CSV", csv_bytes(history_frame), "phishguard-session.csv", "text/csv", width="stretch")
            if b.button("Clear session history", width="stretch"):
                for key in ["history", "last_result", "scan_count", "flagged_count"]:
                    st.session_state.pop(key, None)
                st.rerun()
        st.caption("History belongs to this Streamlit session. It is not saved as a permanent scan log.")

st.markdown('<footer class="pg-footer"><span>PhishGuard AI · URL intelligence workspace</span><span>Understand the result. Review the context.</span></footer>', unsafe_allow_html=True)
