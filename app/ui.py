"""Presentation helpers for the PhishGuard analysis workspace."""
from html import escape

LIGHT = {
    "bg": "#f4f6f8", "surface": "#ffffff", "sidebar": "#fafbfc",
    "text": "#142529", "muted": "#617277", "border": "#dfe6e8",
    "accent": "#137664", "accent_soft": "#e6f3ee", "input": "#f8fafb",
}
DARK = {
    "bg": "#101b20", "surface": "#18272e", "sidebar": "#142229",
    "text": "#e9f2f1", "muted": "#a2b6ba", "border": "#30434a",
    "accent": "#8edbc0", "accent_soft": "#213f38", "input": "#122127",
}


def stylesheet(dark=False):
    palette = DARK if dark else LIGHT
    variables = ";".join(f"--pg-{key}:{value}" for key, value in palette.items())
    return "<style>" + ":root{" + variables + "}" + """
    .stApp { background:var(--pg-bg); color:var(--pg-text); }
    .stApp header[data-testid="stHeader"] { background:transparent; }
    .block-container { max-width:1250px; padding-top:2.6rem; padding-bottom:3rem; }
    [data-testid="stSidebar"] { background:var(--pg-sidebar); border-right:1px solid var(--pg-border); }
    [data-testid="stSidebar"] > div:first-child { padding-top:2rem; }
    .stApp h1,.stApp h2,.stApp h3,.stApp h4 { color:var(--pg-text); letter-spacing:-.035em; }
    .stApp p,.stApp label,.stApp [data-testid="stMetricLabel"] { color:var(--pg-text); }
    .stApp [data-testid="stCaptionContainer"] p { color:var(--pg-muted); line-height:1.65; }
    .stApp a { color:var(--pg-accent); }
    .stApp hr { border-color:var(--pg-border); }
    .stApp [data-testid="stMetric"] { background:var(--pg-surface); border:1px solid var(--pg-border); border-radius:14px; padding:18px 20px; min-height:112px; }
    .stApp [data-testid="stMetricLabel"] p { color:var(--pg-muted); font-size:12px; font-weight:550; }
    .stApp [data-testid="stMetricValue"] { color:var(--pg-text); font-size:clamp(18px,1.7vw,27px); line-height:1.2; white-space:normal; overflow-wrap:anywhere; letter-spacing:-.04em; }
    .stApp [data-testid="stVerticalBlockBorderWrapper"] > div { border-color:var(--pg-border); border-radius:16px; }
    .st-key-analysis-panel,.st-key-explainer-panel,.st-key-batch-panel,.st-key-model-panel,.st-key-history-panel,.st-key-result-panel { background:var(--pg-surface); }
    .stApp [data-testid="stForm"] { border:0; padding:0; }
    .stApp [data-baseweb="input"],.stApp [data-baseweb="base-input"],.stApp [data-baseweb="textarea"],.stApp [data-baseweb="select"] > div { background:var(--pg-input); border-color:var(--pg-border); border-radius:10px; }
    .stApp input,.stApp textarea { color:var(--pg-text)!important; }
    .stApp input::placeholder,.stApp textarea::placeholder { color:var(--pg-muted)!important; }
    .stApp button { border-radius:9px; font-size:13px; }
    .stApp button[kind="primary"],.stApp button[kind="primaryFormSubmit"] { background:var(--pg-accent); color:var(--pg-bg); border-color:var(--pg-accent); font-weight:650; }
    .stApp button[kind="primary"] p,.stApp button[kind="primaryFormSubmit"] p { color:var(--pg-bg); }
    .stApp button[kind="secondary"],.stApp button[kind="secondaryFormSubmit"] { background:var(--pg-surface); border-color:var(--pg-border); color:var(--pg-text); }
    .stApp button:focus-visible,.stApp a:focus-visible { outline:3px solid var(--pg-accent); outline-offset:3px; }
    .stApp [data-testid="stFileUploaderDropzone"] { background:var(--pg-input); border:1px dashed var(--pg-border); border-radius:12px; }
    .stApp [data-testid="stExpander"] details { background:var(--pg-surface); border-color:var(--pg-border); border-radius:12px; }
    .pg-brand { display:flex;align-items:center;gap:10px;margin:0 0 30px; }
    .pg-brand-mark { display:grid;place-items:center;width:36px;height:40px;background:var(--pg-accent);color:var(--pg-bg);font-size:22px;border-radius:11px 11px 16px 16px;font-weight:700; }
    .pg-brand-name { font-size:20px;font-weight:720;letter-spacing:-.7px;line-height:1.2;color:var(--pg-text); }
    .pg-brand-name small { display:block;font-size:9px;font-weight:600;letter-spacing:2px;color:var(--pg-muted);margin-top:5px; }
    .pg-nav-label,.pg-eyebrow { font-size:10px;letter-spacing:1.8px;font-weight:650;color:var(--pg-muted); }
    .pg-topline { display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:30px;padding-bottom:19px;border-bottom:1px solid var(--pg-border);font-size:12px;color:var(--pg-muted); }
    .pg-topline strong { color:var(--pg-text);font-weight:550; }
    .pg-status { display:inline-flex;align-items:center;gap:7px;padding:5px 10px;border:1px solid var(--pg-border);border-radius:20px;background:var(--pg-surface);font-size:11px;color:var(--pg-muted);white-space:nowrap; }
    .pg-status i { width:6px;height:6px;border-radius:50%;background:var(--pg-accent);display:block; }
    .pg-hero { margin:8px 0 26px; }
    .pg-hero h1 { font-size:clamp(31px,3.7vw,46px);line-height:1.12;letter-spacing:-1.9px;margin:10px 0 13px;font-weight:720; }
    .pg-hero h1 span { color:var(--pg-accent); }
    .pg-hero p { max-width:630px;font-size:14px;line-height:1.75;color:var(--pg-muted);margin:0; }
    .pg-panel-header { display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:18px; }
    .pg-panel-header h3 { font-size:17px;margin:0;letter-spacing:-.45px; }
    .pg-chip { display:inline-block;border:1px solid var(--pg-border);padding:4px 9px;border-radius:6px;font-size:10px;color:var(--pg-muted); }
    .pg-section-gap { height:22px; }
    .pg-empty { border:1px dashed var(--pg-border);border-radius:12px;min-height:155px;padding:27px;text-align:center;margin-top:20px; }
    .pg-empty .pg-empty-icon { display:inline-grid;place-items:center;width:35px;height:35px;border-radius:10px;background:var(--pg-accent-soft);color:var(--pg-accent);font-size:19px;margin-bottom:10px; }
    .pg-empty strong { display:block;font-size:13px;color:var(--pg-text);font-weight:600; }
    .pg-empty p { color:var(--pg-muted);font-size:12px;line-height:1.6;margin:7px 0 0; }
    .pg-step { display:flex;gap:13px;margin:20px 0; }
    .pg-step > b { color:var(--pg-accent);font-size:11px;background:var(--pg-accent-soft);width:28px;height:28px;display:grid;place-items:center;border-radius:8px;flex-shrink:0; }
    .pg-step strong { display:block;color:var(--pg-text);font-size:12px;font-weight:600;margin:0 0 5px; }
    .pg-step p { color:var(--pg-muted);font-size:12px;line-height:1.7;margin:0; }
    .pg-note { color:var(--pg-muted);font-size:11px;line-height:1.7;border-top:1px solid var(--pg-border);padding-top:15px;margin-top:21px; }
    .pg-verdict { border-left:3px solid var(--pg-accent);padding:3px 0 3px 17px;margin:6px 0 22px; }
    .pg-verdict.flagged { border-color:#d48045; }
    .pg-verdict h2 { font-size:23px;margin:5px 0 6px; }
    .pg-verdict p { font-size:12px;color:var(--pg-muted);margin:0; }
    .pg-result-url { font:12px/1.7 ui-monospace,monospace;overflow-wrap:anywhere;padding:12px 15px;background:var(--pg-input);border:1px solid var(--pg-border);border-radius:9px;margin-bottom:18px;color:var(--pg-text); }
    .pg-signals { display:flex;flex-wrap:wrap;gap:7px;margin:15px 0 22px; }
    .pg-signal { padding:5px 10px;border-radius:6px;background:var(--pg-input);border:1px solid var(--pg-border);font-size:11px;color:var(--pg-muted); }
    .pg-signal.detected { background:var(--pg-accent-soft);color:var(--pg-accent);border-color:transparent; }
    .pg-footer { display:flex;justify-content:space-between;gap:15px;border-top:1px solid var(--pg-border);padding-top:19px;margin-top:35px;font-size:10px;color:var(--pg-muted); }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding:10px 8px;border-radius:8px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background:var(--pg-accent-soft); }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p { font-size:13px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p { color:var(--pg-accent);font-weight:650; }
    @media(max-width:700px) { .block-container{padding:2rem 1rem}.pg-topline{margin-bottom:22px}.pg-hero h1{letter-spacing:-1.2px}.pg-footer{flex-direction:column;gap:4px}.pg-status{font-size:10px} }
    @media(prefers-reduced-motion:reduce) { *,*::before,*::after { animation:none!important;transition:none!important;scroll-behavior:auto!important; } }
    </style>"""


def hero(title, description, section="URL INTELLIGENCE"):
    return (
        '<section class="pg-hero"><div class="pg-eyebrow">'
        + escape(section)
        + '</div><h1>' + title + '</h1><p>' + escape(description) + '</p></section>'
    )


def panel_header(title, badge=""):
    return (
        '<div class="pg-panel-header"><h3>' + escape(title) + '</h3>'
        + ('<span class="pg-chip">' + escape(badge) + '</span>' if badge else '')
        + '</div>'
    )


def empty_state(title, description):
    return (
        '<div class="pg-empty"><div class="pg-empty-icon">⌕</div><strong>'
        + escape(title) + '</strong><p>' + escape(description) + '</p></div>'
    )


def signals(features):
    items = [
        ("HTTPS", bool(features["has_https"])),
        ("IP address host", bool(features["has_ip"])),
        ("Embedded @", bool(features["has_at_sign"])),
        ("Shortened address", bool(features["has_shortener"])),
        ("TLD risk signal", bool(features["suspicious_tld"])),
    ]
    return '<div class="pg-signals">' + ''.join(
        '<span class="pg-signal' + (' detected' if found else '') + '">'
        + ('● ' if found else '○ ') + escape(name) + '</span>'
        for name, found in items
    ) + '</div>'


def csv_bytes(frame):
    """Keep formula-like untrusted text inert in spreadsheet exports."""
    safe = frame.copy()
    for column in safe.select_dtypes(include=["object", "string"]):
        safe[column] = safe[column].map(
            lambda value: "'" + value
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@"))
            else value
        )
    return safe.to_csv(index=False).encode("utf-8-sig")
