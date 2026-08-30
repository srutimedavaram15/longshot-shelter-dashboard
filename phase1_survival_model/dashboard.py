import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt
from datetime import date

AGE_BINS = [-0.01, 0.5, 2, 7, 30]
AGE_LABELS = ["Juvenile (0-6mo)", "Young (6mo-2yr)", "Adult (2-7yr)", "Senior (7yr+)"]
AGE_ORDER = AGE_LABELS + ["Unknown"]

FEATURE_READABLE = {
    "Intake Type": "intake type",
    "Condition_grouped": "intake condition",
    "Animal Type": "animal type",
    "Sex upon Intake": "sex",
    "age_group": "age group",
}

TIER_COLORS = {
    "High (70-100)": "#6B6A90",   # deep lavender — clearly visible, signals urgency
    "Medium (30-70)": "#9C9BB5",  # medium lavender
    "Low (0-30)":    "#C1C0D5",   # lightest lavender — still visible on #F2E8F1
}

WARM_PALETTE = ["#8E8DA8", "#6B6A90", "#A8A7C0", "#5A597A", "#9C9BB5", "#7A799A", "#B4B3CC"]

REQUIRED_COLUMNS = [
    "Animal Type",
    "Age upon Intake",
    "Intake Type",
    "Intake Condition",
    "Sex upon Intake",
    "intake_date",
]

# ── CSS ────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bitter:wght@400;600;700&family=Fraunces:wght@400;700&display=swap');

/* === PAGE === */
.stApp { background-color: #DAC7DB !important; }

/* Streamlit JS hardcodes paddingTop:"6rem" on .stMainBlockContainer via Emotion.
   .main .block-container never matched (parent is .stMain, not .main).
   These selectors hit the actual element. !important beats the non-!important JS value. */
[data-testid="stMainBlockContainer"],
.stMainBlockContainer,
.block-container {
    background-color: #DAC7DB !important;
    padding-top: 1.2rem !important;
    padding-bottom: 3rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    animation: fadeIn 0.4s ease-out both;
}
/* Collapse the invisible wrapper that st.markdown("<style>…") creates.
   :has(style) matches ONLY the stMarkdown containing the injected <style> tag —
   no other stMarkdown on the page has a <style> child, so there is no
   font-size/line-height bleed onto expander lists or any other content. */
.stMarkdown:has(style) {
    height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(7px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* === TYPOGRAPHY === */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Bitter', Georgia, serif !important;
    color: #000000 !important;
    letter-spacing: -0.01em;
}
p, li {
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
    color: #000000;
}
.stMarkdown {
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stCaption"] p, .stCaption p {
    color: #000000 !important;
    font-size: 0.83rem !important;
}

/* === WIDGET LABELS (file uploader, selectbox, all st.X() labels)
   Emotion target: e1kt9bl70 — data-testid="stWidgetLabel"
   Text lives inside: <label> > <span aria-hidden> > <div.stMarkdown> > <p>
   Use parent-context selectors (specificity 0-2-0/0-2-1) to beat Emotion class rules. */
[data-testid="stWidgetLabel"],
label[data-testid="stWidgetLabel"] {
    font-family: 'Bitter', Georgia, serif !important;
    color: #000000 !important;
}
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"],
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"] span,
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"] p,
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"] div,
[data-testid="stSelectbox"] [data-testid="stWidgetLabel"],
[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] span {
    font-family: 'Bitter', Georgia, serif !important;
    color: #000000 !important;
}
/* Selectbox filter labels: all typographic properties set as OWN rules on <p>/<div>
   (not inherited from the parent <label>) so Emotion's own <p> styles cannot block them.
   Specificity 0-3-1 + !important beats any Emotion compound selector. */
[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] p,
[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] div {
    font-family: 'Bitter', Georgia, serif !important;
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    color: #000000 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
.logo-tagline {
    font-family: 'Bitter', Georgia, serif;
    font-size: 1.2rem;
    color: #000000;
    margin: 0 0 1.6rem 0;
    line-height: 1.5;
}
.logo-heading {
    font-family: 'Fraunces', Georgia, serif !important;
    font-size: 4.2rem !important;
    line-height: 1.05 !important;
    margin: 0 0 0.5rem 0 !important;
    padding: 0 !important;
    color: #000000 !important;
    letter-spacing: -0.03em;
}
.logo-heading .bold   { font-weight: 700 !important; }
.logo-heading .regular { font-weight: 400 !important; }

/* === SECTION DIVIDER === */
.section-divider {
    height: 2px;
    background: linear-gradient(
        to right,
        transparent, #C1C0D5 25%, #F2E8F1 60%, #C1C0D5 80%, transparent
    );
    border-radius: 2px;
    margin: 1.8rem 0 1.3rem 0;
    opacity: 0.65;
}

/* === FILE UPLOADER === */
/* Label above the uploader and the "Limit Xmb • CSV" helper text */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label p,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] small,
[data-testid="stFileUploadDropzone"] span {
    color: #000000 !important;
}
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploadDropzone"] {
    background: #F2E8F1 !important;
    border: 1.5px solid #000000 !important;
    border-radius: 999px !important;
    transition: border-color 0.18s ease, background 0.18s ease;
}
[data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #C1C0D5 !important;
    background: #F2E8F1 !important;
}

/* === ALERTS === */
[data-testid="stAlert"] {
    background-color: #F2E8F1 !important;
    border-radius: 14px !important;
    border: none !important;
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] div {
    color: #000000 !important;
}

/* === TABS — iOS-style segmented control === */

/* Kill default underline indicators — attribute-only (0-1-0) loses to Emotion;
   element+attribute (0-1-1) wins. Cover both plus visibility as belt-and-suspenders. */
[data-baseweb="tab-border"],
[data-baseweb="tab-highlight"],
div[data-baseweb="tab-border"],
div[data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"],
.stTabs [data-baseweb="tab-highlight"],
.stTabs div[data-baseweb="tab-border"],
.stTabs div[data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"],
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] div[data-baseweb="tab-border"],
[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    width: 0 !important;
    opacity: 0 !important;
    border: none !important;
    background: transparent !important;
    position: absolute !important;
    pointer-events: none !important;
}
/* Kill any bottom border or pseudo-element underline on the tab list and individual tabs */
[role="tablist"],
.stTabs [data-baseweb="tab-list"],
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    border-bottom: none !important;
    box-shadow: none !important;
}
.stTabs [data-baseweb="tab"],
[data-testid="stTabs"] [data-baseweb="tab"],
[role="tab"] {
    border-bottom: none !important;
    text-decoration: none !important;
    outline: none !important;
}
.stTabs [data-baseweb="tab"]::before,
.stTabs [data-baseweb="tab"]::after,
[data-testid="stTabs"] [data-baseweb="tab"]::before,
[data-testid="stTabs"] [data-baseweb="tab"]::after,
[role="tab"]::before,
[role="tab"]::after {
    display: none !important;
    border: none !important;
    background: transparent !important;
    height: 0 !important;
}

/* Pill track */
.stTabs [data-baseweb="tab-list"],
[data-testid="stTabs"] [data-baseweb="tab-list"],
[role="tablist"] {
    background-color: #F2E8F1 !important;
    border-radius: 999px !important;
    border: 1.5px solid #000000 !important;
    padding: 4px !important;
    gap: 2px !important;
    margin-bottom: 0.8rem !important;
    display: flex !important;
    align-items: center !important;
    width: fit-content !important;
    overflow: hidden !important;
}

/* Individual tab segments */
.stTabs [data-baseweb="tab"],
[data-testid="stTabs"] [data-baseweb="tab"],
[role="tab"] {
    background-color: transparent !important;
    border-radius: 999px !important;
    color: #000000 !important;
    font-family: 'Bitter', Georgia, serif !important;
    font-size: 0.96rem !important;
    font-weight: 600 !important;
    padding: 1rem 2rem !important;
    border: none !important;
    margin: 0 !important;
    letter-spacing: 0.01em;
    transition: background 0.15s ease, color 0.15s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    white-space: nowrap !important;
}

/* Hover */
.stTabs [data-baseweb="tab"]:hover,
[data-testid="stTabs"] [data-baseweb="tab"]:hover,
[role="tab"]:hover {
    background-color: rgba(193, 192, 213, 0.30) !important;
    color: #000000 !important;
    border-radius: 999px !important;
}

/* Active pill */
.stTabs [aria-selected="true"],
[data-testid="stTabs"] [aria-selected="true"],
[role="tab"][aria-selected="true"],
.stTabs div[aria-selected="true"],
[data-testid="stTabs"] div[aria-selected="true"] {
    background-color: #C1C0D5 !important;
    color: #000000 !important;
    border-radius: 999px !important;
    box-shadow: none !important;
    filter: none !important;
}

/* Tab content panel */
.stTabs [data-baseweb="tab-panel"] {
    background: #F2E8F1 !important;
    border-radius: 16px !important;
    border: 1.5px solid #000000 !important;
    padding: 1.5rem 1.5rem 2rem 1.5rem !important;
    margin-top: 0 !important;
}

/* === TABLE / DATAFRAME === */
[data-testid="stDataFrame"] {
    background: #F2E8F1 !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 18px rgba(0, 0, 0, 0.08) !important;
    overflow: hidden !important;
    border: 1.5px solid #000000 !important;
}
[data-testid="stDataFrame"] > div {
    border-radius: 16px !important;
}

/* === ALTAIR / VEGA-LITE CHART CONTAINERS === */
[data-testid="stVegaLiteChart"] {
    border-radius: 15px !important;
    overflow: hidden !important;
}
[data-testid="stVegaLiteChart"] canvas,
[data-testid="stVegaLiteChart"] svg {
    border-radius: 15px !important;
    overflow: hidden !important;
}

/* === SELECTBOX === */
[data-testid="stSelectbox"] label {
    font-family: 'Bitter', Georgia, serif !important;
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    color: #000000 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child {
    border: 1.5px solid #000000 !important;
    border-radius: 999px !important;
    background-color: #F2E8F1 !important;
    box-shadow: none !important;
    transition: border-color 0.15s ease !important;
    min-height: 2.5rem !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child:hover {
    border-color: #C1C0D5 !important;
    box-shadow: 0 0 0 2px rgba(193, 192, 213, 0.30) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] span {
    color: #000000 !important;
}
[data-testid="stSelectbox"] [data-baseweb="select-arrow"] {
    color: #000000 !important;
}

/* === SECTION HEADER ACCENTS === */
[data-testid="stHeadingWithActionElements"] h2,
[data-testid="stHeadingWithActionElements"] h3,
[data-testid="stMarkdown"] h2,
[data-testid="stMarkdown"] h3 {
    border-left: 3px solid #000000;
    padding-left: 0.65rem;
}

/* === BUTTONS === */
.stButton > button {
    background-color: #C1C0D5 !important;
    color: #000000 !important;
    border: 1.5px solid #000000 !important;
    border-radius: 14px !important;
    font-family: 'Bitter', Georgia, serif !important;
    font-weight: 600 !important;
    transition: background 0.18s ease !important;
}
.stButton > button:hover {
    background-color: #F2E8F1 !important;
}

/* === CHART SECTION LABELS ===
   .stMarkdown .chart-label gives specificity 0-2-0, beating stMarkdown's Emotion
   compound p-selectors (0-1-1) that would override our styles on a <p> element.
   Use <div class="chart-label"> in markup — Emotion's paragraph rules don't touch divs. */
.stMarkdown .chart-label,
.chart-label {
    font-family: 'Bitter', Georgia, serif !important;
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    color: #000000 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    margin-bottom: 1.2rem !important;
    margin-top: 1rem !important;
    line-height: 1 !important;
    display: block !important;
}

/* === UPLOAD LABEL ROW (label + badge on one line) === */
.upload-label-row {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-bottom: 0.35rem;
}
.upload-label-text {
    font-family: 'Bitter', Georgia, serif;
    font-size: 0.875rem;
    font-weight: 400;
    color: #000000;
    line-height: 1.4;
}

/* === DATA SOURCE BADGE === */
.data-badge-wrap {
    margin-top: 0;
    margin-bottom: 0.35rem;
}
.data-badge {
    display: inline-block;
    font-family: 'Bitter', Georgia, serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    border-radius: 999px !important;
    padding: 0.22rem 0.75rem !important;
    letter-spacing: 0.01em;
    line-height: 1.5 !important;
    white-space: nowrap;
}
.data-badge--demo {
    background: rgba(193, 192, 213, 0.22) !important;
    border: 1px solid #C1C0D5 !important;
    color: #3d3c52 !important;
}
.data-badge--live {
    background: #C1C0D5 !important;
    border: 1px solid #9C9BB5 !important;
    color: #000000 !important;
}

/* === LOGO IMAGE === */
[data-testid="stImage"] {
    background: transparent !important;
    margin-bottom: 0.1rem;
}
[data-testid="stImage"] img {
    background: transparent !important;
    display: block !important;
}

/* === SCROLLBAR === */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #DAC7DB; }
::-webkit-scrollbar-thumb { background: #C1C0D5; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #000000; }

/* ═══ DARK-DEFAULT AUDIT OVERRIDES ═══ */

/* 1. DROPDOWN OPEN STATE */
[data-baseweb="popover"] [data-baseweb="block"],
[data-baseweb="popover"] > div > div {
    background-color: #F2E8F1 !important;
    border: 1.5px solid #C1C0D5 !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.12) !important;
    overflow: hidden !important;
}
ul[data-baseweb="menu"] {
    background-color: #F2E8F1 !important;
    padding: 4px 4px !important;
}

/* 2. DROPDOWN OPTIONS */
[role="option"],
li[role="option"],
[data-baseweb="menu"] li {
    background-color: transparent !important;
    color: #000000 !important;
    border-radius: 8px !important;
}
[role="option"]:hover,
li[role="option"]:hover,
[data-baseweb="menu"] li:hover,
[data-baseweb="menu-item"]:hover {
    background-color: rgba(193, 192, 213, 0.30) !important;
    color: #000000 !important;
}
[role="option"][aria-selected="true"],
[data-baseweb="menu"] [aria-selected="true"] {
    background-color: rgba(193, 192, 213, 0.50) !important;
    color: #000000 !important;
    font-weight: 600;
}
[role="option"] span,
[role="option"] div,
li[role="option"] * {
    color: #000000 !important;
}

/* 3. SELECTBOX FOCUS */
[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child:focus-within {
    border-color: #C1C0D5 !important;
    box-shadow: 0 0 0 2px rgba(193, 192, 213, 0.35) !important;
    outline: none !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] div[data-testid="stSelectbox"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] > div > div span {
    color: #000000 !important;
}

/* 4. GLOBAL FOCUS RINGS */
*:focus-visible {
    outline: 2px solid rgba(193, 192, 213, 0.60) !important;
    outline-offset: 2px;
}
button:focus-visible,
[role="tab"]:focus-visible,
[data-baseweb="tab"]:focus-visible {
    outline: 2px solid rgba(193, 192, 213, 0.70) !important;
    box-shadow: none !important;
}

/* 5. FILE UPLOADER BROWSE BUTTON
   JS confirms data-testid="stFileUploaderDropzone" (with 'r') is the actual rendered testid.
   font-family on the button element is safe — the icon span inside has its own
   explicit font-family:'Material Symbols Outlined' which wins over inherited value. */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"],
[data-testid="stFileUploadDropzone"] button,
[data-testid="stFileUploadDropzone"] [data-testid="stBaseButton-secondary"] {
    background-color: transparent !important;
    border: 1.5px solid #000000 !important;
    color: #000000 !important;
    border-radius: 12px !important;
    font-family: 'Bitter', Georgia, serif !important;
    font-weight: 600 !important;
    transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease !important;
}
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploadDropzone"] button:hover {
    background-color: rgba(193, 192, 213, 0.20) !important;
    border-color: #C1C0D5 !important;
    color: #000000 !important;
}

/* 6. STREAMLIT HEADER / TOP TOOLBAR
   Collapsed to zero height — removes the ~4rem section padding-top that
   Streamlit injects to clear the fixed header, giving us full control
   over page-top spacing via block-container padding alone. */
header[data-testid="stHeader"],
[data-testid="stHeader"] {
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    border-bottom: none !important;
    box-shadow: none !important;
}
section[data-testid="stMain"] {
    padding-top: 0 !important;
}

/* 7. BASEWEB INPUT / TEXTAREA */
[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div {
    background-color: #F2E8F1 !important;
    border-color: #C1C0D5 !important;
    color: #000000 !important;
}
[data-baseweb="input"]:focus-within > div,
[data-baseweb="textarea"]:focus-within > div {
    border-color: #000000 !important;
    box-shadow: 0 0 0 2px rgba(193, 192, 213, 0.30) !important;
}

/* 8. TOOLTIP */
[data-baseweb="tooltip"] [role="tooltip"],
[data-baseweb="tooltip"] > div {
    background-color: #000000 !important;
    color: #F2E8F1 !important;
    border-radius: 8px !important;
    border: none !important;
}

/* 9. ALERT / TOAST BORDERS */
[data-testid="stAlert"] > div {
    border-radius: 14px !important;
    border-left: 3px solid #C1C0D5 !important;
    background-color: #F2E8F1 !important;
}

/* 10. SIDEBAR */
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background-color: #F2E8F1 !important;
    border-right: 1.5px solid #C1C0D5 !important;
}

/* 11. BASEWEB TAG */
[data-baseweb="tag"] {
    background-color: rgba(193, 192, 213, 0.25) !important;
    color: #000000 !important;
    border-color: #C1C0D5 !important;
    border-radius: 8px !important;
}
[data-baseweb="tag"] span {
    color: #000000 !important;
}

/* 12. SPINNER */
[data-testid="stSpinner"] > div {
    border-top-color: #C1C0D5 !important;
    border-right-color: rgba(193, 192, 213, 0.30) !important;
    border-bottom-color: rgba(193, 192, 213, 0.30) !important;
    border-left-color: rgba(193, 192, 213, 0.30) !important;
}

/* 13. PROGRESS BAR */
[data-testid="stProgress"] > div {
    background-color: rgba(193, 192, 213, 0.30) !important;
    border-radius: 99px !important;
}
[data-testid="stProgress"] > div > div {
    background-color: #C1C0D5 !important;
    border-radius: 99px !important;
}

/* 14. EXPANDER */
[data-testid="stExpander"] {
    border: 1.5px solid #C1C0D5 !important;
    border-radius: 14px !important;
    background-color: #F2E8F1 !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] details summary {
    color: #000000 !important;
    font-family: 'Bitter', Georgia, serif !important;
    font-weight: 600 !important;
}

/* 15. METRIC CONTAINERS */
[data-testid="metric-container"] {
    background-color: #F2E8F1 !important;
    border: 1.5px solid #C1C0D5 !important;
    border-radius: 14px !important;
    padding: 0.8rem 1rem !important;
}
[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #000000 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #000000 !important;
}
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    with open("shelter_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["feature_columns"], bundle["categorical_features"]


def parse_age_years(age_str):
    if pd.isna(age_str):
        return np.nan
    parts = str(age_str).strip().lower().split()
    if len(parts) < 2:
        return np.nan
    try:
        val = float(parts[0])
    except ValueError:
        return np.nan
    unit = parts[1]
    if "year" in unit:
        return val
    elif "month" in unit:
        return val / 12
    elif "week" in unit:
        return val / 52
    elif "day" in unit:
        return val / 365
    return np.nan


def compute_majority_cols(df_encoded, categorical_features):
    """Return the one-hot column name of the most common category per feature group."""
    majority = set()
    for cat_feature in categorical_features:
        prefix = cat_feature + "_"
        group_cols = [c for c in df_encoded.columns if c.startswith(prefix)]
        if group_cols:
            majority.add(df_encoded[group_cols].sum().idxmax())
    return majority


def generate_reason(row_encoded, hazard_ratios, majority_cols):
    active = []
    for col in hazard_ratios.index:
        if col in row_encoded.index and row_encoded[col] == 1:
            is_majority = col in majority_cols
            log_hr = abs(np.log(hazard_ratios[col]))
            active.append((col, is_majority, log_hr))
    if not active:
        return "Profile matches baseline risk level."
    # Non-majority features first; within each tier, sort by |log(HR)| descending.
    active.sort(key=lambda x: (x[1], -x[2]))
    parts = []
    for col, _, _ in active[:2]:
        for cat_feature, readable in FEATURE_READABLE.items():
            if col.startswith(cat_feature + "_"):
                val = col[len(cat_feature) + 1:]
                parts.append(f"{readable} ({val})")
                break
    if not parts:
        return "Profile matches baseline risk level."
    if len(parts) == 1:
        return f"Flagged primarily due to {parts[0]}."
    return f"Flagged primarily due to {parts[0]} and {parts[1]}."


@st.cache_data
def prepare_and_score(_model, feature_columns, categorical_features, data_bytes,
                      age_labels=tuple(AGE_LABELS), is_upload=False):
    df_raw = pd.read_csv(pd.io.common.BytesIO(data_bytes)) if isinstance(data_bytes, bytes) else data_bytes

    if is_upload:
        # All rows are current shelter animals — no outcome filter needed
        df = df_raw.copy().reset_index(drop=True)
    else:
        # Demo/training data: filter to animals still in shelter at snapshot time
        df = df_raw[df_raw["is_censored"] == True].copy().reset_index(drop=True)

    df["age_years"] = df["Age upon Intake"].apply(parse_age_years).clip(upper=30)
    df["age_group"] = (
        pd.cut(df["age_years"], bins=AGE_BINS, labels=list(age_labels))
        .astype(str)
        .replace("nan", "Unknown")
    )

    condition_counts = df_raw["Intake Condition"].value_counts()
    kept_conditions = set(condition_counts[condition_counts >= 200].index)
    df["Condition_grouped"] = df["Intake Condition"].apply(
        lambda x: x if x in kept_conditions else "Other"
    )

    df_cat = df[categorical_features].astype(str)
    df_encoded = pd.get_dummies(df_cat, columns=categorical_features)

    covariate_cols = [c for c in feature_columns if c not in ("length_of_stay", "event_observed")]
    df_encoded = df_encoded.reindex(columns=covariate_cols, fill_value=0)

    hazard = _model.predict_partial_hazard(df_encoded)
    neg_hazard = -hazard
    mn, mx = neg_hazard.min(), neg_hazard.max()
    if mx > mn:
        scores = ((neg_hazard - mn) / (mx - mn) * 100).round(1)
    else:
        scores = pd.Series([50.0] * len(neg_hazard), index=neg_hazard.index)
    df["Risk Score"] = scores.values

    today = pd.Timestamp(date.today())
    intake_parsed = pd.to_datetime(df["intake_date"], utc=True, errors="coerce").dt.tz_convert(None)
    df["Days in Shelter"] = (today - intake_parsed).dt.days.clip(lower=0)

    name_col = next((c for c in ("Name", "Animal ID") if c in df.columns), None)
    df["Animal Name"] = df[name_col].fillna("Unnamed") if name_col else "Unnamed"

    hr = _model.hazard_ratios_
    majority_cols = compute_majority_cols(df_encoded, categorical_features)
    df["Reason"] = [generate_reason(df_encoded.iloc[i], hr, majority_cols) for i in range(len(df_encoded))]

    df["Risk Tier"] = pd.cut(
        df["Risk Score"],
        bins=[-0.01, 30, 70, 100.01],
        labels=["Low (0-30)", "Medium (30-70)", "High (70-100)"],
    )

    return df


def warm_bar(df, x_col, y_col, *, color="#8E8DA8", color_map=None, sort_x=None):
    x_axis = alt.Axis(
        labelColor="#000000", labelAngle=-18,
        labelFont="Inter, sans-serif",
        tickColor="#A8A7C0", domainColor="#A8A7C0",
    )
    y_axis = alt.Axis(
        labelColor="#000000", labelFont="Inter, sans-serif",
        gridColor="#C1C0D5", domainOpacity=0,
        titleColor="#000000", titleFont="Inter, sans-serif",
    )

    y_max = df[y_col].max() * 1.15

    x_enc = alt.X(f"{x_col}:N", title=None, sort=sort_x, axis=x_axis)
    y_enc = alt.Y(f"{y_col}:Q", title=y_col, axis=y_axis,
                  scale=alt.Scale(domainMax=y_max))

    if color_map:
        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, opacity=0.9)
            .encode(
                x=x_enc, y=y_enc,
                color=alt.Color(
                    f"{x_col}:N",
                    scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
                    legend=None,
                ),
            )
        )
    else:
        chart = (
            alt.Chart(df)
            .mark_bar(color=color, cornerRadiusTopLeft=5, cornerRadiusTopRight=5, opacity=0.9)
            .encode(x=x_enc, y=y_enc)
        )

    return (
        chart
        .properties(
            height=260,
            background="#F2E8F1",
            padding={"top": 20, "right": 16, "bottom": 16, "left": 16},
        )
        .configure_view(strokeWidth=0, fill="#F2E8F1", stroke=None, cornerRadius=15)
    )


# ── App ────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Longshot", layout="wide", page_icon="🐾")
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

    model, feature_columns, categorical_features = load_model()

    # Logo heading + tagline
    st.markdown(
        '<h1 class="logo-heading">'
        '<span class="bold">Long</span><span class="regular">shot</span>'
        '</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="logo-tagline">Giving every animal their shot at a home ♥</div>',
        unsafe_allow_html=True,
    )

    # Data source — label and badge rendered together in one row
    _label_row_slot = st.empty()
    uploaded = st.file_uploader(
        "Upload your shelter's intake CSV", type="csv", label_visibility="collapsed"
    )
    with st.expander("Format requirements"):
        st.markdown(
            "Your CSV needs: **Animal Type**, **Age upon Intake**, **Intake Type**, "
            "**Intake Condition**, **Sex upon Intake**, **Intake Date** "
            "(YYYY-MM-DD or MM/DD/YYYY format), and **Animal ID** or **Name**. "
            "Days in shelter is calculated automatically from the intake date."
        )

    if uploaded:
        _badge_text = "Showing your shelter's data"
        _badge_cls = "data-badge--live"
    else:
        _badge_text = "Showing sample shelter data — upload yours to get started"
        _badge_cls = "data-badge--demo"
    _label_row_slot.markdown(
        '<div class="upload-label-row">'
        '<span class="upload-label-text">Upload your shelter&#39;s intake CSV</span>'
        f'<span class="data-badge {_badge_cls}">{_badge_text}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    if uploaded:
        data_bytes = uploaded.read()
        headers = pd.read_csv(pd.io.common.BytesIO(data_bytes), nrows=0).columns.tolist()
        has_name = "Name" in headers or "Animal ID" in headers
        missing = [c for c in REQUIRED_COLUMNS if c not in headers]
        if not has_name:
            missing.append("Name (or Animal ID)")
        if missing:
            st.error(
                f"Your CSV is missing {len(missing)} required column(s): "
                f"**{', '.join(missing)}**\n\n"
                "Please check the Format requirements above and re-export your data."
            )
            st.stop()
        df = prepare_and_score(model, feature_columns, categorical_features,
                               data_bytes, is_upload=True)
    else:
        df_raw = pd.read_csv("data/cleaned_length_of_stay.csv")
        df = prepare_and_score(model, feature_columns, categorical_features, df_raw)

    if df.empty:
        st.warning("No animals found in the uploaded data.")
        return

    tab1, tab2 = st.tabs(["Priority List", "Shelter Analytics"])

    # ── Tab 1: Priority List ───────────────────────────────────────────────────
    with tab1:
        st.subheader("Today's Priority List")

        c1, c2 = st.columns(2)
        with c1:
            types = ["All"] + sorted(df["Animal Type"].dropna().unique().tolist())
            sel_type = st.selectbox("Animal Type", types)
        with c2:
            present_ages = [a for a in AGE_ORDER if a in df["age_group"].values]
            sel_age = st.selectbox("Age Group", ["All"] + present_ages)

        filtered = df.copy()
        if sel_type != "All":
            filtered = filtered[filtered["Animal Type"] == sel_type]
        if sel_age != "All":
            filtered = filtered[filtered["age_group"] == sel_age]

        display = (
            filtered[["Animal Name", "Animal Type", "age_group",
                       "Days in Shelter", "Risk Score", "Reason"]]
            .sort_values("Risk Score", ascending=False)
            .reset_index(drop=True)
        )
        display.columns = ["Animal Name", "Animal Type", "Age Group",
                           "Days in Shelter", "Risk Score", "Reason"]

        st.dataframe(display, use_container_width=True)
        st.caption(f"Showing {len(display):,} of {len(df):,} animals currently in the shelter.")

    # ── Tab 2: Shelter Overview ────────────────────────────────────────────────
    with tab2:
        st.subheader("Shelter Analytics")

        c1, c2 = st.columns(2)

        with c1:
            tier_order = ["High (70-100)", "Medium (30-70)", "Low (0-30)"]
            tier_counts = (
                df["Risk Tier"].value_counts()
                .reindex(tier_order).fillna(0).astype(int)
            )
            tier_df = tier_counts.reset_index()
            tier_df.columns = ["Risk Tier", "Count"]
            st.markdown('<div class="chart-label">Animal Count by Risk Tier</div>',
                        unsafe_allow_html=True)
            st.altair_chart(
                warm_bar(tier_df, "Risk Tier", "Count",
                         color_map=TIER_COLORS, sort_x=tier_order),
                use_container_width=True,
            )

        with c2:
            type_counts = df["Animal Type"].value_counts()
            type_df = type_counts.reset_index()
            type_df.columns = ["Animal Type", "Count"]
            type_sort = type_df["Animal Type"].tolist()
            type_color_map = {t: WARM_PALETTE[i % len(WARM_PALETTE)]
                              for i, t in enumerate(type_sort)}
            st.markdown('<div class="chart-label">Animal Count by Animal Type</div>',
                        unsafe_allow_html=True)
            st.altair_chart(
                warm_bar(type_df, "Animal Type", "Count",
                         color_map=type_color_map, sort_x=type_sort),
                use_container_width=True,
            )

        age_avg = df.groupby("age_group")["Risk Score"].mean()
        present = [a for a in AGE_ORDER if a in age_avg.index]
        age_avg = age_avg.reindex(present)
        age_df = age_avg.reset_index()
        age_df.columns = ["Age Group", "Avg Risk Score"]
        st.markdown('<div class="chart-label">Average Risk Score by Age Group</div>',
                    unsafe_allow_html=True)
        st.altair_chart(
            warm_bar(age_df, "Age Group", "Avg Risk Score",
                     color="#8E8DA8", sort_x=present),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
