"""
Fake News Detection — Streamlit Web Application
================================================
A professional UI for the Fake News Detection pipeline.
Delegates all ML logic to src/predict.py.
"""

import sys
import os

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st
from predict import predict_news

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Sample texts (defined early so callbacks can reference them) ──────────────
SAMPLE_FAKE = (
    "Trump supporters spread a viral video claiming massive election "
    "fraud, but the footage was from a completely different country and "
    "had nothing to do with the US election. Obama and Hillary Clinton "
    "were also mentioned in conspiracy theories shared widely on social media. "
    "The video was debunked by multiple fact-checking organizations."
)
SAMPLE_SYNTHETIC = (
    "The municipal council published its quarterly service update in "
    "Northfield, covering road repairs and drainage maintenance. "
    "Residents can review the schedule at the local information office. "
    "The regional programme outlines community improvements for the "
    "upcoming quarter, including public transport enhancements."
)

# ── Session state initialisation ──────────────────────────────────────────────
if "news_input" not in st.session_state:
    st.session_state.news_input = ""
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

# ── Callbacks (execute BEFORE the next render, so widget sees new value) ──────
def _load_fake_sample():
    st.session_state.news_input = SAMPLE_FAKE
    st.session_state.run_analysis = False

def _load_synthetic_sample():
    st.session_state.news_input = SAMPLE_SYNTHETIC
    st.session_state.run_analysis = False

def _analyze_fake_sample():
    st.session_state.news_input = SAMPLE_FAKE
    st.session_state.run_analysis = True

def _analyze_synthetic_sample():
    st.session_state.news_input = SAMPLE_SYNTHETIC
    st.session_state.run_analysis = True

def _clear_input():
    st.session_state.news_input = ""
    st.session_state.run_analysis = False

def _trigger_analysis():
    st.session_state.run_analysis = True


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.7;
        margin-top: 0;
    }

    /* Result cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
    }
    .result-fake {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 2px solid #ef4444;
    }
    .result-synthetic {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border: 2px solid #10b981;
    }
    .result-label {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .result-label-fake { color: #dc2626; }
    .result-label-synthetic { color: #059669; }
    .result-confidence {
        font-size: 1.2rem;
        font-weight: 600;
        opacity: 0.85;
    }

    /* Probability bars */
    .prob-container {
        padding: 1rem 1.5rem;
        background: #f8fafc;
        border-radius: 10px;
        margin: 0.75rem 0;
        border: 1px solid #e2e8f0;
    }
    .prob-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.3rem;
    }
    .prob-label { font-weight: 600; font-size: 0.95rem; }
    .prob-value { font-weight: 700; font-size: 0.95rem; }
    .prob-bar {
        width: 100%;
        height: 10px;
        background: #e2e8f0;
        border-radius: 5px;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    .prob-bar-fill-fake {
        height: 100%;
        border-radius: 5px;
        background: linear-gradient(90deg, #fca5a5, #ef4444);
    }
    .prob-bar-fill-synthetic {
        height: 100%;
        border-radius: 5px;
        background: linear-gradient(90deg, #6ee7b7, #10b981);
    }

    /* Disclaimer */
    .disclaimer-box {
        background: #fffbeb;
        border: 1px solid #f59e0b;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        font-size: 0.88rem;
        color: #92400e;
        line-height: 1.5;
    }

    /* Error card */
    .error-card {
        background: #fef3c7;
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .error-card p {
        font-size: 1.05rem;
        color: #92400e;
        margin: 0;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        font-size: 0.82rem;
        opacity: 0.55;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
        line-height: 1.6;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        padding-top: 1rem;
    }
    .sidebar-section {
        padding: 0.75rem 0;
        border-bottom: 1px solid rgba(128,128,128,0.2);
        margin-bottom: 0.5rem;
    }
    .sidebar-section:last-child { border-bottom: none; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/news.png", width=64)
    st.title("📰 Project Info")

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**Fake News Detection**")
    st.caption("AI-powered News Text Classification")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### 🔧 Technical Stack")
    st.markdown("""
    - **Model**: Logistic Regression
    - **Features**: TF-IDF (20,000 terms)
    - **N-grams**: Unigrams + Bigrams
    - **Framework**: Scikit-learn
    - **UI**: Streamlit
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### 📊 Model Performance")
    st.markdown("""
    | Metric | Score |
    |--------|-------|
    | Accuracy | 99.96% |
    | F1-Score | 99.96% |
    | ROC-AUC | 0.9998 |
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### 🏷️ Label Definitions")
    st.markdown("""
    - **Label 0 — Fake**: Scraped fake / misleading news articles
    - **Label 1 — Synthetic**: AI-generated synthetic articles used as the counter-class
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### ⚠️ Important Notice")
    st.info(
        "This model distinguishes **writing-style differences** between two "
        "datasets. It does **not** verify factual truth. Results are for "
        "educational and research purposes only.",
        icon="⚠️",
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 Fake News Detection</h1>
    <p>AI-powered News Text Classification</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Text input area ──────────────────────────────────────────────────────────
st.markdown("#### 📝 Enter or paste a news article below:")

# The widget uses "news_input" as its key directly.
# Callbacks set st.session_state.news_input BEFORE this widget renders,
# so the text area always shows the correct value.
news_text = st.text_area(
    label="News article text",
    height=200,
    placeholder=(
        "Paste a news article here to analyze whether it matches the writing "
        "style of fake news or synthetic (AI-generated) news articles..."
    ),
    label_visibility="collapsed",
    key="news_input",           # widget IS the session-state entry
)

# ── Action buttons ────────────────────────────────────────────────────────────
col_analyze, col_clear, col_spacer = st.columns([1.5, 1, 3])

with col_analyze:
    st.button(
        "🔎  Analyze News",
        type="primary",
        use_container_width=True,
        on_click=_trigger_analysis,
    )

with col_clear:
    st.button(
        "🗑️  Clear",
        use_container_width=True,
        on_click=_clear_input,
    )

# ── Run analysis (triggered by callback flag) ─────────────────────────────────
def _display_result(result: dict):
    """Render a prediction result (success or error)."""
    if result["prediction"] == -1:
        st.markdown(f"""
        <div class="error-card">
            <p>⚠️ {result['warning']}</p>
        </div>
        """, unsafe_allow_html=True)
        return

    is_fake = result["prediction"] == 0
    card_class = "result-fake" if is_fake else "result-synthetic"
    label_class = "result-label-fake" if is_fake else "result-label-synthetic"
    icon = "🚨" if is_fake else "✅"
    label_text = "Fake News Detected" if is_fake else "Synthetic / Non-Fake"
    confidence_pct = result["confidence"] * 100

    st.markdown(f"""
    <div class="result-card {card_class}">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">{icon}</div>
        <div class="result-label {label_class}">{label_text}</div>
        <div class="result-confidence">Confidence: {confidence_pct:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    fake_pct = result["fake_probability"] * 100
    synth_pct = result["synthetic_probability"] * 100

    st.markdown(f"""
    <div class="prob-container">
        <div class="prob-row">
            <span class="prob-label">🚨 Fake Probability</span>
            <span class="prob-value" style="color: #dc2626;">{fake_pct:.2f}%</span>
        </div>
        <div class="prob-bar">
            <div class="prob-bar-fill-fake" style="width: {fake_pct}%;"></div>
        </div>
        <div class="prob-row">
            <span class="prob-label">✅ Synthetic Probability</span>
            <span class="prob-value" style="color: #059669;">{synth_pct:.2f}%</span>
        </div>
        <div class="prob-bar">
            <div class="prob-bar-fill-synthetic" style="width: {synth_pct}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 Detailed Prediction Metrics"):
        met_col1, met_col2, met_col3 = st.columns(3)
        met_col1.metric("Prediction", result["label"])
        met_col2.metric("Confidence", f"{confidence_pct:.2f}%")
        met_col3.metric("Label Index", result["prediction"])

        st.markdown("---")
        st.markdown("**Raw probabilities:**")
        st.json({
            "fake_probability": result["fake_probability"],
            "synthetic_probability": result["synthetic_probability"],
            "confidence": result["confidence"],
            "prediction": result["prediction"],
            "label": result["label"],
        })

    st.markdown(f"""
    <div class="disclaimer-box">
        ⚠️ <strong>Disclaimer:</strong> {result['warning']}
    </div>
    """, unsafe_allow_html=True)


if st.session_state.run_analysis:
    st.session_state.run_analysis = False   # consume the flag
    input_text = news_text.strip() if news_text else ""

    if not input_text:
        st.markdown("""
        <div class="error-card">
            <p>⚠️ Please enter a news article to analyze.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("Analyzing article..."):
            result = predict_news(input_text)
        _display_result(result)


# ── Sample texts section ──────────────────────────────────────────────────────
st.markdown("---")

with st.expander("📋 Try Sample Texts", expanded=True):
    st.markdown("**Load a sample into the text area, or analyze it directly:**")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("##### 🚨 Fake-Style Sample")
        st.caption(SAMPLE_FAKE[:120] + "…")
        btn_col_a, btn_col_b = st.columns(2)
        with btn_col_a:
            st.button(
                "📋 Load Text",
                key="load_fake",
                use_container_width=True,
                on_click=_load_fake_sample,
            )
        with btn_col_b:
            st.button(
                "🔎 Analyze",
                key="analyze_fake",
                use_container_width=True,
                on_click=_analyze_fake_sample,
            )

    with col_s2:
        st.markdown("##### ✅ Synthetic-Style Sample")
        st.caption(SAMPLE_SYNTHETIC[:120] + "…")
        btn_col_c, btn_col_d = st.columns(2)
        with btn_col_c:
            st.button(
                "📋 Load Text",
                key="load_synthetic",
                use_container_width=True,
                on_click=_load_synthetic_sample,
            )
        with btn_col_d:
            st.button(
                "🔎 Analyze",
                key="analyze_synthetic",
                use_container_width=True,
                on_click=_analyze_synthetic_sample,
            )

    st.caption(
        "These samples illustrate the two writing styles in the training data. "
        "The model detects stylistic patterns, not factual accuracy."
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <strong>⚠️ Experimental Research Tool</strong><br>
    This application is a demonstration prototype built for educational purposes.<br>
    The model distinguishes writing-style differences between a fake-news corpus
    and AI-generated synthetic articles.<br>
    It does <strong>not</strong> independently verify factual truth and should
    <strong>not</strong> be used as a production fact-checker.<br><br>
    Built with Scikit-learn · Streamlit · Python<br>
    Fake News Detection Project © 2026
</div>
""", unsafe_allow_html=True)
