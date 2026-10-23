"""
frontend/app.py
---------------
Phase 3: Interactive Streamlit Web Platform — LawAI Legal Tech Predictor

4-tab legal dashboard:
  Tab 1 ⚖️  Case Triage & Prediction Engine     — FIR NLP auto-fill + feature controls
  Tab 2 🔍  Explainable AI & Precedent Matcher  — SHAP, LIME, K-NN precedent cards
  Tab 3 🎛️  What-If Sensitivity Simulator       — Baseline vs modified side-by-side
  Tab 4 📊  Model Evaluation Benchmark          — ROC, PR, Confusion Matrix, Log Loss

Run with:
    streamlit run frontend/app.py
(FastAPI backend must be running on http://localhost:8000)
"""

from datetime import datetime
import json
import os
import sys
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# ── Path Setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pdf_generator import generate_case_report

MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH  = PROJECT_ROOT / "data" / "criminal_theft_dataset_v2_1500.csv"
EVAL_JSON  = MODELS_DIR / "evaluation_report.json"

# ── Design Constants ──────────────────────────────────────────────────────────
SEVERITY_LABELS  = {0: "Acquittal / Minimal", 1: "Light Sentence",
                    2: "Moderate Sentence",   3: "Severe Sentence"}
SEVERITY_COLORS  = {0: "#2ECC71", 1: "#F39C12", 2: "#E67E22", 3: "#E74C3C"}
SEVERITY_EMOJIS  = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}
FEATURE_NAMES    = ["stolen_value_inr", "counsel_type_enc", "weapon_present",
                    "prior_convictions", "property_recovered_pct",
                    "confession_recorded", "forced_entry"]
CONFIDENCE_COLORS = {"regex": "🟢", "regex_fallback": "🟢", "keyword": "🟡",
                     "spacy_ner": "🟡", "default": "🟠", "not_found": "🔴"}

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LawAI — Legal Tech Predictor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .verdict-box {
        border-radius: 10px; padding: 18px; text-align: center;
        color: white; font-size: 1.1rem; font-weight: bold;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .precedent-card {
        border: 1px solid #e0e0e0; border-radius: 8px;
        padding: 14px; background: #fafafa;
    }
    .lime-card {
        border-left: 4px solid #3498DB; border-radius: 4px;
        padding: 10px 14px; margin-bottom: 8px;
        background: #f0f8ff;
    }
    .lime-card.negative {
        border-left-color: #2ECC71;
        background: #f0fff4;
    }
    .conf-badge { font-size: 0.75rem; margin-left: 6px; }
    .section-header {
        font-size: 1.0rem; font-weight: 600; color: #2C3E50;
        border-bottom: 2px solid #3498DB; padding-bottom: 4px;
        margin: 12px 0 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "api_base_url":        os.getenv("API_BASE_URL", "http://localhost:8000"),
        "case_features":       None,
        "prediction_result":   None,
        "explanation_result":  None,
        "similar_cases_result": None,
        "extraction_confidence": {},
        "fir_text":            "",
        # What-if state
        "whatif_result":       None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()
API = st.session_state.api_base_url


# ══════════════════════════════════════════════════════════════════════════════
# API & EMBEDDED INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _get_local_registry():
    """Lazily load models in-process for standalone Streamlit Cloud deployment."""
    from backend.services import get_model_registry
    return get_model_registry()


def _api_get(endpoint: str, timeout: float = 3.0):
    try:
        r = httpx.get(f"{API}{endpoint}", timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception:
        # Seamless in-process fallback when FastAPI microservice is not running
        try:
            if endpoint == "/health":
                reg = _get_local_registry()
                ready = bool(reg and reg.xgboost and reg.knn)
                return {"status": "online", "mode": "embedded", "models_loaded": ready}, None
            elif endpoint == "/api/v1/sections":
                from src.statutory_mapping import get_all_sections
                sections = get_all_sections()
                return {k: (v.__dict__ if hasattr(v, "__dict__") else v) for k, v in sections.items()}, None
        except Exception as local_e:
            return None, f"Local fallback error: {local_e}"
        return None, "Backend unavailable"


def _api_post(endpoint: str, payload: dict, timeout: float = 30.0):
    try:
        r = httpx.post(f"{API}{endpoint}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception:
        # Seamless in-process fallback when FastAPI microservice is not running
        try:
            from backend.schemas import CaseInputSchema
            from backend.services import run_explanation, run_prediction, run_similar_cases
            from src.fir_parser import parse_fir_text

            if endpoint == "/api/v1/parse-fir":
                parsed = parse_fir_text(payload.get("raw_text", ""))
                return parsed.to_feature_dict(), None

            reg = _get_local_registry()
            schema = CaseInputSchema(**payload)

            if endpoint == "/api/v1/predict":
                return run_prediction(schema, reg).model_dump(), None
            elif endpoint == "/api/v1/explain":
                return run_explanation(schema, reg).model_dump(), None
            elif endpoint == "/api/v1/similar-cases":
                return run_similar_cases(schema, reg).model_dump(), None
        except Exception as local_e:
            return None, f"Local fallback error: {local_e}"
        return None, "Backend unavailable"


# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_eval_report():
    with open(EVAL_JSON) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_statutory_sections():
    data, err = _api_get("/api/v1/sections")
    return data or {}


@st.cache_data(show_spinner=False)
def load_benchmark_data():
    """Load dataset + models, recreate test split, run all model predictions."""
    df = pd.read_csv(DATA_PATH)
    le     = joblib.load(MODELS_DIR / "label_encoder.pkl")
    scaler = joblib.load(MODELS_DIR / "feature_scaler.pkl")

    df["counsel_type_enc"] = le.transform(df["counsel_type"])
    X = df[FEATURE_NAMES].copy()
    X["stolen_value_inr"] = np.log1p(X["stolen_value_inr"])

    y = df["sentence_severity"].values
    X_scaled = scaler.transform(pd.DataFrame(X, columns=FEATURE_NAMES))

    _, X_test, _, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42, stratify=y
    )

    models = {
        "XGBoost":       joblib.load(MODELS_DIR / "xgboost_calibrated.pkl"),
        "Random Forest": joblib.load(MODELS_DIR / "random_forest_calibrated.pkl"),
        "LightGBM":      joblib.load(MODELS_DIR / "lgbm_calibrated.pkl"),
    }

    probas = {name: m.predict_proba(X_test) for name, m in models.items()}
    return y_test, probas


@st.cache_data(show_spinner=False)
def _cached_pdf(case_json: str, pred_json: str, expl_json: str, sim_json: str) -> bytes:
    return generate_case_report(
        json.loads(case_json) if case_json else {},
        json.loads(pred_json) if pred_json else {},
        json.loads(expl_json) if expl_json else None,
        json.loads(sim_json) if sim_json else None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _gauge_chart(value: float, title: str = "Confidence") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 1),
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 16}},
        number={"suffix": "%", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": "#3498DB"},
            "steps": [
                {"range": [0,  40],  "color": "#EBF5FB"},
                {"range": [40, 70],  "color": "#D6EAF8"},
                {"range": [70, 100], "color": "#AED6F1"},
            ],
            "threshold": {"line": {"color": "red", "width": 2},
                          "thickness": 0.75, "value": 80},
        }
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=10, l=10, r=10))
    return fig


def _probability_bar(probs: dict, predicted: int) -> go.Figure:
    labels = [f"Sev {i}: {SEVERITY_LABELS[i]}" for i in range(4)]
    values = [probs.get(f"severity_{i}", 0) for i in range(4)]
    colors = [SEVERITY_COLORS[i] if i == predicted else "#BDC3C7" for i in range(4)]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v*100:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        height=200, margin=dict(t=10, b=10, l=10, r=50),
        xaxis=dict(range=[0, max(values) * 1.25], showgrid=False,
                   tickformat=".0%", title="Probability"),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        plot_bgcolor="white",
    )
    return fig


def _shap_waterfall(contributions: list) -> go.Figure:
    names  = [c["feature_name"].replace("_enc", "") for c in contributions]
    vals   = [c["shap_value"] for c in contributions]
    colors = ["#E74C3C" if v >= 0 else "#2ECC71" for v in vals]
    # Human-readable value labels
    raw_vals = [c["unscaled_value"] for c in contributions]
    display_names = [f"{n} = {rv:.0f}" if "inr" in n or "pct" in n
                     else f"{n} = {int(rv)}" for n, rv in zip(names, raw_vals)]

    fig = go.Figure(go.Bar(
        x=vals, y=display_names, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        height=320, margin=dict(t=20, b=10, l=10, r=60),
        xaxis=dict(title="SHAP Value", zeroline=True, zerolinewidth=2,
                   zerolinecolor="black"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        title=dict(text="SHAP Feature Contributions", font=dict(size=14)),
    )
    return fig


def _confusion_heatmap(cm_data: list, title: str) -> go.Figure:
    cm = np.array(cm_data)
    labels = ["Sev 0", "Sev 1", "Sev 2", "Sev 3"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Blues",
        text=cm, texttemplate="%{text}",
        textfont={"size": 14},
        showscale=False,
    ))
    fig.update_layout(
        title=title, height=300,
        xaxis_title="Predicted", yaxis_title="Actual",
        margin=dict(t=40, b=30, l=50, r=10),
    )
    return fig


def _roc_figure(y_test, probas: dict, selected_model: str) -> go.Figure:
    classes  = [0, 1, 2, 3]
    y_bin    = label_binarize(y_test, classes=classes)
    y_prob   = probas[selected_model]
    fig      = go.Figure()
    class_colors = ["#2ECC71", "#F39C12", "#E67E22", "#E74C3C"]

    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{SEVERITY_LABELS[cls]}  (AUC={roc_auc:.3f})",
            line=dict(color=class_colors[i], width=2),
        ))

    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
        line=dict(dash="dash", color="gray"), showlegend=False))
    fig.update_layout(
        title=f"ROC Curves — {selected_model} (One-vs-Rest)",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        height=380, legend=dict(x=0.55, y=0.1),
        plot_bgcolor="white",
    )
    return fig


def _pr_figure(y_test, probas: dict, selected_model: str) -> go.Figure:
    classes  = [0, 1, 2, 3]
    y_bin    = label_binarize(y_test, classes=classes)
    y_prob   = probas[selected_model]
    fig      = go.Figure()
    class_colors = ["#2ECC71", "#F39C12", "#E67E22", "#E74C3C"]

    for i, cls in enumerate(classes):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap           = average_precision_score(y_bin[:, i], y_prob[:, i])
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            name=f"{SEVERITY_LABELS[cls]}  (AP={ap:.3f})",
            line=dict(color=class_colors[i], width=2),
        ))

    fig.update_layout(
        title=f"Precision-Recall Curves — {selected_model}",
        xaxis_title="Recall", yaxis_title="Precision",
        height=380, legend=dict(x=0.01, y=0.01),
        plot_bgcolor="white",
    )
    return fig


def _radar_chart(baseline_probs: dict, modified_probs: dict) -> go.Figure:
    categories = [SEVERITY_LABELS[i] for i in range(4)]
    b_vals = [baseline_probs.get(f"severity_{i}", 0) * 100 for i in range(4)]
    m_vals = [modified_probs.get(f"severity_{i}", 0) * 100 for i in range(4)]
    # close the loop
    categories += [categories[0]]
    b_vals     += [b_vals[0]]
    m_vals     += [m_vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=b_vals, theta=categories, fill="toself",
        name="Baseline", line_color="#3498DB", opacity=0.6,
    ))
    fig.add_trace(go.Scatterpolar(
        r=m_vals, theta=categories, fill="toself",
        name="Modified", line_color="#E74C3C", opacity=0.6,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Probability Distribution Comparison",
        height=380, showlegend=True,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/law.png", width=72)
    st.title("LawAI ⚖️")
    st.caption("Legal Tech Predictor v1.0")
    st.divider()

    st.subheader("🔌 Backend Connection")
    api_url_input = st.text_input("FastAPI URL", value=st.session_state.api_base_url,
                                  label_visibility="collapsed")
    if api_url_input != st.session_state.api_base_url:
        st.session_state.api_base_url = api_url_input
        API = api_url_input
        st.cache_data.clear()

    # Health check
    health_data, health_err = _api_get("/health", timeout=3.0)
    if health_data and health_data.get("mode") == "embedded":
        st.success("🟢  Embedded ML Engine Active")
        st.caption("Running in standalone in-process mode (Streamlit Cloud).")
    elif health_data and health_data.get("status") == "online":
        st.success("🟢  FastAPI Backend Online")
    else:
        st.warning("🟠  Embedded Engine Active")
        st.caption("FastAPI not detected; using in-process ML pipeline.")

    st.divider()

    # Active case summary
    if st.session_state.prediction_result:
        pr = st.session_state.prediction_result
        sev = pr["predicted_severity"]
        st.subheader("📋 Active Case")
        st.markdown(f"""
        **Verdict:** {SEVERITY_EMOJIS[sev]} {pr['severity_label']}
        **Confidence:** {pr['confidence']*100:.1f}%
        **Section:** {pr['statutory_info']['bns_section']}
        **Bailable:** {'✅' if pr['statutory_info']['is_bailable'] else '❌'}
        """)
        try:
            pdf_bytes = _cached_pdf(
                json.dumps(st.session_state.case_features, default=str),
                json.dumps(st.session_state.prediction_result, default=str),
                json.dumps(st.session_state.explanation_result, default=str),
                json.dumps(st.session_state.similar_cases_result, default=str),
            )
            ts_label = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=f"LawAI_Report_{ts_label}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.caption(f"PDF unavailable: {e}")
        st.divider()

    with st.expander("⚠️ Legal Disclaimer"):
        st.caption(
            "This tool is intended solely for educational and research purposes. "
            "Predictions are based on historical patterns and do not constitute "
            "legal advice. Always consult a qualified legal professional. "
            "Model outputs may contain errors and must not be used for actual "
            "judicial decisions."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "⚖️  Case Triage & Prediction",
    "🔍  Explainability & Precedents",
    "🎛️  What-If Simulator",
    "📊  Model Benchmark",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CASE TRIAGE & FIR AUTO-FILL INPUT
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("⚖️ Case Triage & Prediction Engine")
    st.caption("Paste raw FIR text to auto-extract features, or fill in the controls manually.")

    col_left, col_right = st.columns([1, 1], gap="large")

    # ── LEFT: FIR Text Input ─────────────────────────────────────────────────
    with col_left:
        st.markdown('<div class="section-header">📄 FIR Narrative Input</div>',
                    unsafe_allow_html=True)

        fir_text = st.text_area(
            "Paste FIR text here",
            value=st.session_state.fir_text,
            height=200,
            placeholder=(
                "E.g. Complainant reported theft of articles valued at INR 46454 "
                "by breaking open the entrance lock. Accused was apprehended without "
                "weapon. Prior records show 0 past offenses. Recovered property "
                "stands at 80.3%."
            ),
            label_visibility="collapsed",
        )
        st.session_state.fir_text = fir_text

        extract_btn = st.button(
            "🔍 Auto-Extract Features via NLP",
            type="primary", use_container_width=True,
            disabled=not fir_text.strip(),
        )

        if extract_btn:
            with st.spinner("Running spaCy + Regex parser..."):
                result, err = _api_post("/api/v1/parse-fir", {"raw_text": fir_text})
            if err:
                st.error(f"Parser error: {err}")
            else:
                st.session_state.case_features = result
                st.session_state.extraction_confidence = result.get("extraction_confidence", {})
                st.success("✅ Features extracted! Review and adjust below.")

        # Confidence badges
        if st.session_state.extraction_confidence:
            st.markdown('<div class="section-header">Extraction Confidence</div>',
                        unsafe_allow_html=True)
            conf = st.session_state.extraction_confidence
            badge_cols = st.columns(3)
            items = list(conf.items())
            for i, (field, method) in enumerate(items):
                badge = CONFIDENCE_COLORS.get(method, "⚪")
                badge_cols[i % 3].markdown(
                    f"{badge} `{field}`  \n<span class='conf-badge'>{method}</span>",
                    unsafe_allow_html=True,
                )

    # ── RIGHT: Feature Controls ───────────────────────────────────────────────
    with col_right:
        st.markdown('<div class="section-header">🎚️ Case Feature Controls</div>',
                    unsafe_allow_html=True)

        feats = st.session_state.case_features or {}

        stolen_val = st.slider(
            "💰 Stolen Value (INR)",
            min_value=0, max_value=500_000, step=500,
            value=int(feats.get("stolen_value_inr", 25_000)),
            format="₹%d",
        )
        counsel_type = st.selectbox(
            "👨‍⚖️ Counsel Type",
            options=["Public Defender", "Private Counsel"],
            index=0 if feats.get("counsel_type", "Public Defender") == "Public Defender" else 1,
        )
        col_w, col_fe, col_conf = st.columns(3)
        weapon_present = col_w.toggle(
            "🔫 Weapon", value=bool(feats.get("weapon_present", 0))
        )
        forced_entry = col_fe.toggle(
            "🚪 Forced Entry", value=bool(feats.get("forced_entry", 0))
        )
        confession_recorded = col_conf.toggle(
            "📝 Confession", value=bool(feats.get("confession_recorded", 0))
        )

        prior_convictions = st.number_input(
            "📋 Prior Convictions", min_value=0, max_value=20,
            value=int(feats.get("prior_convictions", 0)),
        )
        property_recovered = st.slider(
            "📦 Property Recovered (%)",
            min_value=0.0, max_value=100.0, step=0.5,
            value=float(feats.get("property_recovered_pct", 50.0)),
        )

        # Build current case payload
        current_case = {
            "stolen_value_inr":       float(stolen_val),
            "counsel_type":           counsel_type,
            "weapon_present":         int(weapon_present),
            "prior_convictions":      int(prior_convictions),
            "property_recovered_pct": float(property_recovered),
            "confession_recorded":    int(confession_recorded),
            "forced_entry":           int(forced_entry),
            "extraction_confidence":  st.session_state.extraction_confidence,
        }
        st.session_state.case_features = current_case

        st.divider()
        analyze_col, reset_col = st.columns([3, 1])
        analyze_btn = analyze_col.button(
            "⚖️ Analyze Case", type="primary", use_container_width=True
        )
        reset_btn = reset_col.button("🔄 Reset", use_container_width=True)

        if reset_btn:
            st.session_state.case_features        = None
            st.session_state.prediction_result    = None
            st.session_state.explanation_result   = None
            st.session_state.similar_cases_result = None
            st.session_state.extraction_confidence = {}
            st.session_state.fir_text             = ""
            st.session_state.whatif_result         = None
            st.rerun()

        if analyze_btn:
            payload = {k: v for k, v in current_case.items()
                       if k != "extraction_confidence"}
            payload["extraction_confidence"] = current_case["extraction_confidence"]

            with st.spinner("Running prediction models..."):
                pred_res, pred_err = _api_post("/api/v1/predict", payload)
            if pred_err:
                st.error(f"Prediction error: {pred_err}")
            else:
                st.session_state.prediction_result = pred_res

            with st.spinner("Computing SHAP & LIME explanations..."):
                exp_res, exp_err = _api_post("/api/v1/explain", payload)
            if exp_err:
                st.warning(f"Explanation warning: {exp_err}")
            else:
                st.session_state.explanation_result = exp_res

            with st.spinner("Finding similar precedent cases..."):
                sim_res, sim_err = _api_post("/api/v1/similar-cases", payload)
            if sim_err:
                st.warning(f"Precedent warning: {sim_err}")
            else:
                st.session_state.similar_cases_result = sim_res

            if pred_res:
                sev = pred_res["predicted_severity"]
                st.success(
                    f"✅ Analysis complete — {SEVERITY_EMOJIS[sev]} **{pred_res['severity_label']}** "
                    f"| Confidence: {pred_res['confidence']*100:.1f}% "
                    f"| Switch to **Tab 2** to view results."
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VERDICT OUTPUT, SHAP EXPLANATIONS & PRECEDENTS
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("🔍 Explainable AI & Precedent Matcher")

    pred = st.session_state.prediction_result
    expl = st.session_state.explanation_result
    sims = st.session_state.similar_cases_result

    if not pred:
        st.info("⬅️ Run a case analysis in **Tab 1** first.")
        st.stop()

    # ── A: Verdict Banner ─────────────────────────────────────────────────────
    sev       = pred["predicted_severity"]
    sev_color = SEVERITY_COLORS[sev]
    sev_emoji = SEVERITY_EMOJIS[sev]

    col_verdict, col_gauge, col_probs = st.columns([1, 1, 2])

    with col_verdict:
        st.markdown(f"""
        <div class="verdict-box" style="background:{sev_color};">
            <div style="font-size:2.5rem;">{sev_emoji}</div>
            <div style="font-size:1.4rem; margin:4px 0;">Severity {sev}</div>
            <div style="font-size:1.1rem;">{pred['severity_label']}</div>
            <div style="font-size:0.85rem; margin-top:8px; opacity:0.9;">
                {pred['statutory_info']['bns_section']}<br>
                {'❌ Non-Bailable' if not pred['statutory_info']['is_bailable'] else '✅ Bailable'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_gauge:
        st.plotly_chart(_gauge_chart(pred["confidence"]), use_container_width=True)

    with col_probs:
        st.markdown('<div class="section-header">Ensemble Probability Distribution</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            _probability_bar(pred["ensemble_probabilities"], sev),
            use_container_width=True,
        )

    # ── B: Statutory Information ──────────────────────────────────────────────
    with st.expander("⚖️ Statutory Mapping Details", expanded=True):
        stat = pred["statutory_info"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("IPC Section", stat["ipc_section"].split("(")[0].strip())
        col2.metric("BNS Section", stat["bns_section"].split("(")[0].strip())
        col3.metric("Offense Type", stat["offense_type"])
        col4.metric("Bailable", "Yes ✅" if stat["is_bailable"] else "No ❌")
        st.caption(f"**Max Punishment (BNS):** {stat['max_punishment_bns']}")
        st.caption(f"**BNS Description:** {stat['bns_description']}")

    # ── C: Per-Model Breakdown ────────────────────────────────────────────────
    with st.expander("🤖 Individual Model Predictions"):
        for mv in pred["model_breakdown"]:
            c1, c2, c3 = st.columns([2, 1, 1])
            c1.markdown(f"**{mv['model_name']}**")
            c2.metric("Predicted Severity", f"{mv['predicted_severity']} — {SEVERITY_LABELS[mv['predicted_severity']]}")
            c3.metric("Confidence", f"{mv['confidence']*100:.1f}%")

    st.divider()

    # ── D: SHAP Waterfall ─────────────────────────────────────────────────────
    if expl:
        st.markdown('<div class="section-header">🧠 SHAP Feature Attributions</div>',
                    unsafe_allow_html=True)
        col_shap, col_legend = st.columns([3, 1])
        with col_shap:
            shap_sorted = sorted(expl["shap_contributions"], key=lambda x: x["rank"])
            st.plotly_chart(_shap_waterfall(shap_sorted), use_container_width=True)
        with col_legend:
            st.markdown("""
            **How to read:**
            - 🔴 **Red bars** push the prediction toward *higher* severity.
            - 🟢 **Green bars** push toward *lower* severity (mitigating).
            - Length = impact magnitude.
            """)
            st.caption(f"Base value (E[f(x)]): **{expl['shap_base_value']:.3f}**")
            st.caption(f"Model: {expl['explanation_model']}")

        # ── E: LIME Natural Language Cards ────────────────────────────────────
        st.markdown('<div class="section-header">📜 LIME Interpretation Cards</div>',
                    unsafe_allow_html=True)
        lime_cols = st.columns(2)
        for idx, card in enumerate(expl["lime_cards"]):
            card_class = "lime-card" if card["direction"] == "positive" else "lime-card negative"
            dir_icon   = "📈" if card["direction"] == "positive" else "📉"
            lime_cols[idx % 2].markdown(f"""
            <div class="{card_class}">
                <strong>{dir_icon} {card['feature_name'].replace('_', ' ').title()}</strong><br>
                <code>{card['condition']}</code><br>
                <em style="font-size:0.88rem;">{card['natural_language']}</em><br>
                <small>Weight: {card['weight']:+.3f}</small>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── F: K-NN Precedent Cards ───────────────────────────────────────────────
    if sims and sims.get("precedents"):
        st.markdown('<div class="section-header">📚 Top 3 Matching Historical Precedents</div>',
                    unsafe_allow_html=True)
        prec_cols = st.columns(3)

        for i, prec in enumerate(sims["precedents"]):
            p_sev     = prec["sentence_severity"]
            p_color   = SEVERITY_COLORS[p_sev]
            p_emoji   = SEVERITY_EMOJIS[p_sev]
            sim_pct   = prec["similarity_score"] * 100

            with prec_cols[i]:
                st.markdown(f"""
                <div class="precedent-card">
                    <div style="font-size:0.8rem; color:#888; margin-bottom:4px;">
                        {prec['case_id']}
                    </div>
                    <div style="font-weight:bold; font-size:1.05rem;">
                        {p_emoji} Severity {p_sev} — {SEVERITY_LABELS[p_sev]}
                    </div>
                    <div style="margin:6px 0;">
                        <strong>Similarity:</strong>
                        <span style="color:#3498DB; font-size:1.1rem; font-weight:bold;">
                            {sim_pct:.1f}%
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.progress(prec["similarity_score"])

                detail_cols = st.columns(2)
                detail_cols[0].caption(f"💰 ₹{prec['stolen_value_inr']:,.0f}")
                detail_cols[1].caption(f"{'🔫' if prec['weapon_present'] else '✋'} {'Weapon' if prec['weapon_present'] else 'No Weapon'}")
                detail_cols[0].caption(f"{'🚪' if prec['forced_entry'] else '🔓'} {'Forced Entry' if prec['forced_entry'] else 'No Forced Entry'}")
                detail_cols[1].caption(f"📋 {prec['prior_convictions']} prior(s)")
                st.caption(f"📦 {prec['property_recovered_pct']}% recovered")
                st.caption(f"⚖️ {prec['bns_section']}")

                with st.expander("📄 View FIR Text"):
                    st.write(prec["fir_text"])

    # ── G: PDF Report Generation ──────────────────────────────────────────────
    if pred:
        st.divider()
        st.markdown('<div class="section-header">📄 Formal Case Evaluation Report</div>',
                    unsafe_allow_html=True)
        st.caption("Generate and download a comprehensive, publication-ready PDF case assessment report containing all case features, verdict, SHAP risk factors, and precedent matches.")
        try:
            pdf_bytes = _cached_pdf(
                json.dumps(st.session_state.case_features, default=str),
                json.dumps(pred, default=str),
                json.dumps(expl, default=str),
                json.dumps(sims, default=str),
            )
            ts_label = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="⬇️ Download Formal Case Evaluation PDF",
                data=pdf_bytes,
                file_name=f"LawAI_Report_{ts_label}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WHAT-IF SENSITIVITY SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("🎛️ What-If Sensitivity Simulator")
    st.caption("Compare how changing individual case factors shifts the predicted verdict.")

    base_case = st.session_state.case_features
    base_pred = st.session_state.prediction_result

    if not base_pred:
        st.info("⬅️ Run a case analysis in **Tab 1** first to establish a baseline.")
        st.stop()

    col_base, col_mod = st.columns(2, gap="large")

    # ── Baseline (locked) ─────────────────────────────────────────────────────
    with col_base:
        st.markdown("### 📌 Baseline Case (Locked)")

        b_sev = base_pred["predicted_severity"]
        st.markdown(f"""
        <div class="verdict-box" style="background:{SEVERITY_COLORS[b_sev]};">
            {SEVERITY_EMOJIS[b_sev]} Severity {b_sev} — {base_pred['severity_label']}<br>
            <small>Confidence: {base_pred['confidence']*100:.1f}% &nbsp;|&nbsp;
            {base_pred['statutory_info']['bns_section']}</small>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        st.caption(f"💰 Stolen Value: ₹{base_case['stolen_value_inr']:,.0f}")
        st.caption(f"👨‍⚖️ Counsel: {base_case['counsel_type']}")
        st.caption(f"🔫 Weapon: {'Yes' if base_case['weapon_present'] else 'No'}")
        st.caption(f"🚪 Forced Entry: {'Yes' if base_case['forced_entry'] else 'No'}")
        st.caption(f"📋 Prior Convictions: {base_case['prior_convictions']}")
        st.caption(f"📦 Recovered: {base_case['property_recovered_pct']:.1f}%")
        st.caption(f"📝 Confession: {'Yes' if base_case['confession_recorded'] else 'No'}")

    # ── Modified Scenario ─────────────────────────────────────────────────────
    with col_mod:
        st.markdown("### 🔧 Modified Scenario")

        wi_stolen = st.slider(
            "💰 Stolen Value", 0, 500_000, int(base_case["stolen_value_inr"]),
            step=500, format="₹%d", key="wi_stolen"
        )
        wi_counsel = st.selectbox(
            "👨‍⚖️ Counsel Type",
            ["Public Defender", "Private Counsel"],
            index=0 if base_case["counsel_type"] == "Public Defender" else 1,
            key="wi_counsel",
        )
        wc1, wc2, wc3 = st.columns(3)
        wi_weapon  = wc1.toggle("🔫 Weapon",       value=bool(base_case["weapon_present"]),    key="wi_weapon")
        wi_forced  = wc2.toggle("🚪 Forced Entry", value=bool(base_case["forced_entry"]),      key="wi_forced")
        wi_conf    = wc3.toggle("📝 Confession",   value=bool(base_case["confession_recorded"]), key="wi_conf")

        wi_prior    = st.number_input("📋 Prior Convictions", 0, 20, int(base_case["prior_convictions"]), key="wi_prior")
        wi_recovery = st.slider("📦 Recovered %", 0.0, 100.0, float(base_case["property_recovered_pct"]), step=0.5, key="wi_recovery")

        recalc_btn = st.button("🔄 Recalculate Modified Prediction",
                               type="primary", use_container_width=True)

        if recalc_btn:
            wi_payload = {
                "stolen_value_inr":       float(wi_stolen),
                "counsel_type":           wi_counsel,
                "weapon_present":         int(wi_weapon),
                "prior_convictions":      int(wi_prior),
                "property_recovered_pct": float(wi_recovery),
                "confession_recorded":    int(wi_conf),
                "forced_entry":           int(wi_forced),
                "extraction_confidence":  {},
            }
            with st.spinner("Running modified prediction..."):
                wi_res, wi_err = _api_post("/api/v1/predict", wi_payload)
            if wi_err:
                st.error(f"Error: {wi_err}")
            else:
                st.session_state.whatif_result = wi_res

    # ── Delta Metrics & Radar ─────────────────────────────────────────────────
    wi_res = st.session_state.whatif_result
    if wi_res:
        st.divider()
        st.markdown("### 📊 Comparison Results")

        m1, m2, m3 = st.columns(3)
        b_sev_val  = base_pred["predicted_severity"]
        wi_sev_val = wi_res["predicted_severity"]
        sev_delta  = wi_sev_val - b_sev_val

        m1.metric(
            "Severity Class",
            f"{SEVERITY_EMOJIS[wi_sev_val]} {wi_sev_val} — {wi_res['severity_label']}",
            delta=f"{'↑ +' if sev_delta > 0 else '↓ ' if sev_delta < 0 else '='}{sev_delta}",
            delta_color="inverse" if sev_delta < 0 else "normal",
        )
        m2.metric(
            "Confidence",
            f"{wi_res['confidence']*100:.1f}%",
            delta=f"{(wi_res['confidence'] - base_pred['confidence'])*100:+.1f}%",
        )
        m3.metric(
            "BNS Section",
            wi_res['statutory_info']['bns_section'],
        )

        col_radar, col_bar = st.columns([1, 1])
        with col_radar:
            st.plotly_chart(
                _radar_chart(base_pred["ensemble_probabilities"],
                             wi_res["ensemble_probabilities"]),
                use_container_width=True,
            )
        with col_bar:
            st.markdown("**Probability Shifts**")
            labels = [SEVERITY_LABELS[i] for i in range(4)]
            b_vals = [base_pred["ensemble_probabilities"].get(f"severity_{i}", 0) for i in range(4)]
            w_vals = [wi_res["ensemble_probabilities"].get(f"severity_{i}", 0) for i in range(4)]
            delta_vals = [w - b for w, b in zip(w_vals, b_vals)]
            shift_fig = go.Figure()
            shift_fig.add_trace(go.Bar(
                x=[f"{d*100:+.1f}%" for d in delta_vals], y=labels,
                orientation="h",
                marker_color=["#E74C3C" if d > 0 else "#2ECC71" for d in delta_vals],
                text=[f"{d*100:+.1f}%" for d in delta_vals],
                textposition="outside",
            ))
            shift_fig.update_layout(
                height=300, xaxis_title="Probability Change",
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="white",
                margin=dict(t=10, b=10, l=10, r=50),
                showlegend=False,
            )
            st.plotly_chart(shift_fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL EVALUATION BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("📊 Multi-Model Evaluation Benchmark")
    st.caption("Performance comparison across XGBoost, Random Forest, and LightGBM on the 225-sample held-out test set.")

    # Load data
    eval_report = load_eval_report()

    # ── A: Summary Metrics Table ──────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Summary Metrics (Test Set — 225 samples)</div>',
                unsafe_allow_html=True)

    model_labels = {"xgboost": "XGBoost", "random_forest": "Random Forest", "lgbm": "LightGBM"}
    metrics_rows = []
    for key, label in model_labels.items():
        r = eval_report[key]
        metrics_rows.append({
            "Model": label,
            "Accuracy": f"{r['accuracy']*100:.1f}%",
            "Log Loss ↓": f"{r['log_loss']:.4f}",
            "ROC-AUC (macro) ↑": f"{r['roc_auc_macro_ovr']:.4f}",
        })

    metrics_df = pd.DataFrame(metrics_rows).set_index("Model")
    st.dataframe(
        metrics_df.style.highlight_min("Log Loss ↓", color="#d4edda")
                         .highlight_max("ROC-AUC (macro) ↑", color="#d4edda")
                         .highlight_max("Accuracy", color="#d4edda"),
        use_container_width=True,
    )
    st.caption("🟢 Highlighted cells = best performing model for that metric.")

    st.divider()

    # ── B: Confusion Matrices ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔢 Confusion Matrices</div>',
                unsafe_allow_html=True)

    cm_cols = st.columns(3)
    for idx, (key, label) in enumerate(model_labels.items()):
        cm_data = eval_report[key]["confusion_matrix"]
        cm_cols[idx].plotly_chart(
            _confusion_heatmap(cm_data, label),
            use_container_width=True,
        )

    st.divider()

    # ── C: Per-Class Precision/Recall/F1 Table ────────────────────────────────
    st.markdown('<div class="section-header">🎯 Per-Class Classification Report</div>',
                unsafe_allow_html=True)

    selected_model_key = st.radio(
        "Select model to inspect", list(model_labels.keys()),
        format_func=lambda k: model_labels[k], horizontal=True,
    )
    cls_report = eval_report[selected_model_key]["classification_report"]
    cls_rows = []
    for cls_id in ["0", "1", "2", "3"]:
        row = cls_report[cls_id]
        cls_rows.append({
            "Severity Class": f"Severity {cls_id} — {SEVERITY_LABELS[int(cls_id)]}",
            "Precision": f"{row['precision']:.3f}",
            "Recall":    f"{row['recall']:.3f}",
            "F1-Score":  f"{row['f1-score']:.3f}",
            "Support":   int(row['support']),
        })
    st.dataframe(pd.DataFrame(cls_rows).set_index("Severity Class"), use_container_width=True)

    st.divider()

    # ── D: ROC & PR Curves (requires local model loading) ─────────────────────
    st.markdown('<div class="section-header">📉 ROC & Precision-Recall Curves</div>',
                unsafe_allow_html=True)
    st.info("⏳ Loading models and generating curves (cached after first run)...")

    try:
        y_test, probas = load_benchmark_data()

        roc_model = st.selectbox(
            "Select model for ROC / PR curves",
            list(probas.keys()), key="roc_model_select",
        )

        col_roc, col_pr = st.columns(2)
        with col_roc:
            st.plotly_chart(_roc_figure(y_test, probas, roc_model),
                            use_container_width=True)
        with col_pr:
            st.plotly_chart(_pr_figure(y_test, probas, roc_model),
                            use_container_width=True)

    except Exception as e:
        st.error(f"Could not generate curves: {e}")
        st.caption("Ensure model .pkl files are present in `models/` directory.")

    st.divider()

    # ── E: Log Loss Comparison Bar ─────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Log Loss Comparison</div>',
                unsafe_allow_html=True)

    ll_models = [model_labels[k] for k in model_labels]
    ll_values = [eval_report[k]["log_loss"] for k in model_labels]
    ll_colors = ["#2ECC71" if v == min(ll_values) else "#AED6F1" for v in ll_values]

    ll_fig = go.Figure(go.Bar(
        x=ll_models, y=ll_values,
        marker_color=ll_colors,
        text=[f"{v:.4f}" for v in ll_values],
        textposition="outside",
    ))
    ll_fig.update_layout(
        height=280,
        yaxis_title="Log Loss (lower = better)",
        yaxis=dict(range=[0, max(ll_values) * 1.15]),
        plot_bgcolor="white",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(ll_fig, use_container_width=True)
    st.caption("🟢 Green = lowest log loss (best calibrated probability output).")
