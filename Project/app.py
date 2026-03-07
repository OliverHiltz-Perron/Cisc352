"""
Virtual Lifter — Streamlit Front-End
=====================================
A polished UI that wraps the Bayesian Network from prob-interface.py.
Users enter their lifestyle inputs (Sleep, Macros, TimeRest) and
the app performs probabilistic inference to recommend a workout plan.

Run with:  python -m streamlit run app.py
"""

import streamlit as st
import numpy as np
from pgmpy.inference import VariableElimination

# Import the model builder from the refactored prob-interface module
import importlib
prob_interface = importlib.import_module("prob-interface")

# ──────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Virtual Lifter",
    page_icon="🏋️",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* Header gradient */
.hero-title {
    background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0;
}
.hero-sub {
    text-align: center;
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0;
    margin-bottom: 2rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #1e1b4b, #312e81);
    border: 1px solid #4338ca44;
    border-radius: 16px;
    padding: 1.3rem 1.2rem;
    text-align: center;
    margin-bottom: 0.8rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(99,102,241,0.25);
}
.metric-label {
    color: #a5b4fc;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.metric-value {
    color: #e0e7ff;
    font-size: 1.55rem;
    font-weight: 700;
}
.metric-prob {
    color: #818cf8;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.15rem;
}

/* Probability bar */
.prob-bar-bg {
    background: #1e1b4b;
    border-radius: 8px;
    height: 10px;
    width: 100%;
    margin-top: 6px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 10px;
    border-radius: 8px;
    background: linear-gradient(90deg, #6366f1, #a855f7);
}

/* Risk badge */
.risk-low { color: #4ade80; }
.risk-elevated { color: #facc15; }
.risk-critical { color: #f87171; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# BUILD MODEL (cached — only runs once)
# ──────────────────────────────────────────────
@st.cache_resource
def get_model_and_inference():
    """Build the BN model and inference engine once, then cache."""
    model = prob_interface.build_model()
    return model, VariableElimination(model)


model, inference_engine = get_model_and_inference()

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown('<p class="hero-title">🏋️ Virtual Lifter</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">AI-powered workout recommendations using a Bayesian Network</p>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# USER INPUTS
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📋 Your Lifestyle Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    sleep = st.selectbox(
        "😴 Hours of Sleep",
        options=['<6 hours', '6-8 hours', '>8 hours'],
        index=1,
        help="How many hours did you sleep last night?"
    )

with col2:
    macros = st.selectbox(
        "🍽️ Nutrition / Macros",
        options=['Deficit', 'Maintenance', 'Surplus'],
        index=1,
        help="Are you in a caloric deficit, at maintenance, or in a surplus?"
    )

with col3:
    time_rest = st.selectbox(
        "⏸️ Days Since Last Session",
        options=['0 Days', '1-2 Days', '3+ Days'],
        index=1,
        help="How many rest days since your last workout?"
    )

# ──────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────
evidence = {
    'Sleep': sleep,
    'Macros': macros,
    'TimeRest': time_rest,
}

query_vars = ['Recovery', 'Fatigue', 'Readiness', 'Risk',
              'Soreness', 'Weight', 'Volume', 'RPE']

results = {}
for var in query_vars:
    posterior = inference_engine.query([var], evidence=evidence)
    states = posterior.state_names[var]
    probs = posterior.values
    best_idx = int(np.argmax(probs))
    results[var] = {
        'best_state': states[best_idx],
        'best_prob': float(probs[best_idx]),
        'all_states': states,
        'all_probs': [float(p) for p in probs],
    }


def render_metric_card(label, value, prob, extra_class=""):
    """Render a styled metric card with a probability bar."""
    bar_width = int(prob * 100)
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {extra_class}">{value}</div>
        <div class="metric-prob">{prob:.0%} confidence</div>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{bar_width}%"></div>
        </div>
    </div>
    """


# ──────────────────────────────────────────────
# BODY STATUS SECTION
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🧠 Inferred Body Status")

bc1, bc2, bc3, bc4 = st.columns(4)

with bc1:
    st.markdown(render_metric_card(
        "Recovery", results['Recovery']['best_state'],
        results['Recovery']['best_prob']
    ), unsafe_allow_html=True)

with bc2:
    st.markdown(render_metric_card(
        "Fatigue", results['Fatigue']['best_state'],
        results['Fatigue']['best_prob']
    ), unsafe_allow_html=True)

with bc3:
    st.markdown(render_metric_card(
        "Readiness", results['Readiness']['best_state'],
        results['Readiness']['best_prob']
    ), unsafe_allow_html=True)

with bc4:
    risk_val = results['Risk']['best_state']
    risk_class = {
        'Low': 'risk-low', 'Elevated': 'risk-elevated', 'Critical': 'risk-critical'
    }.get(risk_val, '')
    st.markdown(render_metric_card(
        "Injury Risk", risk_val,
        results['Risk']['best_prob'], extra_class=risk_class
    ), unsafe_allow_html=True)


# ──────────────────────────────────────────────
# WORKOUT RECOMMENDATION SECTION
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("### 💪 Recommended Workout")

wc1, wc2, wc3, wc4 = st.columns(4)

with wc1:
    st.markdown(render_metric_card(
        "Weight / Load", results['Weight']['best_state'],
        results['Weight']['best_prob']
    ), unsafe_allow_html=True)

with wc2:
    st.markdown(render_metric_card(
        "Volume", results['Volume']['best_state'],
        results['Volume']['best_prob']
    ), unsafe_allow_html=True)

with wc3:
    st.markdown(render_metric_card(
        "Expected RPE", results['RPE']['best_state'],
        results['RPE']['best_prob']
    ), unsafe_allow_html=True)

with wc4:
    st.markdown(render_metric_card(
        "Soreness", results['Soreness']['best_state'],
        results['Soreness']['best_prob']
    ), unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DETAILED PROBABILITY BREAKDOWN (expandable)
# ──────────────────────────────────────────────
st.markdown("---")

with st.expander("📊 Full Probability Breakdown", expanded=False):
    for var in query_vars:
        st.markdown(f"**{var}**")
        r = results[var]
        for state, prob in zip(r['all_states'], r['all_probs']):
            bar_pct = int(prob * 100)
            marker = " ◀" if state == r['best_state'] else ""
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
                f'<span style="min-width:180px;color:#cbd5e1;font-size:0.85rem;">{state}{marker}</span>'
                f'<div style="flex:1;background:#1e1b4b;border-radius:6px;height:8px;overflow:hidden;">'
                f'<div style="width:{bar_pct}%;height:8px;border-radius:6px;'
                f'background:linear-gradient(90deg,#6366f1,#a855f7);"></div></div>'
                f'<span style="min-width:45px;text-align:right;color:#818cf8;font-size:0.82rem;">{prob:.0%}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#64748b;font-size:0.8rem;">'
    'Virtual Lifter — CISC 352 Project · Powered by pgmpy Bayesian Inference'
    '</p>',
    unsafe_allow_html=True,
)
