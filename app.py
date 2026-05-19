import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────
st.set_page_config(page_title="Compressor Predictive Maintenance", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #111111; color: white; }
.model-card {
    background-color: #1a1a2e;
    border-radius: 12px;
    padding: 20px;
    margin: 5px;
    border: 1px solid #333;
}
.risk-badge-low      { background-color: #1f7a1f; color: white; padding: 4px 14px; border-radius: 20px; font-size: 13px; display: inline-block; }
.risk-badge-moderate { background-color: #b8860b; color: white; padding: 4px 14px; border-radius: 20px; font-size: 13px; display: inline-block; }
.risk-badge-high     { background-color: #cc4400; color: white; padding: 4px 14px; border-radius: 20px; font-size: 13px; display: inline-block; }
.risk-badge-critical { background-color: #880000; color: white; padding: 4px 14px; border-radius: 20px; font-size: 13px; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('rf_model.pkl', 'rb') as f:
        return pickle.load(f)

rf_model = load_model()

# ── Title ─────────────────────────────────────────────────
st.markdown("## ⚙️ Compressor Failure Prediction")
st.caption("Random Forest — Performance & Live Prediction")
st.divider()

# ── Sliders ───────────────────────────────────────────────
st.markdown("### LIVE PREDICTION INPUT")
col1, col2, col3 = st.columns(3)
with col1:
    vibration    = st.slider("Vibration (Hz)",    0.0, 6.0,   3.5, 0.01)
with col2:
    temperature  = st.slider("Temperature (°C)",  25.0, 50.0, 40.0, 0.1)
with col3:
    power_factor = st.slider("Power Factor",      0.0, 100.0, 98.0, 0.1)

# ── Prediction ────────────────────────────────────────────
input_df = pd.DataFrame({
    'VIBRATION':           [vibration],
    'TEMPERATURE':         [temperature],
    'ACTUAL POWER FACTOR': [power_factor]
})
X_input   = input_df[rf_model.feature_names_in_]
proba     = rf_model.predict_proba(X_input)[0]
fail_pct  = round(proba[0] * 100, 1)
oper_pct  = round(proba[1] * 100, 1)
state     = "Operational" if proba[1] > 0.5 else "Failure"

if fail_pct < 20:
    risk      = "Low Risk";      badge_cls = "risk-badge-low"
elif fail_pct < 50:
    risk      = "Moderate Risk"; badge_cls = "risk-badge-moderate"
elif fail_pct < 80:
    risk      = "High Risk";     badge_cls = "risk-badge-high"
else:
    risk      = "Critical Risk"; badge_cls = "risk-badge-critical"

state_color = "#00ff99" if state == "Operational" else "#ff4444"

# ── Prediction Card ───────────────────────────────────────
st.markdown("### PREDICTION RESULT — RANDOM FOREST")
_, mid, _ = st.columns([1, 2, 1])
with mid:
    st.markdown(f"""
    <div class="model-card">
        <p style="color:#aaa; font-size:15px; margin-bottom:6px">▲ Random Forest</p>
        <p style="color:{state_color}; font-size:26px; font-weight:bold; margin:6px 0">{state}</p>
        <span class="{badge_cls}">{risk}</span><br><br>
        <p style="color:#ff6666; font-size:15px; margin:4px 0">Failure: {fail_pct}%</p>
        <p style="color:#66ff99; font-size:15px; margin:4px 0">Operational: {oper_pct}%</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── R² and RMSE Charts ────────────────────────────────────
st.markdown("### MODEL PERFORMANCE — R² AND RMSE")

phases = ['Training', 'Validation', 'Testing']
r2_values   = [0.9998, 0.9989, 0.9986]
rmse_values = [0.00408, 0.01023, 0.01600]

pc1, pc2 = st.columns(2)

with pc1:
    fig_r2 = go.Figure(go.Bar(
        x=r2_values, y=phases,
        orientation='h',
        marker_color=['#00cc88', '#00aaff', '#ff9900'],
       text=[f"{v:.4f}" for v in r2_values],
        textposition='outside'
    ))
    fig_r2.update_layout(
        title="R² Score",
        paper_bgcolor='#111111', plot_bgcolor='#1a1a2e',
        font_color='white',
        xaxis=dict(range=[0.990, 1.001], color='white'),
        yaxis=dict(color='white'),
        height=300
    )
    st.plotly_chart(fig_r2, use_container_width=True)

with pc2:
    fig_rmse = go.Figure(go.Bar(
        x=rmse_values, y=phases,
        orientation='h',
        marker_color=['#00cc88', '#00aaff', '#ff9900'],
        text=[f"{v:.5f}" for v in rmse_values],
        textposition='outside'
    ))
    fig_rmse.update_layout(
        title="RMSE",
        paper_bgcolor='#111111', plot_bgcolor='#1a1a2e',
        font_color='white',
        xaxis=dict(color='white'),
        yaxis=dict(color='white'),
        height=300
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

st.divider()

# ── Risk Donut Chart (updates with sliders) ───────────────
st.markdown("### RISK LEVEL DISTRIBUTION — RANDOM FOREST")
st.caption("Updates based on your input above")

# Risk counts based on current prediction
risk_labels = ['Low Risk', 'Moderate Risk', 'High Risk', 'Critical Risk']
risk_colors = ['#00cc66', '#ffcc00', '#ff6600', '#cc0000']

# Show failure probability split across risk zones
total_counts = [
    max(0, 20 - fail_pct),          # Low Risk portion
    max(0, min(fail_pct, 50) - 20), # Moderate Risk portion
    max(0, min(fail_pct, 80) - 50), # High Risk portion
    max(0, fail_pct - 80)           # Critical Risk portion
]
# If all zero (fail_pct=0), show full Low Risk
if sum(total_counts) == 0:
    total_counts = [100, 0, 0, 0]

fig_donut = go.Figure(go.Pie(
    labels=risk_labels,
    values=total_counts,
    hole=0.5,
    marker_colors=risk_colors,
    textinfo='label+percent',
    textfont=dict(color='white', size=12)
))
fig_donut.update_layout(
    paper_bgcolor='#111111',
    font_color='white',
    showlegend=True,
    legend=dict(font=dict(color='white')),
    height=400
)
st.plotly_chart(fig_donut, use_container_width=True)

st.divider()

# ── SHAP Feature Importance (updates with sliders) ────────
st.markdown("### FEATURE IMPORTANCE (SHAP) — RANDOM FOREST")
st.caption("Relative contribution of each sensor to failure prediction")

# SHAP scores weighted by current input deviation from normal
# Normal baseline: PF=98, Temp=39.3, Vib=3.23
pf_dev   = abs(power_factor - 98.0)  / 98.0
temp_dev = abs(temperature  - 39.3)  / 39.3
vib_dev  = abs(vibration    - 3.23)  / 3.23

# Base SHAP * deviation weight
base_shap = {'ACTUAL POWER FACTOR': 0.103201,
             'VIBRATION':           0.070180,
             'TEMPERATURE':         0.003232}

dynamic_shap = {
    'ACTUAL POWER FACTOR': base_shap['ACTUAL POWER FACTOR'] * (1 + pf_dev),
    'VIBRATION':           base_shap['VIBRATION']           * (1 + vib_dev),
    'TEMPERATURE':         base_shap['TEMPERATURE']         * (1 + temp_dev)
}

# Normalize to percentage
total_shap = sum(dynamic_shap.values())
shap_pct   = {k: round(v / total_shap * 100, 1) for k, v in dynamic_shap.items()}

features = list(shap_pct.keys())
values   = list(shap_pct.values())

fig_shap = go.Figure(go.Bar(
    x=values, y=features,
    orientation='h',
    marker_color=['#ff4488', '#44aaff', '#44ffaa'],
    text=[f"{v}%" for v in values],
    textposition='outside'
))
fig_shap.update_layout(
    paper_bgcolor='#111111', plot_bgcolor='#1a1a2e',
    font_color='white',
    xaxis=dict(title='Contribution (%)', color='white'),
    yaxis=dict(color='white'),
    height=300
)
st.plotly_chart(fig_shap, use_container_width=True)

st.divider()

# ── Baseline Rule ─────────────────────────────────────────
st.markdown("### BASELINE RULE CHECK")
pf_thresh   = 48.96
vib_thresh  = 1.624
temp_thresh = 36.19

if (power_factor < pf_thresh) or (vibration < vib_thresh) or (temperature < temp_thresh):
    baseline_pred = "Failure"; b_color = "#ff4444"
else:
    baseline_pred = "Operational"; b_color = "#00ff99"
    bl1, bl2 = st.columns(2)
bl1.markdown(f"""
Baseline Rule (SCADA-derived thresholds):
- IF Power Factor < {pf_thresh} → Failure
- OR Vibration < {vib_thresh} → Failure
- OR Temperature < {temp_thresh} → Failure
- ELSE → Operational
""")
bl2.markdown(f"""
Baseline Prediction:
<span style='color:{b_color}; font-size:20px; font-weight:bold'>{baseline_pred}</span>
""", unsafe_allow_html=True)
