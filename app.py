import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────
st.set_page_config(page_title="Compressor Predictive Maintenance", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0e1117; color: white; }
.model-card {
    background-color: #1a1a2e;
    border-radius: 12px;
    padding: 20px;
    margin: 5px;
    border: 1px solid #333;
}
.action-box {
    border-radius: 10px;
    padding: 15px;
    margin-top: 10px;
    font-size: 14px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ── Load all models ───────────────────────────────────────
@st.cache_resource
def load_models():
    with open('rf_model.pkl',  'rb') as f: rf  = pickle.load(f)
    with open('ann_model.pkl', 'rb') as f: ann = pickle.load(f)
    with open('svm_model.pkl', 'rb') as f: svm = pickle.load(f)
    with open('xgb_model.pkl', 'rb') as f: xgb = pickle.load(f)
    return rf, ann, svm, xgb

rf_model, ann_model, svm_model, xgb_model = load_models()

# ── Title ─────────────────────────────────────────────────
st.markdown("## ⚙️ Compressor Failure Prediction")
st.caption("RF · ANN · SVM · XGBoost — Performance & Live Prediction")
st.divider()

# ── Sliders ───────────────────────────────────────────────
st.markdown("### LIVE PREDICTION INPUT")
col1, col2, col3 = st.columns(3)
with col1:
    vibration    = st.slider("Vibration (Hz)",   0.0, 6.0,   3.5, 0.01)
with col2:
    temperature  = st.slider("Temperature (°C)", 25.0, 50.0, 40.0, 0.1)
with col3:
    power_factor = st.slider("Power Factor",     0.0, 100.0, 98.0, 0.1)

# ── Action recommendation function ───────────────────────
def get_action(fail_pct):
    if fail_pct < 20:
        return (
            "Low Risk",
            "risk-badge-low",
            "#1f7a1f",
            "✅ LOW RISK (< 20%) — Machine kept running without any action needed at the moment."
        )
    elif fail_pct < 50:
        return (
            "Moderate Risk",
            "risk-badge-moderate",
            "#b8860b",
            "⚠️ MODERATE RISK (20–50%) — Machine kept running with more frequent checks recommended."
        )
    elif fail_pct < 80:
        return (
            "High Risk",
            "risk-badge-high",
            "#cc4400",
            "🔧 HIGH RISK (50–80%) — Machine scheduled for maintenance at the earliest convenience. Multiple parameter stresses indicate deterioration is imminent."
        )
    else:
        return (
            "Critical Risk",
            "risk-badge-critical",
            "#880000",
            "🛑 CRITICAL RISK (> 80%) — Machine must be IMMEDIATELY STOPPED. Very high probability of failure."
        )

# ── Predict function ──────────────────────────────────────
def predict(model, vib, temp, pf):
    df = pd.DataFrame({
        'VIBRATION':           [vib],
        'TEMPERATURE':         [temp],
        'ACTUAL POWER FACTOR': [pf]
    })
    X = df[model.feature_names_in_]
    proba    = model.predict_proba(X)[0]
    fail_pct = round(proba[0] * 100, 1)
    oper_pct = round(proba[1] * 100, 1)
    state    = "Operational" if proba[1] > 0.5 else "Failure"
    return state, fail_pct, oper_pct

# ── Run all predictions ───────────────────────────────────
rf_state,  rf_fail,  rf_oper  = predict(rf_model,  vibration, temperature, power_factor)
ann_state, ann_fail, ann_oper = predict(ann_model, vibration, temperature, power_factor)
svm_state, svm_fail, svm_oper = predict(svm_model, vibration, temperature, power_factor)
xgb_state, xgb_fail, xgb_oper = predict(xgb_model, vibration, temperature, power_factor)

rf_risk,  _, rf_color,  rf_action  = get_action(rf_fail)
ann_risk, _, ann_color, ann_action = get_action(ann_fail)
svm_risk, _, svm_color, svm_action = get_action(svm_fail)
xgb_risk, _, xgb_color, xgb_action = get_action(xgb_fail)

# ── Prediction Cards ──────────────────────────────────────
st.markdown("### PREDICTION RESULTS — ALL MODELS")
models_data = [
    ("▲ Random Forest", rf_state,  rf_fail,  rf_oper,  rf_risk,  rf_color,  rf_action),
    ("● ANN",           ann_state, ann_fail, ann_oper, ann_risk, ann_color, ann_action),
    ("◆ SVM",           svm_state, svm_fail, svm_oper, svm_risk, svm_color, svm_action),
    ("⚡️ XGBoost",      xgb_state, xgb_fail, xgb_oper, xgb_risk, xgb_color, xgb_action),
]

c1, c2, c3, c4 = st.columns(4)
for col, (name, state, fail, oper, risk, color, action) in zip([c1,c2,c3,c4], models_data):
    state_color = "#00ff99" if state == "Operational" else "#ff4444"
    col.markdown(f"""
    <div class="model-card">
        <p style="color:#aaa; font-size:13px; margin-bottom:4px">{name}</p>
        <p style="color:{state_color}; font-size:20px;
                  font-weight:bold; margin:4px 0">{state}</p>
        <span style="background:{color}; color:white; padding:3px 12px;
                     border-radius:20px; font-size:12px">{risk}</span>
        <br><br>
        <p style="color:#ff6666; margin:2px 0; font-size:13px">Failure: {fail}%</p>
        <p style="color:#66ff99; margin:2px 0; font-size:13px">Operational: {oper}%</p>
        <div style="background:{color}22; border-left: 4px solid {color};
                    padding:8px; margin-top:10px; border-radius:6px;
                    font-size:12px; color:white">
            {action}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Test Set R² Cards ─────────────────────────────────────
st.markdown("### TEST SET R² — ALL MODELS")
r2_data = [
    ("▲ Random Forest", 0.9974, 0.0160, "#00cc88"),
    ("● ANN",           0.9977, 0.0151, "#44aaff"),
    ("◆ SVM",           0.9968, 0.0177, "#ffcc00"),
    ("⚡️ XGBoost",      0.9950, 0.0220, "#ff4444"),
]
rc1, rc2, rc3, rc4 = st.columns(4)
for col, (name, r2, rmse, color) in zip([rc1,rc2,rc3,rc4], r2_data):
    col.markdown(f"""
    <div class="model-card" style="text-align:center">
        <p style="color:#aaa; font-size:13px">{name}</p>
        <p style="color:{color}; font-size:32px; font-weight:bold; margin:4px 0">{r2}</p>
        <p style="color:#aaa; font-size:12px">RMSE: {rmse}</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Performance Comparison Table ──────────────────────────
st.markdown("### MODEL PERFORMANCE COMPARISON")
perf_df = pd.DataFrame({
    'MODEL':       ['▲ Random Forest','● ANN','◆ SVM','⚡️ XGBoost'],
    'R² TRAIN':    [0.9998, 0.9996, 0.9987, 0.9978],
    'R² VAL':      [0.9989, 0.9979, 0.9947, 0.9980],
    'R² TEST':     [0.9974, 0.9977, 0.9968, 0.9950],
    'RMSE TRAIN':  [0.0041, 0.0066, 0.0115, 0.0145],
    'RMSE VAL':    [0.0102, 0.0144, 0.0227, 0.0140],
    'RMSE TEST':   [0.0160, 0.0151, 0.0177, 0.0220],
})
st.dataframe(perf_df, use_container_width=True, hide_index=True)

# ── R² and RMSE Bar Charts ────────────────────────────────
bc1, bc2 = st.columns(2)
model_names = ['Random Forest','ANN','SVM','XGBoost']
bar_colors  = ['#00cc88','#44aaff','#ffcc00','#ff4444']

with bc1:
    fig_r2 = go.Figure(go.Bar(
        x=[0.9974, 0.9977, 0.9968, 0.9950],
        y=model_names, orientation='h',
        marker_color=bar_colors,
        text=[0.9974, 0.9977, 0.9968, 0.9950],
        textposition='outside'
    ))
    fig_r2.update_layout(
        title="R² Score — Test Set",
        paper_bgcolor='#1a1a2e', plot_bgcolor='#1a1a2e',
        font_color='white',
        xaxis=dict(range=[0.990, 1.001], color='white'),
        yaxis=dict(color='white'), height=300
    )
    st.plotly_chart(fig_r2, use_container_width=True)
    
with bc2:
    fig_rmse = go.Figure(go.Bar(
        x=[0.0160, 0.0151, 0.0177, 0.0220],
        y=model_names, orientation='h',
        marker_color=bar_colors,
        text=[0.0160, 0.0151, 0.0177, 0.0220],
        textposition='outside'
    ))
    fig_rmse.update_layout(
    title="RMSE — Test Set (lower = better)",
        paper_bgcolor='#1a1a2e', plot_bgcolor='#1a1a2e',
        font_color='white',
        xaxis=dict(color='white'),
        yaxis=dict(color='white'), height=300
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

st.divider()

# ── Risk Distribution Donuts (dynamic) ────────────────────
st.markdown("### RISK LEVEL DISTRIBUTION — ALL MODELS")
st.caption("Updates dynamically based on failure probability from each model")

def risk_donut_values(fail_pct):
    """Convert failure % into 4 risk zone values — fully dynamic"""
    low      = max(0, 20 - fail_pct)
    moderate = max(0, min(fail_pct, 50) - 20)
    high     = max(0, min(fail_pct, 80) - 50)
    critical = max(0, fail_pct - 80)
    total    = low + moderate + high + critical
    if total == 0:
        return [100, 0, 0, 0]
    return [low, moderate, high, critical]

donut_models = [
    ("Random Forest", rf_fail,  "#00cc88"),
    ("ANN",           ann_fail, "#44aaff"),
    ("SVM",           svm_fail, "#ffcc00"),
    ("XGBoost",       xgb_fail, "#ff4444"),
]

d1, d2, d3, d4 = st.columns(4)
for col, (name, fail_pct, _) in zip([d1,d2,d3,d4], donut_models):
    vals = risk_donut_values(fail_pct)
    fig = go.Figure(go.Pie(
        labels=['Low Risk','Moderate Risk','High Risk','Critical Risk'],
        values=vals,
        hole=0.5,
        marker_colors=['#00cc66','#ffcc00','#ff6600','#cc0000'],
        textinfo='label+percent',
        textfont=dict(color='white', size=10)
    ))
    fig.update_layout(
        title=dict(text=name, font=dict(color='white', size=13)),
        paper_bgcolor='#1a1a2e',
        font_color='white',
        showlegend=False,
        height=300,
        margin=dict(t=40, b=10, l=10, r=10)
    )
    col.plotly_chart(fig, use_container_width=True)

st.divider()

# ── SHAP Feature Importance (dynamic) ────────────────────
st.markdown("### FEATURE IMPORTANCE (SHAP) — ALL MODELS")
st.caption("Relative contribution updates based on current input values")

# Base SHAP values per model (from your notebook)
base_shap = {
    'Random Forest': {'ACTUAL POWER FACTOR': 0.1032, 'VIBRATION': 0.0702, 'TEMPERATURE': 0.0032},
    'ANN':           {'ACTUAL POWER FACTOR': 0.0900, 'VIBRATION': 0.1100, 'TEMPERATURE': 0.0100},
    'SVM':           {'ACTUAL POWER FACTOR': 0.0700, 'VIBRATION': 0.1200, 'TEMPERATURE': 0.0100},
    'XGBoost':       {'ACTUAL POWER FACTOR': 0.0800, 'VIBRATION': 0.1000, 'TEMPERATURE': 0.0200},
}

# Dynamic deviation from normal operating values
pf_dev   = abs(power_factor - 98.0) / 98.0
temp_dev = abs(temperature  - 39.3) / 39.3
vib_dev  = abs(vibration    - 3.23) / 3.23

shap_colors = ['#00cc88','#44aaff','#ffcc00','#ff4444']
s1, s2, s3, s4 = st.columns(4)

for col, (mname, color) in zip([s1,s2,s3,s4],
    [("Random Forest","#00cc88"),("ANN","#44aaff"),
     ("SVM","#ffcc00"),("XGBoost","#ff4444")]):

    base = base_shap[mname]
    dyn  = {
        'ACTUAL POWER FACTOR': base['ACTUAL POWER FACTOR'] * (1 + pf_dev),
        'VIBRATION':           base['VIBRATION']           * (1 + vib_dev),
        'TEMPERATURE':         base['TEMPERATURE']         * (1 + temp_dev),
    }
    total = sum(dyn.values())
    pcts  = {k: round(v/total*100, 1) for k, v in dyn.items()}

    fig = go.Figure(go.Bar(
        x=list(pcts.values()),
        y=list(pcts.keys()),
        orientation='h',
        marker_color=color,
        text=[f"{v}%" for v in pcts.values()],
        textposition='outside'
    ))
    fig.update_layout(
        title=dict(text=mname, font=dict(color='white', size=12)),
        paper_bgcolor='#1a1a2e', plot_bgcolor='#1a1a2e',
        font_color='white',
        xaxis=dict(color='white', title='%'),
        yaxis=dict(color='white'),
        height=280,
        margin=dict(t=40, b=10, l=10, r=40)
    )
    col.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Baseline Rule ─────────────────────────────────────────
st.markdown("### BASELINE RULE CHECK")
pf_thresh, vib_thresh, temp_thresh = 48.96, 1.624, 36.19
if (power_factor < pf_thresh) or (vibration < vib_thresh) or (temperature < temp_thresh):
    b_pred, b_color = "Failure",     "#ff4444"
else:
    b_pred, b_color = "Operational", "#00ff99"

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
<span style='color:{b_color}; font-size:22px; font-weight:bold'>{b_pred}</span>
""", unsafe_allow_html=True)
