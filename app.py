import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import json
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="ROAS Predictor Pro", layout="wide")

st.title("📈 ROAS Curve Learner & Predictor")
st.markdown("Dynamic Calibration for **Hybrid-Puzzle** Cohorts.")

# --- PERSISTENCE LOGIC ---
MODEL_FILE = "learned_settings.json"

def save_model_params(params):
    with open(MODEL_FILE, 'w') as f:
        json.dump(list(params), f)

def load_model_params():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'r') as f:
            return np.array(json.load(f))
    return None

def log_model(t, a, b):
    return a * np.log(t + 1) + b

def flexible_model(t, a_val, b_val, s_adj, t_adj):
    # s_adj = Early Growth, t_adj = LTV Tail
    return (a_val * (np.log(t + 1) ** t_adj)) * s_adj + b_val

def get_learned_curve(df):
    if 'roas_d180' not in df.columns:
        st.error("Historical file must contain 'roas_d180' column.")
        return None
    df_filtered = df[df['roas_d180'] > 0].copy()
    roas_cols = [f'roas_d{i}' for i in range(181) if f'roas_d{i}' in df.columns]
    mean_curve = df_filtered[roas_cols].replace(0, np.nan).mean()
    days = np.array([int(c.split('_d')[1]) for c in mean_curve.index])
    values = mean_curve.values
    mask = ~np.isnan(values)
    params, _ = curve_fit(log_model, days[mask], values[mask])
    save_model_params(params)
    return {"params": params}

# --- SESSION STATE ---
if 'learned_data' not in st.session_state:
    saved_params = load_model_params()
    st.session_state.learned_data = {"params": saved_params} if saved_params is not None else None

if 'cohort_configs' not in st.session_state:
    st.session_state.cohort_configs = {}

# --- SIDEBAR ---
st.sidebar.header("Step 1: Learn Curve")
history_file = st.sidebar.file_uploader("Upload Cumulative History CSV", type="csv")
if history_file:
    hist_df = pd.read_csv(history_file)
    learned_result = get_learned_curve(hist_df)
    if learned_result:
        st.session_state.learned_data = learned_result
        st.sidebar.success("Model Learned!")

if st.session_state.learned_data:
    if st.sidebar.button("Reset Everything"):
        if os.path.exists(MODEL_FILE): os.remove(MODEL_FILE)
        st.session_state.learned_data = None
        st.session_state.cohort_configs = {}
        st.rerun()

# --- MAIN PANEL ---
st.header("Step 2: Predict & Calibrate")
predict_file = st.file_uploader("Upload Adjust Report (CSV)", type="csv")

if predict_file and st.session_state.learned_data:
    new_df = pd.read_csv(predict_file)
    new_roas_cols = [c for c in new_df.columns if c.startswith('roas_d') and c[6:].isdigit()]
    new_roas_cols.sort(key=lambda x: int(x[6:]))
    orig_a, orig_b = st.session_state.learned_data['params']
    
    # 1. DATA PREP
    plot_data = []
    for _, row in new_df.iterrows():
        days_with_data = np.array([int(c[6:]) for c in new_roas_cols if row[c] > 0])
        if len(days_with_data) < 7: continue
        
        cid = str(row['day'])
        if cid not in st.session_state.cohort_configs:
            st.session_state.cohort_configs[cid] = {"slope": 1.0, "tail": 1.0}
        
        def calc_churn(h, l): return f"{round((1-h/l)*100, 2)}%" if l > 0 else "N/A"

        plot_data.append({
            "day": row['day'], "spend": row.get('network_cost', 0),
            "days_with_data": days_with_data, "actual_series": np.array([row[f'roas_d{d}'] for d in days_with_data]),
            "churn": [calc_churn(row.get('retention_rate_d3',0), row.get('retention_rate_d1',0)),
                      calc_churn(row.get('retention_rate_d7',0), row.get('retention_rate_d3',0)),
                      calc_churn(row.get('retention_rate_d14',0), row.get('retention_rate_d7',0))]
        })

    # 2. CALIBRATION & CHART (Grouped for UX)
    st.write("---")
    st.subheader("🔍 Cohort Deep Dive & Live Calibration")
    
    if plot_data:
        # Layout: Control Column | Graph Column
        col_ctrl, col_viz = st.columns([1, 2], gap="large")
        
        with col_ctrl:
            selected_cohort = st.selectbox("Select Cohort", [d['day'] for d in plot_data])
            cid = str(selected_cohort)
            c_data = next(d for d in plot_data if d['day'] == selected_cohort)
            
            st.write("**Adjust Curve Shape**")
            st.session_state.cohort_configs[cid]['slope'] = st.slider(
                "Early Velocity", 0.5, 1.5, st.session_state.cohort_configs[cid]['slope'], key=f"s_{cid}"
            )
            st.session_state.cohort_configs[cid]['tail'] = st.slider(
                "LTV Tail", 0.5, 2.0, st.session_state.cohort_configs[cid]['tail'], key=f"t_{cid}"
            )
            
            # Calibration Logic
            cfg = st.session_state.cohort_configs[cid]
            l_day = c_data["days_with_data"][-1]
            l_val = c_data["actual_series"][-1]
            pivot_mul = l_val / flexible_model(l_day, orig_a, orig_b, cfg['slope'], cfg['tail'])
            
            def p_func(t): return pivot_mul * flexible_model(t, orig_a, orig_b, cfg['slope'], cfg['tail'])
            
            st.write("---")
            st.metric("D180 Prediction", f"{round(p_func(180)*100, 1)}%")
            st.metric("D365 Prediction", f"{round(p_func(365)*100, 1)}%")
            try:
                be_idx = np.where(p_func(np.arange(0, 1001)) >= 1.0)[0]
                be_val = f"Day {be_idx[0]}" if len(be_idx) > 0 else ">1000"
            except: be_val = ">1000"
            st.metric("Estimated Break-Even", be_val)

        with col_viz:
            fig = go.Figure()
            days_range = np.arange(0, 366)
            fig.add_trace(go.Scatter(x=days_range, y=[p_func(d) for d in days_range], 
                                     name='Predicted', line=dict(color='#FF4B4B', width=3, dash='dash'),
                                     hovertemplate='Day %{x}<br>ROAS: %{y:.2%}<extra></extra>'))
            fig.add_trace(go.Scatter(x=c_data["days_with_data"], y=c_data["actual_series"], 
                                     name='Actual', mode='markers', marker=dict(size=10, color='#0068C9'),
                                     hovertemplate='Day %{x}<br>ROAS: %{y:.2%}<extra></extra>'))
            fig.update_layout(height=450, hovermode="x unified", template="plotly_white", margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # 3. MASTER TABLE (Synced)
    st.write("---")
    st.subheader("📊 Master Summary Table (Synced with Calibration)")
    summary = []
    for d in plot_data:
        cid_loop = str(d['day'])
        cfg_loop = st.session_state.cohort_configs[cid_loop]
        # Multiplier per row
        m_loop = d["actual_series"][-1] / flexible_model(d["days_with_data"][-1], orig_a, orig_b, cfg_loop['slope'], cfg_loop['tail'])
        def p_loop(t): return m_loop * flexible_model(t, orig_a, orig_b, cfg_loop['slope'], cfg_loop['tail'])
        
        try:
            be_idx = np.where(p_loop(np.arange(0, 1001)) >= 1.0)[0]
            be_str = str(be_idx[0]) if len(be_idx) > 0 else ">1000"
        except: be_str = ">1000"

        summary.append({
            "Cohort Day": d["day"], "Spend": round(d["spend"], 2),
            "Actual ROAS": round(d["actual_series"][-1], 4),
            "Pred D180": round(p_loop(180), 4), "Pred D365": round(p_loop(365), 4),
            "Break-Even Day": be_str, "Slope": f"{int(cfg_loop['slope']*100)}%", "Tail": f"{cfg_loop['tail']}x",
            "Churn D3/D1": d["churn"][0], "Churn D7/D3": d["churn"][1], "Churn D14/D7": d["churn"][2]
        })
    st.dataframe(pd.DataFrame(summary), use_container_width=True)

    csv = pd.DataFrame(summary).to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Report", csv, "roas_sync_report.csv", "text/csv")
