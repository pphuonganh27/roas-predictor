import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import json
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="ROAS Predictor: Full Year (D365) Projection", layout="wide")

st.title("📈 ROAS Curve Learner & Predictor (D365)")
st.markdown("Designed for **Mid-term Hybrid Hypercasual Puzzle** games with projections up to one year.")

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

# --- HELPER FUNCTIONS ---
def log_model(t, a, b):
    # ROAS = a * ln(t + 1) + b
    return a * np.log(t + 1) + b

def get_learned_curve(df):
    # Requirement: Only learn cohorts having roas_d180
    if 'roas_d180' not in df.columns:
        st.error("Historical file must contain 'roas_d180' column.")
        return None
    
    # Requirement: Ignore cohorts where roas_d180 = 0 (data hasn't arrived/no spend)
    df_filtered = df[df['roas_d180'] > 0].copy()
    
    # Identify available ROAS columns from d0 to d180
    roas_cols = [f'roas_d{i}' for i in range(181) if f'roas_d{i}' in df.columns]
    
    # Calculate average while skipping 0s (no advertising/data hasn't arrived)
    mean_curve = df_filtered[roas_cols].replace(0, np.nan).mean()
    
    days = np.array([int(c.split('_d')[1]) for c in mean_curve.index])
    values = mean_curve.values
    
    # Fit Logarithmic Model to establish the "Historical Skeleton"
    mask = ~np.isnan(values)
    params, _ = curve_fit(log_model, days[mask], values[mask])
    
    save_model_params(params)
    return {"params": params, "days": days, "actual_means": values}

# --- INITIALIZE SESSION STATE ---
if 'learned_data' not in st.session_state:
    saved_params = load_model_params()
    if saved_params is not None:
        st.session_state.learned_data = {"params": saved_params}
    else:
        st.session_state.learned_data = None

# --- SIDEBAR: HISTORICAL DATA ---
st.sidebar.header("Step 1: Learn Curve")
history_file = st.sidebar.file_uploader("Upload Cumulative History CSV to (Re)Learn", type="csv")

if history_file:
    hist_df = pd.read_csv(history_file)
    learned_result = get_learned_curve(hist_df)
    if learned_result:
        st.session_state.learned_data = learned_result
        st.sidebar.success("New ROAS Curve Learned and Saved!")

if st.session_state.learned_data:
    st.sidebar.info("✅ Model Active: Projecting to D365 using historical baseline.")
    if st.sidebar.button("Reset/Forget Model"):
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
        st.session_state.learned_data = None
        st.rerun()
else:
    st.sidebar.warning("⚠️ No model found. Please upload historical data.")

# --- MAIN PANEL: PREDICTIONS ---
st.header("Step 2: Predict New Cohorts")
predict_file = st.file_uploader("Upload New Adjust Report (CSV)", type="csv")

learned_data = st.session_state.learned_data

if predict_file and learned_data:
    new_df = pd.read_csv(predict_file)
    new_roas_cols = [c for c in new_df.columns if c.startswith('roas_d') and c[6:].isdigit()]
    new_roas_cols.sort(key=lambda x: int(x[6:]))
    
    results = []
    plot_data = [] 

    for _, row in new_df.iterrows():
        # Identify valid data days
        days_with_data = np.array([int(c[6:]) for c in new_roas_cols if row[c] > 0])
        actual_vals = np.array([row[f'roas_d{d}'] for d in days_with_data])
        
        # Calculate Churn Rates
        def calc_churn(r_high, r_low):
            if r_low > 0:
                return f"{round((1 - r_high / r_low) * 100, 2)}%"
            return "N/A"

        c3_1 = calc_churn(row.get('retention_rate_d3', 0), row.get('retention_rate_d1', 0))
        c7_3 = calc_churn(row.get('retention_rate_d7', 0), row.get('retention_rate_d3', 0))
        c14_7 = calc_churn(row.get('retention_rate_d14', 0), row.get('retention_rate_d7', 0))

        # Rule: Only predict cohorts having at least 7 days of ROAS
        if len(days_with_data) < 7:
            results.append({
                "Cohort Day": row['day'],
                "Latest Data Day": max(days_with_data) if len(days_with_data) > 0 else 0,
                "Break-Even Day": "N/A",
                "Churn D3/D1": c3_1,
                "Churn D7/D3": c7_3,
                "Churn D14/D7": c14_7,
                "Status": "N/A (<7 days)"
            })
            continue
            
        latest_day = days_with_data[-1]
        latest_val = actual_vals[-1]
        
        # Auto-Scaling Factor (Best fit anchor across all points)
        model_vals_at_days = log_model(days_with_data, *learned_data['params'])
        auto_scaling_factor = np.mean(actual_vals / model_vals_at_days)
        
        def predict_auto(d):
            return log_model(d, *learned_data['params']) * auto_scaling_factor
        
        # Automatic Break-Even Calculation
        a, b = learned_data['params']
        try:
            target_ln = (1.0 / auto_scaling_factor - b) / a
            be_day = int(round(np.exp(target_ln) - 1))
            be_display = str(max(0, be_day)) if be_day < 1000 else ">365"
        except:
            be_display = ">365"

        results.append({
            "Cohort Day": row['day'],
            "Latest Data Day": latest_day,
            "Actual ROAS": round(latest_val, 4),
            "Pred D30": round(predict_auto(30), 4),
            "Pred D180": round(predict_auto(180), 4),
            "Pred D365": round(predict_auto(365), 4),
            "Break-Even Day": be_display,
            "Churn D3/D1": c3_1,
            "Churn D7/D3": c7_3,
            "Churn D14/D7": c14_7,
            "Status": "Predicted"
        })

        plot_data.append({
            "day": row['day'],
            "days_with_data": days_with_data,
            "actual_series": actual_vals,
            "auto_scaling_factor": auto_scaling_factor,
            "Status": "Predicted"
        })

    res_df = pd.DataFrame(results)
    st.write("### Prediction Results (Full Year Summary)")
    st.dataframe(res_df.style.highlight_max(axis=0, subset=['Pred D365']))
    
    # --- INTERACTIVE DEEP DIVE & MANUAL CALIBRATION ---
    st.write("---")
    st.header("🔍 Cohort Deep Dive: Slope & Year-Long Projection")
    
    available_cohorts = [d['day'] for d in plot_data if d.get("Status") == "Predicted"]
    
    if available_cohorts:
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_cohort = st.selectbox("Select a cohort to calibrate", available_cohorts)
            c_data = next(item for item in plot_data if item["day"] == selected_cohort)
            
            st.markdown("### 🛠 Adjust Growth Slope")
            st.caption("Manually tweak the growth rate to match current product/marketing behavior.")
            
            # Slope Adjustment Slider
            slope_adj = st.slider("Slope Intensity (%)", 50, 150, 100, key=f"slope_{selected_cohort}") / 100.0
            
            # Calibration Logic (Pivot around latest actual data point)
            orig_a, orig_b = learned_data['params']
            new_a = orig_a * slope_adj
            latest_day = c_data["days_with_data"][-1]
            latest_actual = c_data["actual_series"][-1]
            # C = Actual / (Adjusted_A * ln(t+1) + B)
            pivot_multiplier = latest_actual / (new_a * np.log(latest_day + 1) + orig_b)

            def predict_calibrated(t):
                return pivot_multiplier * (new_a * np.log(t + 1) + orig_b)

            # Display calibrated metrics
            st.metric("Calibrated D180", f"{round(predict_calibrated(180)*100, 2)}%")
            st.metric("Calibrated D365", f"{round(predict_calibrated(365)*100, 2)}%")
            
            try:
                target_ln_cal = (1.0/pivot_multiplier - orig_b) / new_a
                be_day_cal = int(round(np.exp(target_ln_cal) - 1))
                be_display_cal = str(max(0, be_day_cal)) if be_day_cal < 1000 else ">365"
            except: be_display_cal = ">365"
            st.metric("Calibrated Break-Even", f"Day {be_display_cal}")

        with col2:
            fig = go.Figure()
            days_range = np.arange(0, 366) # Extend graph to 1 year
            
            # Predicted Calibrated Line
            fig.add_trace(go.Scatter(
                x=days_range, 
                y=[predict_calibrated(d) for d in days_range],
                mode='lines',
                name='Calibrated Prediction',
                line=dict(color='red', width=3, dash='dash'),
                hovertemplate='Day %{x}<br>Predicted: %{y:.4f}'
            ))
            
            # Actual Data points
            fig.add_trace(go.Scatter(
                x=c_data["days_with_data"], 
                y=c_data["actual_series"],
                mode='markers',
                name='Actual Performance',
                marker=dict(size=10, color='blue'),
                hovertemplate='Day %{x}<br>Actual: %{y:.4f}'
            ))
            
            # Visual markers
            fig.add_vline(x=180, line_dash="dot", line_color="gray", annotation_text="D180 Baseline")
            
            fig.update_layout(
                title=f"ROAS Year-Long Trend: {selected_cohort}",
                xaxis_title="Day Since Install",
                yaxis_title="Cumulative ROAS",
                hovermode="x unified",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

    csv = res_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions", csv, "roas_predictions_d365.csv", "text/csv")

elif predict_file and not learned_data:
    st.warning("Please upload historical data in the sidebar first to learn the model.")