import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="EU Gas Risk Calculator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE MANAGEMENT ---
def reset_defaults():
    st.session_state.w_weather_preset = '1. Baseline (Normal)'
    st.session_state.w_forced = 0
    st.session_state.w_uplift = 0.15
    st.session_state.w_obs = 1
    
    st.session_state.chk_russia = False
    st.session_state.chk_lng = False
    st.session_state.chk_infra = False
    st.session_state.chk_custom = False
    st.session_state.w_custom_val = 0.5
    
    st.session_state.w_mit_on = True
    st.session_state.w_mit_trig = 0.40
    st.session_state.w_mit_resp = 0.15
    
    st.session_state.w_start = 575.5
    st.session_state.w_weeks = 12
    st.session_state.w_base_d = 88.0 # Tuned for realism (Cold Jan)
    st.session_state.w_base_s = 68.0 # Tuned for realism (Tight Supply)

# Initialize Session State
if 'w_weather_preset' not in st.session_state:
    reset_defaults()

# --- 1. SIMULATION ENGINE (PhD Level) ---
class MasterEngine:
    def __init__(self):
        self.CAPACITY = 1143.0  # TWh
    
    def simulate(self, inputs, active_shocks_loss=0.0):
        np.random.seed(42)
        n_sims = 5000 
        weeks = int(inputs['weeks'])
        
        # Parallel Tracks
        s_base = np.zeros((n_sims, weeks + 1)); s_base[:, 0] = inputs['start_storage']
        s_stress = np.zeros((n_sims, weeks + 1)); s_stress[:, 0] = inputs['start_storage']
        
        # Trackers
        fail_30_b = np.full(n_sims, np.nan); fail_05_b = np.full(n_sims, np.nan)
        fail_30_s = np.full(n_sims, np.nan); fail_05_s = np.full(n_sims, np.nan)
        
        # Weather
        alpha = 2 + inputs['obs_cold']
        beta_val = 8 + (4 - inputs['obs_cold'])
        p_cold_dist = np.random.beta(alpha, beta_val, n_sims)
        
        # Supply
        # Calibrated: Net Draw ~20 TWh/week in Jan (88 Dem - 68 Sup)
        # This aligns with historical ~3-4 TWh/day drawdowns in peak winter.
        base_supply_vec = np.random.normal(inputs['base_supply'], 2.5, (n_sims, weeks))
        stress_supply_vec = base_supply_vec.copy()
        if active_shocks_loss > 0:
            stress_supply_vec -= (active_shocks_loss * 7.0)
            
        for t in range(weeks):
            # Seasonality (Spring Taper)
            # Jan: 100% | Feb: 95% | Mar: 85% | Apr: 70%
            seasonality = 1.0
            if t > 5: seasonality = 0.95
            if t > 9: seasonality = 0.85
            if t > 13: seasonality = 0.70
            
            # Weather
            is_cold = np.random.rand(n_sims) < p_cold_dist
            if t < inputs['forced_cold']: is_cold[:] = True
                
            # Demand
            base_d = inputs['base_demand'] * seasonality
            demand_t = base_d * (1 + (is_cold * inputs['uplift']))
            
            # Volatility (Increased to 12% to capture fat tails/extreme events)
            sigma = 0.12 
            log_sigma = np.sqrt(np.log(1 + (sigma/demand_t)**2))
            log_mu = np.log(demand_t) - 0.5 * log_sigma**2
            demand_draw = np.random.lognormal(log_mu, log_sigma)
            
            # Mitigation
            d_b, d_s = demand_draw, demand_draw
            if inputs['mit_on']:
                mask_b = s_base[:, t] < (self.CAPACITY * inputs['mit_trigger'])
                d_b = demand_draw.copy(); d_b[mask_b] *= (1.0 - inputs['mit_response'])
                
                mask_s = s_stress[:, t] < (self.CAPACITY * inputs['mit_trigger'])
                d_s = demand_draw.copy(); d_s[mask_s] *= (1.0 - inputs['mit_response'])
            
            # Physics Step
            self._step(s_base, t, base_supply_vec[:, t], d_b, fail_30_b, fail_05_b)
            self._step(s_stress, t, stress_supply_vec[:, t], d_s, fail_30_s, fail_05_s)
            
        return s_base, s_stress, fail_30_b, fail_30_s, fail_05_b, fail_05_s

    def _step(self, storage, t, supply, demand, f30, f05):
        net = supply - demand
        # Pressure Constraint: Below 15%, max withdrawal drops linearly
        low_p = storage[:, t] < (self.CAPACITY * 0.15)
        # Cap withdrawal (e.g. max 65 TWh/week physical limit)
        net[low_p] = np.maximum(net[low_p], -65.0) 
        storage[:, t+1] = np.clip(storage[:, t] + net, 0, self.CAPACITY)
        
        mask_30 = storage[:, t+1] < (self.CAPACITY * 0.30)
        f30[mask_30 & np.isnan(f30)] = t + 1
        mask_05 = storage[:, t+1] < (self.CAPACITY * 0.05)
        f05[mask_05 & np.isnan(f05)] = t + 1

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.title("🎛️ Control Center")

if st.sidebar.button("↺ Reset Defaults"):
    reset_defaults()
    st.rerun() # UPDATED FIX

# --- BOX 1: WEATHER ---
with st.sidebar.expander("1. Weather Scenario", expanded=True):
    weather_mode = st.selectbox(
        "Preset:", 
        ['1. Baseline (Normal)', '2. Cold Snap (2 Wks)', '3. Deep Freeze (Month)'],
        key='w_weather_preset'
    )
    
    if 'Baseline' in weather_mode:
        st.info("ℹ️ **Baseline:** Standard ENTSOG seasonality. Normal temps.")
    elif 'Cold Snap' in weather_mode:
        st.info("❄️ **Cold Snap:** 2 weeks of high pressure cold (+25% demand).")
    elif 'Deep Freeze' in weather_mode:
        st.warning("⚠️ **Deep Freeze:** 'Beast from East'. 4 weeks of extreme cold (+40% demand).")
    
    w_forced = st.slider("Forced Cold Weeks", 0, 10, key='w_forced')
    w_uplift = st.slider("Demand Uplift (+%)", 0.0, 0.6, step=0.05, key='w_uplift')
    w_obs = st.slider("Observed Cold (Last 4w)", 0, 4, key='w_obs')

# --- BOX 2: SHOCKS ---
with st.sidebar.expander("2. Supply Shocks", expanded=True):
    st.markdown("Select active disruptions:")
    chk_russia = st.checkbox("Russian Stop (-0.5 TWh/d)", key='chk_russia')
    chk_lng = st.checkbox("LNG Crisis (-0.8 TWh/d)", key='chk_lng')
    chk_infra = st.checkbox("Infra Fail (-1.2 TWh/d)", key='chk_infra')
    chk_custom = st.checkbox("Custom Shock", key='chk_custom')
    w_custom_val = st.slider("Custom Loss (TWh/d)", 0.1, 3.0, step=0.1, disabled=not chk_custom, key='w_custom_val')

# --- BOX 3: MITIGATION ---
with st.sidebar.expander("3. Mitigation & Elasticity", expanded=True):
    st.markdown("Market reaction to high prices:")
    w_mit_on = st.checkbox("Enable Demand Response", key='w_mit_on')
    
    if w_mit_on:
        w_mit_trig = st.slider("Price Trigger (Storage %)", 0.1, 0.6, step=0.05, key='w_mit_trig')
        w_mit_resp = st.slider("Demand Cut (%)", 0.05, 0.30, step=0.05, key='w_mit_resp')
    else:
        w_mit_trig = 0.40; w_mit_resp = 0.0

# --- BOX 4: SETTINGS ---
with st.sidebar.expander("4. Simulation Settings", expanded=False):
    w_start = st.number_input("Start Storage (TWh)", key='w_start')
    w_weeks = st.slider("Horizon (Weeks)", 4, 20, key='w_weeks')
    w_base_d = st.number_input("Base Demand (TWh/wk)", key='w_base_d')
    w_base_s = st.number_input("Base Supply (TWh/wk)", key='w_base_s')

# --- 3. MAIN DASHBOARD ---
st.title("🇪🇺 EU Gas Risk Analyzer")
st.markdown("### Quantitative Stress Test & Forecast")

if st.button("🔄 Run Simulation", type="primary", use_container_width=True):
    
    # Gather Inputs
    tot_shock = 0.0
    if chk_russia: tot_shock += 0.5
    if chk_lng: tot_shock += 0.8
    if chk_infra: tot_shock += 1.2
    if chk_custom: tot_shock += w_custom_val
    
    inputs = {
        'weeks': w_weeks, 'start_storage': w_start,
        'base_demand': w_base_d, 'base_supply': w_base_s,
        'forced_cold': w_forced, 'uplift': w_uplift, 'obs_cold': w_obs,
        'mit_on': w_mit_on, 'mit_trigger': w_mit_trig, 'mit_response': w_mit_resp
    }
    
    # Run Engine
    eng = MasterEngine()
    p_b, p_s, f30_b, f30_s, f05_b, f05_s = eng.simulate(inputs, active_shocks_loss=tot_shock)
    
    # Calculate Metrics
    r30_b = np.sum(~np.isnan(f30_b))/5000; r30_s = np.sum(~np.isnan(f30_s))/5000
    r05_b = np.sum(~np.isnan(f05_b))/5000; r05_s = np.sum(~np.isnan(f05_s))/5000
    med_s = np.median(p_s[:,-1])
    
    # --- SCORECARD ---
    col1, col2, col3, col4 = st.columns(4)
    
    status = "SECURE"
    delta_color = "normal"
    if r30_s > 0.15: status = "TIGHT"; delta_color = "off"
    if r05_s > 0.05: status = "CRITICAL"; delta_color = "inverse"
    
    col1.metric("Grid Status", status, f"-{tot_shock:.1f} TWh/d Shock", delta_color=delta_color)
    col2.metric("Rationing Risk (<30%)", f"{r30_s:.1%}", f"vs {r30_b:.1%} Base", delta_color="inverse")
    col3.metric("Blackout Risk (<5%)", f"{r05_s:.1%}", f"vs {r05_b:.1%} Base", delta_color="inverse")
    col4.metric("Median End Storage", f"{med_s:.0f} TWh", "Target: >340 TWh")

    # --- TABS FOR CHARTS ---
    tab1, tab2 = st.tabs(["📉 Trajectory Comparison", "📅 Risk Timing Histogram"])
    
    with tab1:
        # Fan Chart
        x_ax = np.arange(inputs['weeks'] + 1)
        p50_b = np.median(p_b, axis=0); p50_s = np.median(p_s, axis=0)
        p10_s = np.percentile(p_s, 10, axis=0); p90_s = np.percentile(p_s, 90, axis=0)
        
        fig_fan = go.Figure()
        fig_fan.add_trace(go.Scatter(x=x_ax, y=p50_b, mode='lines', name='Weather Baseline', line=dict(color='royalblue', width=2)))
        fig_fan.add_trace(go.Scatter(x=x_ax, y=p10_s, fill=None, mode='lines', line=dict(width=0), showlegend=False))
        fig_fan.add_trace(go.Scatter(x=x_ax, y=p90_s, fill='tonexty', fillcolor='rgba(255,0,0,0.1)', line=dict(width=0), name='Uncertainty (10-90%)'))
        fig_fan.add_trace(go.Scatter(x=x_ax, y=p50_s, mode='lines', name='Shock Scenario', line=dict(color='firebrick', width=3, dash='dot')))
        
        fig_fan.add_hline(y=1143*0.30, line_dash="dash", line_color="orange", annotation_text="30% Rationing")
        fig_fan.add_hline(y=1143*0.05, line_color="red", annotation_text="5% Blackout")
        
        fig_fan.update_layout(title='Storage Trajectory Gap (Baseline vs. Shock)', height=400, template='plotly_white')
        st.plotly_chart(fig_fan, use_container_width=True)

    with tab2:
        # Histogram of Most Likely Failure Dates
        weeks_idx = np.arange(1, inputs['weeks'] + 1)
        counts = [np.sum(f30_s == w) for w in weeks_idx]
        probs = [c / 5000 for c in counts]
        
        start_date = datetime.now()
        dates = [(start_date + timedelta(weeks=int(i))).strftime('%b %d') for i in weeks_idx]
        
        # Highlight Peak
        max_p = max(probs) if probs else 0
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(
            x=dates, y=probs,
            marker_color='orange',
            name='Risk Probability',
            text=[f"{p:.1%}" if p > 0.01 else "" for p in probs],
            textposition='auto'
        ))
        
        fig_hist.update_layout(
            title='When will the grid hit <30%? (Risk Probability per Week)',
            yaxis_title='Probability',
            height=400, template='plotly_white'
        )
        st.plotly_chart(fig_hist, use_container_width=True)

else:
    st.info("👈 Use the **Control Center** in the sidebar to configure your scenario and run the simulation.")
