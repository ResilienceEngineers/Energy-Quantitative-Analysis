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

# --- 1. SIMULATION ENGINE ---
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
        base_supply_vec = np.random.normal(inputs['base_supply'], 2.5, (n_sims, weeks))
        stress_supply_vec = base_supply_vec.copy()
        if active_shocks_loss > 0:
            stress_supply_vec -= (active_shocks_loss * 7.0)
            
        for t in range(weeks):
            # Seasonality
            seasonality = 1.0
            if t > 5: seasonality = 0.90
            if t > 9: seasonality = 0.80
            
            # Weather
            is_cold = np.random.rand(n_sims) < p_cold_dist
            if t < inputs['forced_cold']: is_cold[:] = True
                
            # Demand
            base_d = inputs['base_demand'] * seasonality
            demand_t = base_d * (1 + (is_cold * inputs['uplift']))
            sigma = 0.08
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
        low_p = storage[:, t] < (self.CAPACITY * 0.15)
        net[low_p] = np.maximum(net[low_p], -65.0) 
        storage[:, t+1] = np.clip(storage[:, t] + net, 0, self.CAPACITY)
        
        mask_30 = storage[:, t+1] < (self.CAPACITY * 0.30)
        f30[mask_30 & np.isnan(f30)] = t + 1
        mask_05 = storage[:, t+1] < (self.CAPACITY * 0.05)
        f05[mask_05 & np.isnan(f05)] = t + 1

# --- 2. LOGIC FOR PRESET UPDATES ---
def update_weather_settings():
    """Callback to snap sliders to the selected preset."""
    mode = st.session_state.weather_preset
    if 'Baseline' in mode:
        st.session_state.w_forced = 0
        st.session_state.w_uplift = 0.15
        st.session_state.w_obs = 1
    elif 'Cold Snap' in mode:
        st.session_state.w_forced = 2
        st.session_state.w_uplift = 0.25
        st.session_state.w_obs = 3
    elif 'Deep Freeze' in mode:
        st.session_state.w_forced = 4
        st.session_state.w_uplift = 0.40
        st.session_state.w_obs = 4

def reset_all():
    """Reset everything to factory defaults."""
    st.session_state.weather_preset = '1. Baseline (Normal)'
    update_weather_settings() # Apply baseline values
    st.session_state.chk_russia = False
    st.session_state.chk_lng = False
    st.session_state.chk_infra = False
    st.session_state.chk_custom = False
    st.session_state.w_mit_on = True
    st.session_state.w_start = 575.5

# Initialize Session State Variables if they don't exist
if 'w_forced' not in st.session_state:
    st.session_state.w_forced = 0
if 'w_uplift' not in st.session_state:
    st.session_state.w_uplift = 0.15
if 'w_obs' not in st.session_state:
    st.session_state.w_obs = 1

# --- 3. SIDEBAR CONTROL CENTER ---
st.sidebar.title("🎛️ Control Center")

if st.sidebar.button("↺ Reset All Defaults", on_click=reset_all):
    pass # The callback handles the reset

# --- BOX 1: WEATHER ---
with st.sidebar.expander("1. Weather Scenario", expanded=True):
    # The Selectbox now triggers the callback 'update_weather_settings'
    weather_mode = st.selectbox(
        "Select Scenario:", 
        ['1. Baseline (Normal)', '2. Cold Snap (2 Wks)', '3. Deep Freeze (Month)'],
        key='weather_preset',
        on_change=update_weather_settings
    )
    
    # Contextual Help Text
    if 'Baseline' in weather_mode:
        st.info("ℹ️ **Baseline:** Standard seasonality. No extreme cold events.")
    elif 'Cold Snap' in weather_mode:
        st.info("❄️ **Cold Snap:** 2 weeks of high pressure cold (+25% demand).")
    elif 'Deep Freeze' in weather_mode:
        st.warning("⚠️ **Deep Freeze:** 'Beast from East'. 4 weeks of extreme cold (+40% demand).")
    
    st.markdown("---")
    st.markdown("**Detailed Variables:**")
    
    # Sliders are bound to session_state keys
    st.slider("Forced Cold Weeks", 0, 10, key='w_forced')
    st.slider("Demand Uplift (+%)", 0.0, 0.6, step=0.05, key='w_uplift')
    st.slider("Observed Cold (Last 4w)", 0, 4, key='w_obs')

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
    w_mit_on = st.checkbox("Enable Demand Response", value=True, key='w_mit_on')
    
    if w_mit_on:
        st.success("✅ **Active:** Industry cuts demand if prices spike.")
        w_mit_trig = st.slider("Trigger (Storage %)", 0.1, 0.6, 0.40, 0.05, key='w_mit_trig')
        w_mit_resp = st.slider("Demand Cut (%)", 0.05, 0.30, 0.15, 0.05, key='w_mit_resp')
    else:
        st.error("❌ **Inactive:** Demand is inelastic.")
        w_mit_trig = 0.40
        w_mit_resp = 0.0

# --- BOX 4: SETTINGS ---
with st.sidebar.expander("4. Simulation Settings"):
    w_start = st.number_input("Start Storage (TWh)", value=575.5, key='w_start')
    w_weeks = st.slider("Horizon (Weeks)", 4, 20, 12, key='w_weeks')
    w_base_d = st.number_input("Base Demand (TWh)", value=85.0, key='w_base_d')
    w_base_s = st.number_input("Base Supply (TWh)", value=70.0, key='w_base_s')

# --- 4. MAIN DASHBOARD ---
st.title("🇪🇺 EU Gas Risk Analyzer")
st.markdown("### Executive Risk Dashboard")

if st.button("🔄 Update Analysis", type="primary", use_container_width=True):
    
    # Inputs
    tot_shock = 0.0
    if chk_russia: tot_shock += 0.5
    if chk_lng: tot_shock += 0.8
    if chk_infra: tot_shock += 1.2
    if chk_custom: tot_shock += w_custom_val
    
    inputs = {
        'weeks': w_weeks, 'start_storage': w_start,
        'base_demand': w_base_d, 'base_supply': w_base_s,
        'forced_cold': st.session_state.w_forced, 
        'uplift': st.session_state.w_uplift, 
        'obs_cold': st.session_state.w_obs,
        'mit_on': w_mit_on, 'mit_trigger': w_mit_trig, 'mit_response': w_mit_resp
    }
    
    # Engine
    eng = MasterEngine()
    p_b, p_s, f30_b, f30_s, f05_b, f05_s = eng.simulate(inputs, active_shocks_loss=tot_shock)
    
    # Metrics
    r30_b = np.sum(~np.isnan(f30_b))/5000; r30_s = np.sum(~np.isnan(f30_s))/5000
    r05_b = np.sum(~np.isnan(f05_b))/5000; r05_s = np.sum(~np.isnan(f05_s))/5000
    med_b = np.median(p_b[:,-1]); med_s = np.median(p_s[:,-1])
    
    # --- SCORECARD ---
    col1, col2, col3, col4 = st.columns(4)
    
    status = "SECURE"
    delta_color = "normal"
    if r30_s > 0.15: status = "TIGHT"; delta_color = "off"
    if r05_s > 0.05: status = "CRITICAL"; delta_color = "inverse"
    
    col1.metric("Grid Status", status, f"-{tot_shock:.1f} TWh/d Shock", delta_color=delta_color)
    col2.metric("Rationing Risk (<30%)", f"{r30_s:.1%}", f"vs {r30_b:.1%} Base", delta_color="inverse")
    col3.metric("Blackout Risk (<5%)", f"{r05_s:.1%}", f"vs {r05_b:.1%} Base", delta_color="inverse")
    col4.metric("Median Storage", f"{med_s:.0f} TWh", f"Gap: -{med_b - med_s:.0f} TWh", delta_color="inverse")

    # --- CHARTS ---
    tab_chart1, tab_chart2 = st.tabs(["📉 Trajectory Comparison", "📅 Risk Timing Histogram"])
    
    with tab_chart1:
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
        fig_fan.update_layout(title='Storage Trajectory Gap (Baseline vs. Shock)', height=400, template='plotly_white', hovermode="x unified")
        st.plotly_chart(fig_fan, use_container_width=True)

    with tab_chart2:
        weeks_idx = np.arange(1, inputs['weeks'] + 1)
        counts = [np.sum(f30_s == w) for w in weeks_idx]
        probs = [c / 5000 for c in counts]
        dates = [(datetime.now() + timedelta(weeks=int(i))).strftime('%b %d') for i in weeks_idx]
        
        fig_hist = go.Figure(go.Bar(x=dates, y=probs, marker_color='orange', name='Risk'))
        fig_hist.update_layout(title='When will we hit 30%?', height=400, template='plotly_white')
        st.plotly_chart(fig_hist, use_container_width=True)

else:
    st.info("👈 **Start here:** Adjust the Scenario in the sidebar and click **Update Analysis**.")
