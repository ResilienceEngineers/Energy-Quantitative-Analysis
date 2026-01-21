import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="EU Gas Risk Calculator (Ph.D. Edition)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. SIMULATION ENGINE (Audited) ---
class PhDEngine:
    def __init__(self):
        self.CAPACITY = 1143.0  # TWh (EU Total)
    
    def simulate(self, inputs, active_shocks_loss=0.0):
        # Deterministic Seed for Reproducibility
        np.random.seed(42)
        n_sims = 10000 
        weeks = int(inputs['weeks'])
        
        # Initialize Storage Arrays (Start at User Defined Level)
        s_base = np.zeros((n_sims, weeks + 1)); s_base[:, 0] = inputs['start_storage']
        s_stress = np.zeros((n_sims, weeks + 1)); s_stress[:, 0] = inputs['start_storage']
        
        # Failure Trackers (NaN = No Failure)
        fail_30_s = np.full(n_sims, np.nan) # First week < 30%
        fail_05_s = np.full(n_sims, np.nan) # First week < 5%
        
        # 1. Weather Generator (Bayesian Posterior)
        # Prior (Alpha=2, Beta=8) -> 20% historic probability of cold week
        alpha = 2 + inputs['obs_cold']
        beta_val = 8 + (4 - inputs['obs_cold'])
        p_cold_dist = np.random.beta(alpha, beta_val, n_sims)
        
        # 2. Supply Vectors (Vectorized Normal Distribution)
        # Supply Volatility is lower than demand (ships/pipes are steadier than weather)
        # Std Dev = 2.0 TWh/week
        base_supply_vec = np.random.normal(inputs['base_supply'], 2.0, (n_sims, weeks))
        
        # Apply Shocks to Stress Vector
        stress_supply_vec = base_supply_vec.copy()
        if active_shocks_loss > 0:
            # Convert Daily Loss to Weekly
            stress_supply_vec -= (active_shocks_loss * 7.0)
            
        # --- WEEKLY TIME STEP LOOP ---
        for t in range(weeks):
            # A. Seasonality Curve (The "Spring Taper")
            # Demand naturally decays as sun angle increases (Feb -> Mar -> Apr)
            seasonality = 1.0
            if t > 4: seasonality = 0.96  # Late Feb
            if t > 8: seasonality = 0.88  # Early Mar
            if t > 12: seasonality = 0.75 # Late Mar
            
            # B. Weather State (Bernoulli Trial)
            is_cold = np.random.rand(n_sims) < p_cold_dist
            if t < inputs['forced_cold']: 
                is_cold[:] = True # Force Cold for N weeks
                
            # C. Demand Generation (Lognormal)
            # Base Demand adjusted for Seasonality
            base_d_t = inputs['base_demand'] * seasonality
            
            # Apply Uplift if Cold
            demand_mean = base_d_t * (1 + (is_cold * inputs['uplift']))
            
            # Volatility (Sigma = 7%). 
            # Note: 12% was too high, causing unrealistic tail risks in baseline.
            sigma = 0.07 
            log_sigma = np.sqrt(np.log(1 + (sigma/demand_mean)**2))
            log_mu = np.log(demand_mean) - 0.5 * log_sigma**2
            demand_draw = np.random.lognormal(log_mu, log_sigma)
            
            # D. Mitigation (Elasticity) - ONLY if User Enabled
            d_s = demand_draw.copy() # Baseline demand assumes no panic
            
            if inputs['mit_on']:
                # Trigger: If storage < Trigger %, Price Spike -> Demand Cut
                # We check the storage at the START of the week (t)
                panic_mask = s_stress[:, t] < (self.CAPACITY * inputs['mit_trigger'])
                # Apply reduction (e.g. 15% cut)
                d_s[panic_mask] *= (1.0 - inputs['mit_response'])
            
            # E. Physics & Balance
            # 1. Baseline Simulation
            self._step_physics(s_base, t, base_supply_vec[:, t], demand_draw)
            
            # 2. Stress Simulation (Shocks + Mitigation)
            self._step_physics(s_stress, t, stress_supply_vec[:, t], d_s)
            
            # F. Record Failures (Stress Case)
            # Rationing (<30%)
            mask_30 = s_stress[:, t+1] < (self.CAPACITY * 0.30)
            # Record week index (t+1) if it's the FIRST time falling below
            new_30 = mask_30 & np.isnan(fail_30_s)
            fail_30_s[new_30] = t + 1
            
            # Blackout (<5%)
            mask_05 = s_stress[:, t+1] < (self.CAPACITY * 0.05)
            new_05 = mask_05 & np.isnan(fail_05_s)
            fail_05_s[new_05] = t + 1
            
        return s_base, s_stress, fail_30_s, fail_05_s

    def _step_physics(self, storage, t, supply, demand):
        """Calculates Net Flow with Pressure Constraints"""
        net = supply - demand
        
        # Pressure Ratchet:
        # If storage < 15%, max withdrawal capacity drops linearly.
        # Max physical withdrawal ~ 60-70 TWh/week at full pressure.
        # At 0%, withdrawal is 0.
        
        current_fill_pct = storage[:, t] / self.CAPACITY
        
        # Constraint logic
        unconstrained_withdrawal = 70.0 
        constrained_mask = current_fill_pct < 0.15
        
        # For those in constrained zone, max withdrawal is proportional
        # e.g. at 7.5% fill, max withdrawal is 35 TWh
        max_withdraw = unconstrained_withdrawal * (current_fill_pct / 0.15)
        
        # Apply constraint only if trying to withdraw (net < 0)
        # We take the MAXIMUM (closest to zero) of actual net vs constraint
        # e.g. Want -50, Cap is -30 -> Result -30
        
        net[constrained_mask] = np.maximum(net[constrained_mask], -max_withdraw[constrained_mask])
        
        # Update and Clip
        storage[:, t+1] = np.clip(storage[:, t] + net, 0, self.CAPACITY)

# --- 2. SESSION STATE & CALLBACKS ---
if 'w_forced' not in st.session_state:
    st.session_state.w_forced = 0
if 'w_uplift' not in st.session_state:
    st.session_state.w_uplift = 0.15
if 'w_obs' not in st.session_state:
    st.session_state.w_obs = 1

def update_weather():
    mode = st.session_state.weather_preset
    if 'Baseline' in mode:
        st.session_state.w_forced = 0; st.session_state.w_uplift = 0.15; st.session_state.w_obs = 1
    elif 'Cold Snap' in mode:
        st.session_state.w_forced = 2; st.session_state.w_uplift = 0.25; st.session_state.w_obs = 3
    elif 'Deep Freeze' in mode:
        st.session_state.w_forced = 4; st.session_state.w_uplift = 0.40; st.session_state.w_obs = 4

def reset_all():
    st.session_state.weather_preset = '1. Baseline (Normal)'
    update_weather()
    st.session_state.chk_russia = False
    st.session_state.chk_lng = False
    st.session_state.chk_infra = False
    st.session_state.chk_custom = False
    st.session_state.w_mit_on = False # DISABLED BY DEFAULT
    st.session_state.w_start = 575.5

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.title("🎛️ Control Center")
st.sidebar.button("↺ Reset Defaults", on_click=reset_all)

# A. WEATHER
with st.sidebar.expander("1. Weather Scenario", expanded=True):
    weather_mode = st.selectbox(
        "Select Scenario:", 
        ['1. Baseline (Normal)', '2. Cold Snap (2 Wks)', '3. Deep Freeze (Month)'],
        key='weather_preset', on_change=update_weather
    )
    
    # Description Logic
    if 'Baseline' in weather_mode:
        st.info("Standard seasonality. No extreme events.")
    elif 'Cold Snap' in weather_mode:
        st.info("High pressure. **2 weeks** cold (+25% demand).")
    elif 'Deep Freeze' in weather_mode:
        st.warning("Extreme event. **4 weeks** cold (+40% demand).")
    
    st.markdown("**Manual Variables:**")
    st.slider("Forced Cold Weeks", 0, 10, key='w_forced')
    st.slider("Demand Uplift (+%)", 0.0, 0.6, step=0.05, key='w_uplift')
    st.slider("Observed Cold (Last 4w)", 0, 4, key='w_obs')

# B. SHOCKS
with st.sidebar.expander("2. Supply Shocks", expanded=True):
    st.markdown("⚠️ **Add Disruption:**")
    chk_russia = st.checkbox("Russian Stop (-0.5 TWh/d)", key='chk_russia')
    chk_lng = st.checkbox("LNG Crisis (-0.8 TWh/d)", key='chk_lng')
    chk_infra = st.checkbox("Infra Fail (-1.2 TWh/d)", key='chk_infra')
    chk_custom = st.checkbox("Custom Shock", key='chk_custom')
    w_custom_val = st.slider("Custom Loss (TWh/d)", 0.1, 3.0, step=0.1, disabled=not chk_custom, key='w_custom_val')

# C. MITIGATION (Default OFF)
with st.sidebar.expander("3. Mitigation (Optional)", expanded=True):
    w_mit_on = st.checkbox("Enable Demand Response", value=False, key='w_mit_on')
    
    if w_mit_on:
        st.success("Active: Industry cuts demand if prices spike.")
        w_mit_trig = st.slider("Trigger (Storage %)", 0.1, 0.6, 0.40, 0.05, key='w_mit_trig')
        w_mit_resp = st.slider("Demand Cut (%)", 0.05, 0.30, 0.15, 0.05, key='w_mit_resp')
    else:
        st.markdown("Status: **Inactive** (No elasticity).")
        w_mit_trig = 0.40; w_mit_resp = 0.0

# D. SETTINGS
with st.sidebar.expander("4. Settings"):
    w_start = st.number_input("Start Storage (TWh)", value=575.5, key='w_start')
    w_weeks = st.slider("Horizon (Weeks)", 4, 20, 12, key='w_weeks')
    # Updated Baseline for Validity
    w_base_d = st.number_input("Base Demand (TWh)", value=82.0, key='w_base_d') 
    w_base_s = st.number_input("Base Supply (TWh)", value=70.0, key='w_base_s')

# --- 4. MAIN DASHBOARD ---
st.title("🇪🇺 EU Gas Risk Analyzer")
st.markdown("### Executive Risk Dashboard")

if st.button("🔄 Update Analysis", type="primary", use_container_width=True):
    
    # Logic
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
    
    # Run
    eng = PhDEngine()
    p_b, p_s, f30_s, f05_s = eng.simulate(inputs, active_shocks_loss=tot_shock)
    
    # Metrics
    # Probability = Count of Sims where f30_s is NOT NaN / Total Sims
    r30_s = np.sum(~np.isnan(f30_s))/10000
    r05_s = np.sum(~np.isnan(f05_s))/10000
    
    med_b = np.median(p_b[:,-1])
    med_s = np.median(p_s[:,-1])
    
    # --- SCORECARD ---
    col1, col2, col3, col4 = st.columns(4)
    
    status = "SECURE"
    d_col = "normal"
    if r30_s > 0.15: status = "TIGHT"; d_col = "off"
    if r05_s > 0.05: status = "CRITICAL"; d_col = "inverse"
    
    col1.metric("Grid Status", status, f"-{tot_shock:.1f} TWh/d Shock", delta_color=d_col)
    col2.metric("Rationing Risk (<30%)", f"{r30_s:.1%}", "Probability", delta_color="inverse")
    col3.metric("Blackout Risk (<5%)", f"{r05_s:.1%}", "Probability", delta_color="inverse")
    col4.metric("End Storage (Median)", f"{med_s:.0f} TWh", f"Gap: -{med_b - med_s:.0f} TWh", delta_color="inverse")

    # --- TABS ---
    tab1, tab2 = st.tabs(["📉 Trajectory Comparison", "📅 Risk Timing (Histogram)"])
    
    with tab1:
        x_ax = np.arange(inputs['weeks'] + 1)
        p50_b = np.median(p_b, axis=0); p50_s = np.median(p_s, axis=0)
        p10_s = np.percentile(p_s, 10, axis=0); p90_s = np.percentile(p_s, 90, axis=0)
        
        fig_fan = go.Figure()
        fig_fan.add_trace(go.Scatter(x=x_ax, y=p50_b, mode='lines', name='Weather Baseline', line=dict(color='royalblue', width=2)))
        fig_fan.add_trace(go.Scatter(x=x_ax, y=p10_s, fill=None, mode='lines', line=dict(width=0), showlegend=False))
        fig_fan.add_trace(go.Scatter(x=x_ax, y=p90_s, fill='tonexty', fillcolor='rgba(255,0,0,0.1)', line=dict(width=0), name='Uncertainty (10-90%)'))
        fig_fan.add_trace(go.Scatter(x=x_ax, y=p50_s, mode='lines', name='Stress Scenario', line=dict(color='firebrick', width=3, dash='dot')))
        fig_fan.add_hline(y=1143*0.30, line_dash="dash", line_color="orange", annotation_text="30% Rationing")
        fig_fan.add_hline(y=1143*0.05, line_color="red", annotation_text="5% Blackout")
        fig_fan.update_layout(title='Storage Trajectory Gap', height=400, template='plotly_white', hovermode="x unified")
        st.plotly_chart(fig_fan, use_container_width=True)

    with tab2:
        # Histogram Logic
        weeks_idx = np.arange(1, inputs['weeks'] + 1)
        # Count failures occurring in each week
        counts = [np.sum(f30_s == w) for w in weeks_idx]
        probs = [c / 10000 for c in counts]
        
        # Create Dates
        start_date = datetime.now()
        dates = [(start_date + timedelta(weeks=int(i))).strftime('%b %d') for i in weeks_idx]
        
        fig_hist = go.Figure(go.Bar(x=dates, y=probs, marker_color='orange', name='Prob of <30%'))
        fig_hist.update_layout(
            title='<b>Most Likely Week of Failure</b> (Probability of hitting <30%)',
            yaxis_title='Probability',
            height=400, template='plotly_white'
        )
        st.plotly_chart(fig_hist, use_container_width=True)

else:
    st.info("👈 **Start here:** Select a **Weather Scenario** and add **Shocks** in the sidebar.")
