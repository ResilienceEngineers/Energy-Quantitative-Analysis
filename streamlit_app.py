import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIGURATION & PHYSICS ENGINE ---

class GermanEnergyModel:
    def __init__(self):
        # PHYSICAL CONSTANTS [Sources: GIE, BNetzA]
        self.GAS_CAP_TWH = 251.14
        self.TECH_WITHDRAW_CAP = 7.086  # Max technical withdrawal rate (TWh/d)
        
        # DEMAND BASELINES (Winter 2026 calibrated)
        self.BASE_HEAT_DEMAND = 2.8     # TWh/d at 0°C deviation
        self.BASE_ELEC_GAS_DEMAND = 1.2 # TWh/d normal gas-to-power
        self.BASE_IND_DEMAND = 1.1      # TWh/d Industry baseload

    def ornstein_uhlenbeck_temp(self, n_sims, days, mu_target, theta=0.15, sigma=2.5):
        """
        Stochastic Weather Generator.
        Simulates temperature deviations (°C) with mean reversion to 'mu_target'.
        """
        T = np.zeros((n_sims, days))
        T[:,0] = np.random.normal(0, 1, n_sims) # Start at current deviation
        
        for t in range(1, days):
            dW = np.random.normal(0, 1, n_sims)
            # Mean reversion towards the scenario target (mu)
            T[:, t] = T[:, t-1] + theta*(mu_target - T[:, t-1]) + sigma*dW
        return T

    def simulate_path(self, inputs, is_baseline=False):
        """
        Runs the simulation. If is_baseline=True, forces 'Normal' conditions 
        to create a comparison benchmark.
        """
        np.random.seed(42) # Reproducible comparison
        n_sims = 1500
        days = inputs['duration']
        
        # --- A. SETUP ---
        storage = np.zeros((n_sims, days + 1))
        # Input is %, convert to TWh for physics
        start_twh = (inputs['start_pct'] / 100.0) * self.GAS_CAP_TWH
        storage[:, 0] = start_twh
        
        # --- B. DRIVERS (SCENARIO VS BASELINE) ---
        if is_baseline:
            # Baseline: Normal weather (0°C dev), Normal Renewables (beta 2,2), No Outages
            temps = self.ornstein_uhlenbeck_temp(n_sims, days, mu_target=0.0)
            re_yield = np.random.beta(2.0, 2.0, (n_sims, days)) # Avg 0.5
            daily_import = np.full((n_sims, days), inputs['base_import_cap']) 
            ind_cut = 0.0
            mitigation_add = 0.0
        else:
            # Active Risk Scenario
            temps = self.ornstein_uhlenbeck_temp(n_sims, days, mu_target=inputs['temp_deviation'])
            
            # Dunkelflaute Logic
            if inputs['dunkelflaute_active']:
                # Skew Beta dist based on intensity
                alpha_d = max(0.5, 2.0 - (inputs['dunkel_intensity'] * 0.3))
                re_yield = np.random.beta(alpha_d, 5.0, (n_sims, days))
            else:
                re_yield = np.random.beta(2.0, 2.0, (n_sims, days))
            
            # Supply Stack
            daily_import = np.full((n_sims, days), inputs['base_import_cap'])
            if inputs['risk_norway']: daily_import -= 1.1
            if inputs['risk_lng']: daily_import -= 0.6
            if inputs['risk_transit']: daily_import -= 0.4
            
            # Mitigations
            ind_cut = inputs['mitigation_ind_cut']
            mitigation_add = 0.0
            if inputs['mit_fuel_switch']: mitigation_add += 0.3
            if inputs['mit_strategic']: mitigation_add += 0.2

        # --- C. TIME LOOP ---
        shortage_flags = np.zeros(n_sims)
        failure_day_indices = np.full(n_sims, np.nan)
        
        for t in range(days):
            # 1. Demand Calculation
            heat_load = self.BASE_HEAT_DEMAND * (1 + 0.12 * np.maximum(0, -temps[:, t]))
            
            # Power: Gas fills gap from renewables (max ~3.0 TWh/d)
            max_gas_power = 3.0 
            power_load = max_gas_power * (1 - re_yield[:, t])
            
            # Total Demand (Apply Industry Cuts to Base Ind)
            industry_load = self.BASE_IND_DEMAND * (1 - ind_cut/100)
            total_demand = heat_load + industry_load + power_load
            
            # 2. Balance
            net_flow = (daily_import[:, t] + mitigation_add) - total_demand
            
            # 3. Physics (Ratchet Effect)
            # Withdrawal limit drops as storage empties (Pressure ~ Volume)
            fill_pct = storage[:, t] / self.GAS_CAP_TWH
            phys_limit = self.TECH_WITHDRAW_CAP * np.sqrt(np.maximum(0, fill_pct))
            
            # 4. Solve Flow
            desired_withdraw = np.maximum(0, -net_flow)
            actual_withdraw = np.minimum(desired_withdraw, phys_limit)
            
            # Shortage Check
            unmet = desired_withdraw - actual_withdraw
            shortage_flags += (unmet > 0.05).astype(int)
            
            # Critical Level Check (<10%)
            is_critical = storage[:, t] < (self.GAS_CAP_TWH * 0.10)
            new_fails = is_critical & np.isnan(failure_day_indices)
            failure_day_indices[new_fails] = t
            
            # Update Storage
            inj = np.minimum(np.maximum(0, net_flow), 4.5) # Max injection
            delta = inj - actual_withdraw
            storage[:, t+1] = np.clip(storage[:, t] + delta, 0, self.GAS_CAP_TWH)
            
        return storage, failure_day_indices, shortage_flags

# --- 2. STREAMLIT APP LAYOUT ---

st.set_page_config(page_title="DE Energy Risk", layout="wide", page_icon="⚡")

# Custom CSS for styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stMetricValue { font-size: 24px !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🇩🇪 German Energy Risk: Strategic Command Center")
st.markdown("### Stochastic Stress-Test Calculator (Winter 2026)")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. SCENARIO PRESETS
    scenario = st.selectbox(
        "Select Scenario Preset",
        ('1. Normal Winter', '2. Cold Snap (2 Weeks)', '3. Polar Vortex (Severe)'),
        index=0
    )
    
    # Defaults based on selection
    if scenario == '1. Normal Winter':
        def_temp, def_imp, def_dunkel = 0.0, 5.0, False
    elif scenario == '2. Cold Snap (2 Weeks)':
        def_temp, def_imp, def_dunkel = -3.5, 5.2, True
    else: # Polar Vortex
        def_temp, def_imp, def_dunkel = -7.0, 5.5, True

    st.markdown("---")
    
    # 2. VARIABLES
    st.subheader("1. Scenario Variables")
    temp_dev = st.slider("Temp Deviation (°C)", -10.0, 5.0, def_temp, 0.5)
    base_imp = st.slider("Import Capacity (TWh/d)", 2.0, 7.0, def_imp, 0.1)
    dunkel_active = st.checkbox("Activate Dunkelflaute (Low Wind)", value=def_dunkel)
    dunkel_int = st.slider("Dunkelflaute Intensity", 1.0, 5.0, 2.0, 0.5, disabled=not dunkel_active)
    
    st.markdown("---")
    
    # 3. RISKS
    st.subheader("2. Supply Risks")
    risk_norway = st.checkbox("Norway Outage (-1.1 TWh/d)")
    risk_lng = st.checkbox("LNG Congestion (-0.6 TWh/d)")
    risk_transit = st.checkbox("Transit Cut (-0.4 TWh/d)")
    
    st.markdown("---")
    
    # 4. CONFIG
    st.subheader("3. System Config")
    start_date = st.date_input("Start Date", datetime(2026, 1, 20))
    duration = st.slider("Forecast Days", 30, 150, 90)
    start_pct = st.slider("Start Storage %", 20.0, 100.0, 40.5, 0.1)
    
    st.markdown("---")
    
    # 5. MITIGATION
    st.subheader("4. Mitigation Stack")
    mit_ind = st.slider("Industry Demand Cut (%)", 0, 50, 10, 5)
    mit_fuel = st.checkbox("Fuel Switch (+0.3 TWh)")
    mit_res = st.checkbox("Strategic Reserve (+0.2 TWh)")

# --- MAIN LOGIC ---

# Run Model
model = GermanEnergyModel()
inputs = {
    'duration': duration,
    'start_pct': start_pct,
    'base_import_cap': base_imp,
    'temp_deviation': temp_dev,
    'dunkelflaute_active': dunkel_active,
    'dunkel_intensity': dunkel_int,
    'risk_norway': risk_norway,
    'risk_lng': risk_lng,
    'risk_transit': risk_transit,
    'mitigation_ind_cut': mit_ind,
    'mit_fuel_switch': mit_fuel,
    'mit_strategic': mit_res
}

# Simulations
s_risk, f_risk, short_risk = model.simulate_path(inputs, is_baseline=False)
s_base, f_base, short_base = model.simulate_path(inputs, is_baseline=True)

# Stats
dates = [pd.Timestamp(start_date) + timedelta(days=i) for i in range(duration + 1)]
factor = 100.0 / model.GAS_CAP_TWH

p50_risk = np.median(s_risk, axis=0) * factor
p10_risk = np.percentile(s_risk, 10, axis=0) * factor
p90_risk = np.percentile(s_risk, 90, axis=0) * factor
p50_base = np.median(s_base, axis=0) * factor

fail_counts = f_risk[~np.isnan(f_risk)]
fail_dates = [pd.Timestamp(start_date) + timedelta(days=int(d)) for d in fail_counts]
risk_prob = np.mean(short_risk > 0) * 100

# Calculate Net Supply for Passport
net_supply = base_imp
if risk_norway: net_supply -= 1.1
if risk_lng: net_supply -= 0.6
if risk_transit: net_supply -= 0.4

# --- LAYOUT: KPIS ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Start Level", f"{start_pct}%", f"{(start_pct/100)*251.14:.1f} TWh")
with col2:
    delta = p50_risk[-1] - p50_base[-1]
    st.metric("End Level (Median)", f"{p50_risk[-1]:.1f}%", f"{delta:.1f}% vs Base")
with col3:
    st.metric("Critical Risk (<10%)", f"{np.mean(~np.isnan(f_risk))*100:.1f}%", "Probability")
with col4:
    st.metric("Shortage Risk", f"{risk_prob:.1f}%", "Demand Unmet")

# --- PASSPORT ---
st.info(f"""
**📋 ACTIVE SCENARIO PASSPORT:** Weather: {temp_dev}°C | 
Dunkelflaute: {dunkel_active} | 
Net Import: {net_supply:.1f} TWh/d | 
Industry Cut: {mit_ind}%
""")

# --- CHARTS ---
tab1, tab2 = st.tabs(["📉 Trajectory Analysis", "⚠️ Risk Timing"])

with tab1:
    fig = go.Figure()
    
    # Baseline
    fig.add_trace(go.Scatter(x=dates, y=p50_base, mode='lines', name='Baseline (Normal)', line=dict(color='green', dash='dash')))
    
    # Risk Scenario
    fig.add_trace(go.Scatter(x=dates, y=p50_risk, mode='lines', name='Active Scenario', line=dict(color='#c0392b', width=3)))
    
    # Confidence Interval
    fig.add_trace(go.Scatter(x=dates, y=p90_risk, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=dates, y=p10_risk, mode='lines', line=dict(width=0), fill='tonexty', 
                             fillcolor='rgba(192, 57, 43, 0.2)', name='Risk Uncertainty (80%)'))
    
    # Thresholds
    fig.add_hline(y=40, line_dash="dot", annotation_text="40% Re-Inject", annotation_position="bottom right")
    fig.add_hline(y=10, line_color="black", line_width=2, annotation_text="10% CRITICAL", annotation_position="bottom right")
    
    fig.update_layout(
        title="Storage Projection: Baseline vs. Risk Scenario",
        yaxis_title="Storage Level (% Full)",
        height=500,
        hovermode="x unified",
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if len(fail_dates) > 0:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=fail_dates, nbinsx=20, marker_color='#c0392b'))
        fig2.update_layout(
            title="When does the system crash? (Storage < 10%)",
            xaxis_title="Date",
            yaxis_title="Frequency of Failure",
            height=500
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.success("✅ NO CRITICAL FAILURES SIMULATED IN THIS SCENARIO")

# --- CONTEXT ---
with st.expander("📘 BEST PRACTICE GUIDE & REALITY CHECK"):
    st.markdown("""
    #### 1. Standard Import Capacity
    * **Normal Flow:** ~4.0 - 4.5 TWh/day (Norway + LNG + EU Interconnectors).
    * **Crisis Flow:** Can surge to ~5.0 TWh/day if prices are high.
    * **Risk Impact:** Norway provides ~1.1 TWh/d. Losing this is the biggest single point of failure.

    #### 2. Mitigation Measures
    * **Fuel Switch:** Reactivating reserve plants saves ~0.3 TWh/d gas.
    * **Strategic Reserve:** Government "Trading Hub Europe" release adds ~0.2 TWh/d.
    * **Ind. Demand Cut:** 15-20% reduction is the standard BNetzA crisis protocol.

    #### 3. Weather Impact
    * **Normal:** 0°C deviation. Storage ends winter at ~40-50%.
    * **Cold Snap:** -3°C to -5°C. Increases heat demand by 15-20%.
    * **Polar Vortex:** -8°C. Drains storage rapidly; "Ratchet Effect" limits withdrawal below 20% fill.
    """)
