import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# --- PAGE CONFIG ---
# Light, clean, wide layout with no sidebar by default
st.set_page_config(page_title="Nexus Bank | Credit Intelligence", page_icon="🏦", layout="centered", initial_sidebar_state="collapsed")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #1e3d59; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button { background-color: #1e3d59; color: white; border-radius: 8px; width: 100%; font-weight: bold; }
    .stButton>button:hover { background-color: #ff6e40; color: white; }
    .metric-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('models/credit_rf_model.pkl')
    except:
        return None

model = load_model()

# --- HEADER ---
st.markdown("<h1 style='text-align: center;'>🏦 Nexus Bank</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6c757d; font-size: 1.2rem;'>AI Credit Intelligence Engine &middot; Engineered by A.C.Pravin Kumar</p>", unsafe_allow_html=True)
st.markdown("---")

if model is None:
    st.error("Model not found! Please run the training script first to generate the RF model.")
    st.stop()

# --- APPLICATION WIZARD ---
st.markdown("### 📝 Instant Credit Application")

with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Applicant Age", min_value=18, max_value=100, value=30)
        income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=65000, step=5000)
        years_emp = st.number_input("Years Employed", min_value=0, max_value=50, value=5)
        
    with col2:
        debt = st.number_input("Total Outstanding Debt ($)", min_value=0, max_value=200000, value=15000, step=1000)
        credit_util = st.slider("Credit Utilization Ratio", 0.0, 1.0, 0.30, help="Percentage of available credit currently used.")
        late_payments = st.selectbox("Late Payments (Last 12 Months)", [0, 1, 2, 3, 4, 5])

# Feature Engineering happening live in the app
dti_ratio = debt / income if income > 0 else 0

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Evaluate Creditworthiness"):
    with st.spinner("Connecting to underwriting AI..."):
        # Format input array exactly as trained
        input_data = pd.DataFrame([[age, income, debt, dti_ratio, years_emp, credit_util, late_payments]], 
                                  columns=['Age', 'Annual_Income', 'Total_Debt', 'DTI_Ratio', 'Years_Employed', 'Credit_Utilization', 'Late_Payments'])
        
        # Predict
        probability_of_default = model.predict_proba(input_data)[0][1]
        
        # Convert probability of default into a traditional Credit Score (300 to 850)
        # Lower probability of default = Higher Score
        credit_score = int(850 - (probability_of_default * 550))
        
        st.markdown("---")
        st.markdown("### 📊 Underwriting Results")
        
        # Determine status
        if credit_score >= 650:
            status = "APPROVED"
            status_color = "#00cc96"
        else:
            status = "DECLINED"
            status_color = "#ef553b"
            
        # Plotly Gauge Chart for visual flair
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = credit_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"<span style='font-size:1.5em;color:{status_color}'><b>{status}</b></span><br><span style='font-size:0.8em;color:gray'>Calculated Credit Score</span>"},
            gauge = {
                'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': status_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [300, 579], 'color': "rgba(239, 85, 59, 0.3)"},
                    {'range': [580, 669], 'color': "rgba(255, 161, 90, 0.3)"},
                    {'range': [670, 739], 'color': "rgba(0, 204, 150, 0.3)"},
                    {'range': [740, 850], 'color': "rgba(0, 204, 150, 0.6)"}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': credit_score}
            }))
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Risk Factors Analysis
        st.markdown("#### Key Risk Factors Analyzed")
        st.info(f"**Debt-to-Income (DTI):** {(dti_ratio*100):.1f}% (Bank threshold is typically < 36%)")
        if late_payments > 0:
            st.warning(f"**Derogatory Marks:** {late_payments} recent late payment(s) heavily impacted the score.")
        else:
            st.success("**Payment History:** Flawless payment history positively impacted the score.")