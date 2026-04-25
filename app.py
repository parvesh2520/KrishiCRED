import streamlit as st
import os
import pandas as pd
from databricks import sql
from openai import OpenAI
from dotenv import load_dotenv
import re

# 1. Load secrets from .env file
load_dotenv()

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")

# 2. Configure GenAI to use Databricks Foundation Model Endpoints
client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=f"https://{DATABRICKS_HOST}/serving-endpoints"
)

# ✅ WORKING AI MODELS
AVAILABLE_MODELS = {
    "Llama 3.3 70B (Recommended)": "databricks-meta-llama-3-3-70b-instruct",
    "Llama 3.1 405B (Most Powerful)": "databricks-meta-llama-3.1-405b-instruct",
    "Llama 4 Maverick": "databricks-llama-4-maverick",
    "Llama 3.1 8B (Fast)": "databricks-meta-llama-3-1-8b-instruct",
    "Qwen3 Next 80B": "databricks-qwen3-next-80b-a3b-instruct",
    "Gemma 3 12B": "databricks-gemma-3-12b"
}

# --- CUSTOM CSS FOR BEAUTIFUL UI ---
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* ── Reset & Base ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background-color: #0d1117;
        background-image:
            radial-gradient(ellipse 80% 50% at 50% -20%, rgba(34,197,94,0.12), transparent),
            radial-gradient(ellipse 60% 40% at 80% 80%, rgba(16,185,129,0.07), transparent);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] * {
        color: #d1d5db !important;
    }
    section[data-testid="stSidebar"] h3 {
        color: #f9fafb !important;
        font-size: 11px !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stTextInput label {
        color: #9ca3af !important;
        font-size: 12px !important;
    }

    /* ── Typography ── */
    h1, h2, h3 {
        font-family: 'DM Serif Display', Georgia, serif !important;
        color: #f0fdf4 !important;
    }
    h3 {
        font-size: 20px !important;
        margin-bottom: 16px !important;
    }

    /* ── Divider ── */
    hr {
        border: none !important;
        border-top: 1px solid rgba(255,255,255,0.07) !important;
        margin: 24px 0 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 22px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.4), 0 0 0 1px rgba(22,163,74,0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(22,163,74,0.35) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%) !important;
        font-size: 15px !important;
        padding: 12px 28px !important;
    }

    /* ── Inputs ── */
    .stTextInput input {
        background: #1f2937 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        color: #f9fafb !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 15px !important;
        padding: 12px 16px !important;
        transition: border-color 0.2s !important;
    }
    .stTextInput input:focus {
        border-color: #22c55e !important;
        box-shadow: 0 0 0 3px rgba(34,197,94,0.15) !important;
    }
    .stTextInput input::placeholder { color: #6b7280 !important; }

    /* ── Select ── */
    .stSelectbox > div > div {
        background: #1f2937 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        color: #f9fafb !important;
    }

    /* ── Alerts ── */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 10px !important;
        border: none !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stSuccess { background: rgba(20,83,45,0.4) !important; border-left: 3px solid #22c55e !important; }
    .stWarning { background: rgba(120,53,15,0.4) !important; border-left: 3px solid #f59e0b !important; }
    .stError   { background: rgba(127,29,29,0.4) !important; border-left: 3px solid #ef4444 !important; }
    .stInfo    { background: rgba(30,58,138,0.4) !important; border-left: 3px solid #3b82f6 !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #111827 !important;
        border-radius: 10px !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #9ca3af !important;
        border-radius: 7px !important;
        padding: 9px 18px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        transition: all 0.2s !important;
    }
    .stTabs [aria-selected="true"] {
        background: #16a34a !important;
        color: #ffffff !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.3) !important;
    }

    /* ── Dataframe ── */
    .stDataFrame {
        border-radius: 10px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #22c55e !important;
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: #1f2937 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #d1d5db !important;
    }
    .stDownloadButton > button:hover {
        background: #374151 !important;
        border-color: rgba(255,255,255,0.2) !important;
    }

    /* ── Metric card ── */
    .metric-card {
        background: #111827;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 22px 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #16a34a, #22c55e);
        border-radius: 14px 14px 0 0;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 30px;
        font-weight: 400;
        color: #f0fdf4;
        margin: 10px 0 6px;
        letter-spacing: -0.5px;
    }
    .metric-label {
        font-size: 11px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 500;
    }

    /* ── Analysis cards ── */
    .analysis-card {
        background: #111827;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 28px;
        margin: 12px 0;
    }
    .risk-low  { border-color: rgba(34,197,94,0.3);  background: rgba(20,83,45,0.2);  }
    .risk-medium { border-color: rgba(245,158,11,0.3); background: rgba(120,53,15,0.15); }
    .risk-high { border-color: rgba(239,68,68,0.3);  background: rgba(127,29,29,0.15); }

    /* ── AI output text ── */
    .ai-output {
        color: #d1d5db;
        line-height: 1.85;
        font-size: 15px;
        font-family: 'DM Sans', sans-serif;
    }
    .ai-output h3 {
        color: #f0fdf4 !important;
        border-bottom: 1px solid rgba(255,255,255,0.1) !important;
        padding-bottom: 10px !important;
        margin-bottom: 18px !important;
        font-size: 18px !important;
    }
    .ai-output p { margin: 12px 0; }
    .ai-output strong { color: #86efac; font-weight: 600; }

    /* ── Caption / small text ── */
    .stCaption, caption, small {
        color: #6b7280 !important;
        font-size: 12px !important;
    }

    /* ── Code blocks ── */
    code {
        background: #1f2937 !important;
        color: #86efac !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
        font-size: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---
def format_currency(amount):
    """Format numbers as Indian currency"""
    if isinstance(amount, (int, float)):
        return f"₹{amount:,.2f}"
    return str(amount)

def estimate_cibil_score(farmer_data, model_name):
    """Estimate CIBIL score from digital footprint when DB value is null."""
    prompt = f"""You are an agricultural credit scoring engine. Based on this farmer's digital footprint, estimate a synthetic CIBIL score between 300–900.

Return ONLY a JSON object like: {{"score": 720, "grade": "B+", "confidence": "High"}}

Data:
- UPI Monthly Avg: ₹{farmer_data.get('upi_monthly_avg_inr', 0):,.2f}
- UPI Transactions (30d): {farmer_data.get('upi_transaction_count_30d', 0)}
- Telecom Streak: {farmer_data.get('telecom_recharge_streak_months', 0)} months
- Zero Balance Days (6m): {farmer_data.get('days_with_zero_balance_6m', 'N/A')}
- Historical Default: {farmer_data.get('historical_loan_default', 'N/A')}
- Land Verified: {farmer_data.get('land_record_verified', 'N/A')}
- Insurance Enrolled: {farmer_data.get('pmfby_insurance_enrolled', 'N/A')}
- FPO/SHG Member: {farmer_data.get('is_fpo_shg_member', 'N/A')}
- Repayment History: {farmer_data.get('fpo_shg_loan_repayment_history', 'N/A')}

Respond with JSON only. No explanation."""

    try:
        import json
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100
        )
        text = response.choices[0].message.content.strip()
        return json.loads(text)
    except:
        return None

def format_ai_response(response_text):
    """Format AI response with better readability"""
    formatted = response_text.strip()
    return formatted


def display_metric_card(label, value, delta=None):
    """Display a beautiful metric card"""
    if delta:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div style="color: #22c55e; font-size: 13px; margin-top:4px;">{delta}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# --- BACKEND FUNCTIONS ---
def test_connection():
    """Test the database connection and return status."""
    try:
        connection = sql.connect(
            server_hostname=DATABRICKS_HOST,
            http_path=DATABRICKS_HTTP_PATH,
            access_token=DATABRICKS_TOKEN
        )
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        connection.close()
        return True, "✅ Connection successful"
    except Exception as e:
        return False, f"❌ Connection failed: {str(e)}"

@st.cache_data(ttl=600)
def get_available_farmer_ids(limit=10):
    """Fetch sample farmer IDs from the table."""
    try:
        connection = sql.connect(
            server_hostname=DATABRICKS_HOST,
            http_path=DATABRICKS_HTTP_PATH,
            access_token=DATABRICKS_TOKEN
        )
        cursor = connection.cursor()
        
        query = "SELECT farmer_id, synthetic_agri_cibil_score FROM workspace.default.agri_credit_features_1m LIMIT " + str(limit)
        cursor.execute(query)
        
        result = cursor.fetchall()
        cursor.close()
        connection.close()
        
        return [row[0] for row in result] if result else []
    except Exception as e:
        st.error(f"Error fetching farmer IDs: {e}")
        return []

@st.cache_data(ttl=600)
def fetch_farmer_data(farmer_id):
    """Connects to Databricks Delta Lake and fetches the farmer's row."""
    try:
        connection = sql.connect(
            server_hostname=DATABRICKS_HOST,
            http_path=DATABRICKS_HTTP_PATH,
            access_token=DATABRICKS_TOKEN
        )
        cursor = connection.cursor()
        
        query = f"SELECT * FROM workspace.default.agri_credit_features_1m WHERE farmer_id = \'{farmer_id}\' LIMIT 1"
        cursor.execute(query)
        
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        cursor.close()
        connection.close()
        
        if result:
            return dict(zip(columns, result[0]))
        return None
    except Exception as e:
        st.error(f"❌ Databricks Connection Error: {e}")
        return None

def generate_underwriting_summary(farmer_data, model_name):
    """Generate AI underwriting report with enhanced prompting for better output."""
    prompt = f"""You are an expert agricultural loan underwriter in India. Analyze this farmer's digital footprint and provide a comprehensive credit assessment.

STRUCTURE YOUR RESPONSE AS FOLLOWS:

**EXECUTIVE SUMMARY**
Provide a clear one-sentence risk verdict (Low Risk / Medium Risk / High Risk)

**DETAILED ANALYSIS**

1. **Credit Profile Assessment**
   - Synthetic Credit Score: {farmer_data.get('synthetic_agri_cibil_score', 'N/A')}
   - Historical Default Record: {farmer_data.get('historical_loan_default', 'N/A')}
   - Your analysis of creditworthiness

2. **Digital Payment Behavior**
   - Monthly UPI Activity: ₹{farmer_data.get('upi_monthly_avg_inr', 0):,.2f}
   - Transaction Volume: {farmer_data.get('upi_transaction_count_30d', 0)} transactions
   - Merchant vs P2P Ratio: {farmer_data.get('upi_merchant_vs_p2p_ratio', 'N/A')}
   - What this indicates about financial habits

3. **Agricultural Profile**
   - Land Area: {farmer_data.get('land_area_hectares', 'N/A')} hectares
   - Land Records Verified: {farmer_data.get('land_record_verified', 'N/A')}
   - Insurance Enrollment: {farmer_data.get('pmfby_insurance_enrolled', 'N/A')}
   - Assessment of agricultural stability

4. **Financial Reliability Indicators**
   - Telecom Recharge Streak: {farmer_data.get('telecom_recharge_streak_months', 0)} months
   - Zero Balance Days (6m): {farmer_data.get('days_with_zero_balance_6m', 'N/A')}
   - FPO/SHG Membership: {farmer_data.get('is_fpo_shg_member', 'N/A')}
   - Repayment History: {farmer_data.get('fpo_shg_loan_repayment_history', 'N/A')}
   - Analysis of payment discipline

**FINAL RECOMMENDATION**
Clear verdict with key reasoning points.

Make your response professional, data-driven, and easy to read with clear formatting.
"""
    
    try:
        response = client.chat.completions.create(
            model=model_name, 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200
        )
        return format_ai_response(response.choices[0].message.content)
    except Exception as e:
        error_msg = str(e)
        if "PERMISSION_DENIED" in error_msg or "rate limit" in error_msg.lower():
            return f"""⚠️ **Model Temporarily Unavailable**

The selected model has hit its rate limit. Please select **Llama 3.3 70B** from the sidebar.

**Quick Manual Assessment:**

**Credit Score:** {farmer_data.get('synthetic_agri_cibil_score', 'N/A')}  
**Monthly UPI Activity:** {format_currency(farmer_data.get('upi_monthly_avg_inr', 0))}  
**Telecom Reliability:** {farmer_data.get('telecom_recharge_streak_months', 0)} months streak  
**Land Verified:** {farmer_data.get('land_record_verified', 'N/A')}  
**Insurance:** {farmer_data.get('pmfby_insurance_enrolled', 'N/A')}  
**Default History:** {farmer_data.get('historical_loan_default', 'N/A')}
"""
        return f"⚠️ **AI Model Error:** {error_msg}"

def generate_loan_recommendation(farmer_data, model_name):
    """Generate structured loan recommendation."""
    prompt = f"""As a senior agricultural credit analyst, provide a detailed loan recommendation based on this farmer's profile.

STRUCTURE YOUR RESPONSE:

**LOAN RECOMMENDATION SUMMARY**

💰 **Recommended Loan Amount:** [Specify in INR with reasoning]

📊 **Proposed Interest Rate:** [Specify rate with justification]

📅 **Repayment Period:** [Specify duration and structure]

🔒 **Collateral/Security Requirements:**
   - List specific collateral needed
   - Security measures

⚠️ **Key Conditions:**
   - List important conditions or covenants
   - Monitoring requirements

**REASONING:**
Explain your recommendation based on:
- Credit Score: {farmer_data.get('synthetic_agri_cibil_score', 'N/A')}
- Land Holdings: {farmer_data.get('land_area_hectares', 'N/A')} hectares
- Monthly UPI: ₹{farmer_data.get('upi_monthly_avg_inr', 0):,.2f}
- Insurance Status: {farmer_data.get('pmfby_insurance_enrolled', 'N/A')}
- Default History: {farmer_data.get('historical_loan_default', 'N/A')}

Provide specific numbers and clear reasoning.
"""
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800
        )
        return format_ai_response(response.choices[0].message.content)
    except Exception as e:
        error_msg = str(e)
        if "PERMISSION_DENIED" in error_msg:
            return "⚠️ Model rate limited. Switch to **Llama 3.3 70B** in sidebar."
        return f"⚠️ Error: {error_msg}"

# --- MAIN APP UI ---
st.set_page_config(
    page_title="Krishi Command Center",
    layout="wide",
    page_icon="🌾",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_custom_css()

# Header
st.markdown("""
    <div style='text-align: center; padding: 48px 0 32px;'>
        <div style='font-size: 13px; letter-spacing: 0.18em; color: #22c55e; text-transform: uppercase; font-weight: 600; margin-bottom: 14px; font-family: DM Sans, sans-serif;'>
            Agricultural Intelligence Platform
        </div>
        <h1 style='
            font-family: DM Serif Display, Georgia, serif;
            color: #f0fdf4;
            font-size: 52px;
            font-weight: 400;
            margin: 0 0 14px;
            letter-spacing: -1px;
            line-height: 1.1;
        '>
            🌾 Krishi Command Center
        </h1>
        <p style='color: #6b7280; font-size: 16px; margin: 0; font-family: DM Sans, sans-serif; font-weight: 300; letter-spacing: 0.01em;'>
            Databricks Delta Lake &nbsp;·&nbsp; Foundation AI Models &nbsp;·&nbsp; Real-time Analytics
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔌 Test DB", use_container_width=True):
            with st.spinner("Testing..."):
                success, message = test_connection()
                if success:
                    st.success("✅ Connected")
                else:
                    st.error("❌ Failed")
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.success("✅ Cleared")
    
    st.markdown("---")
    
    st.markdown("### 🤖 AI Model Selection")
    selected_model_display = st.selectbox(
        "Choose Model:",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
        label_visibility="collapsed"
    )
    selected_model = AVAILABLE_MODELS[selected_model_display]
    
    if "llama" in selected_model.lower():
        st.success("✅ Verified • No limits")
    else:
        st.warning("⚠️ May have rate limits")
    
    st.caption(f"Endpoint: `{selected_model}`")
    
    st.markdown("---")
    
    st.markdown("### 📋 Sample Farmer IDs")
    if st.button("Load Sample IDs", use_container_width=True):
        with st.spinner("Fetching from Delta table..."):
            sample_ids = get_available_farmer_ids(10)
            if sample_ids:
                st.success(f"Found {len(sample_ids)} farmers")
                for fid in sample_ids[:5]:
                    if st.button(fid, key=f"btn_{fid}", use_container_width=True):
                        st.session_state.search_id = fid
            else:
                st.warning("No farmer IDs found")
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.info("This platform uses Databricks AI to provide instant credit assessments for agricultural loans based on digital footprints.")

# Main search interface
st.markdown("### 🔍 Farmer Credit Assessment")
col_search1, col_search2 = st.columns([4, 1])

with col_search1:
    search_id = st.text_input(
        "Enter Farmer ID",
        placeholder="e.g., ADHR-123456789",
        value=st.session_state.get('search_id', ''),
        label_visibility="collapsed"
    )

with col_search2:
    analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)

# Analysis section
if analyze_button and search_id:
    with st.spinner("🔄 Fetching data from Databricks Delta Lake..."):
        farmer_profile = fetch_farmer_data(search_id)
        
    if farmer_profile:
        st.success(f"✅ Successfully retrieved profile for **{search_id}**")
        
        st.markdown("### 📊 Key Metrics Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        # AFTER
        with col1:
            raw_score = farmer_profile.get('synthetic_agri_cibil_score')
            if raw_score is not None:
                display_metric_card("Credit Score", str(int(raw_score)))
            else:
                with st.spinner("Scoring..."):
                    est = estimate_cibil_score(farmer_profile, selected_model)
                if est:
                    display_metric_card(
                        "Credit Score (AI Est.)",
                                f"{est['score']} {est['grade']}",
                        delta=f"Confidence: {est['confidence']}"
                    )
                else:
                    display_metric_card("Credit Score", "Unavailable")
        
        with col2:
            display_metric_card(
                "Monthly UPI",
                format_currency(farmer_profile.get('upi_monthly_avg_inr', 0))
            )
        
        with col3:
            display_metric_card(
                "Land Area",
                f"{farmer_profile.get('land_area_hectares', 'N/A')} ha"
            )
        
        with col4:
            telecom_streak = farmer_profile.get('telecom_recharge_streak_months', 0)
            display_metric_card(
                "Telecom Streak",
                f"{telecom_streak} months"
            )
        
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs([
            "📊 Complete Profile",
            "🧠 AI Risk Assessment",
            "💰 Loan Recommendation"
        ])
        
        with tab1:
            st.markdown("### 📋 Farmer Profile Data")
            
            df_profile = pd.DataFrame([farmer_profile]).T
            df_profile.columns = ['Value']
            df_profile.index.name = 'Feature'
            
            st.dataframe(
                df_profile,
                use_container_width=True,
                height=500
            )
            
            csv = df_profile.to_csv()
            st.download_button(
                label="📥 Download Complete Profile (CSV)",
                data=csv,
                file_name=f"{search_id}_profile.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with tab2:
            st.markdown("### 🧠 AI-Powered Risk Assessment")
            
            with st.spinner(f"🤖 Analyzing with {selected_model_display}..."):
                ai_report = generate_underwriting_summary(farmer_profile, selected_model)
            
            risk_class = "analysis-card"
            if "low risk" in ai_report.lower():
                risk_class += " risk-low"
                st.success("✅ **RISK LEVEL: LOW**")
            elif "high risk" in ai_report.lower():
                risk_class += " risk-high"
                st.error("⚠️ **RISK LEVEL: HIGH**")
            else:
                risk_class += " risk-medium"
                st.warning("⚡ **RISK LEVEL: MEDIUM**")
            
            st.markdown(f"""
            <div class="{risk_class}">
                <div class="ai-output">
                    {ai_report.replace('**', '<strong>').replace('**', '</strong>')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### 💰 Loan Recommendation")
            
            with st.spinner(f"🤖 Generating recommendation with {selected_model_display}..."):
                loan_rec = generate_loan_recommendation(farmer_profile, selected_model)
            
            st.markdown(f"""
            <div class="analysis-card">
                <div class="ai-output">
                    {loan_rec.replace('**', '<strong>').replace('**', '</strong>')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.button("✅ Approve Loan", type="primary", use_container_width=True)
            with col2:
                st.button("📝 Request More Info", use_container_width=True)
    
    else:
        st.error(f"❌ Farmer ID **{search_id}** not found in database")
        st.info("💡 Click **Load Sample IDs** in the sidebar to see available farmers")

elif analyze_button:
    st.warning("⚠️ Please enter a Farmer ID to proceed")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #4b5563; padding: 24px 0 32px; font-family: DM Sans, sans-serif;'>
        <p style='margin: 0; font-size: 13px;'>
            🌾 <strong style="color: #6b7280;">Krishi Command Center</strong>
            &nbsp;·&nbsp; Databricks &nbsp;·&nbsp; Delta Lake &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; AI Foundation Models
        </p>
        <p style='margin: 6px 0 0; font-size: 11px; letter-spacing: 0.05em; text-transform: uppercase; color: #374151;'>
            Secure · Scalable · Real-time
        </p>
    </div>
""", unsafe_allow_html=True)
