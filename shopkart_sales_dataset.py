import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="E-Commerce Profit Predictor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD MODEL
# =========================================================
model = joblib.load("gradient_boosting_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

    /* ---------- Main Background ---------- */
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(99,102,241,0.12), transparent 30%),
            radial-gradient(circle at 90% 20%, rgba(14,165,233,0.12), transparent 30%),
            linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    }

    /* ---------- Remove Default Padding ---------- */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* ---------- Main Header ---------- */
    .hero {
        padding: 35px 40px;
        border-radius: 25px;
        background: linear-gradient(
            135deg,
            #4f46e5 0%,
            #6366f1 45%,
            #0ea5e9 100%
        );
        color: white;
        box-shadow: 0 15px 40px rgba(79,70,229,0.25);
        margin-bottom: 30px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 17px;
        opacity: 0.92;
        margin: 0;
    }

    /* ---------- Section Headers ---------- */
    .section-title {
        font-size: 24px;
        font-weight: 750;
        color: #1e293b;
        margin: 12px 0 18px 0;
    }

    /* ---------- Cards ---------- */
    .card {
        background: rgba(255,255,255,0.82);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.8);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(15,23,42,0.07);
        margin-bottom: 20px;
    }

    /* ---------- Input Labels ---------- */
    label {
        font-weight: 600 !important;
        color: #334155 !important;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 14px;
        padding: 14px 20px;
        font-size: 18px;
        font-weight: 700;
        color: white;
        background: linear-gradient(
            135deg,
            #4f46e5,
            #6366f1,
            #0ea5e9
        );
        box-shadow: 0 8px 20px rgba(79,70,229,0.25);
        transition: all 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 28px rgba(79,70,229,0.35);
    }

    /* ---------- Result Cards ---------- */
    .profit-high {
        background: linear-gradient(
            135deg,
            rgba(16,185,129,0.15),
            rgba(34,197,94,0.08)
        );
        border: 2px solid rgba(16,185,129,0.35);
        border-radius: 22px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 12px 30px rgba(16,185,129,0.12);
    }

    .profit-low {
        background: linear-gradient(
            135deg,
            rgba(239,68,68,0.13),
            rgba(248,113,113,0.07)
        );
        border: 2px solid rgba(239,68,68,0.35);
        border-radius: 22px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 12px 30px rgba(239,68,68,0.12);
    }

    .result-icon {
        font-size: 55px;
        margin-bottom: 8px;
    }

    .result-title {
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .result-text {
        font-size: 16px;
        color: #475569;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 14px;
        padding-top: 30px;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827, #1e1b4b);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO SECTION
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">🛍️ E-Commerce Profit Predictor</div>
    <p class="hero-subtitle">
        AI-powered prediction system to determine whether an order
        is likely to generate High Profit or Low Profit.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("## 🧠 ML Dashboard")

    st.markdown("---")

    st.markdown("""
    ### 📊 About

    This application uses a trained
    **Gradient Boosting Machine Learning model**
    to predict ecommerce profit categories.

    ### ⚡ Workflow

    1. Enter customer details
    2. Enter order information
    3. Select location & category
    4. Click **Predict Profit**
    5. View prediction confidence
    """)

    st.markdown("---")

    st.info(
        "💡 Tip: Accurate customer and order information "
        "helps improve prediction reliability."
    )

# =========================================================
# CUSTOMER INFORMATION
# =========================================================
st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    customer_age = st.number_input(
        "🎂 Customer Age",
        min_value=18,
        max_value=80,
        value=30
    )

with col2:
    gender_option = st.selectbox(
        "⚧ Gender",
        ["Male", "Female"]
    )

with col3:
    rating = st.slider(
        "⭐ Customer Rating",
        min_value=1.0,
        max_value=5.0,
        value=4.0,
        step=0.1
    )

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# ORDER INFORMATION
# =========================================================
st.markdown(
    '<div class="section-title">📦 Order Information</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    qty = st.number_input(
        "🔢 Quantity",
        min_value=1,
        max_value=100,
        value=2
    )

with col2:
    unit_price = st.number_input(
        "💰 Unit Price",
        min_value=1.0,
        value=500.0,
        step=50.0
    )

with col3:
    discount = st.number_input(
        "🏷️ Discount (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0
    )

col1, col2, col3 = st.columns(3)

with col1:
    shipping = st.number_input(
        "🚚 Shipping Cost",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

with col2:
    delivery = st.number_input(
        "📅 Delivery Days",
        min_value=1,
        max_value=60,
        value=5
    )

with col3:
    order_date = st.date_input(
        "🗓️ Order Date",
        datetime.today()
    )

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PRODUCT INFORMATION
# =========================================================
st.markdown(
    '<div class="section-title">🏪 Product & Location</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    city = st.selectbox(
        "📍 City",
        [
            "Bengaluru",
            "Chennai",
            "Delhi",
            "Hyderabad",
            "Jaipur",
            "Lucknow",
            "Mumbai",
            "Pune"
        ]
    )

with col2:

    category = st.selectbox(
        "🛒 Product Category",
        [
            "Beauty",
            "Electronics",
            "Fashion",
            "Furniture",
            "Grocery",
            "Sports"
        ]
    )

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# DATE FEATURES
# =========================================================
month = order_date.month
year = order_date.year
day_of_week = order_date.weekday()
weekend = 1 if day_of_week >= 5 else 0

# Encode Gender
gender = 1 if gender_option == "Male" else 0

# =========================================================
# INPUT DATA
# =========================================================
input_data = {

    "Customer_Age": customer_age,
    "Gender": gender,
    "Qty": qty,
    "Unit Price": unit_price,
    "Discount": discount,
    "Shipping": shipping,
    "Delivery": delivery,
    "Rating": rating,
    "Month": month,
    "Year": year,
    "Day_of_Week": day_of_week,
    "Weekend": weekend,

    "City_Chennai": 0,
    "City_Delhi": 0,
    "City_Hyderabad": 0,
    "City_Jaipur": 0,
    "City_Lucknow": 0,
    "City_Mumbai": 0,
    "City_Pune": 0,

    "Category_Electronics": 0,
    "Category_Fashion": 0,
    "Category_Furniture": 0,
    "Category_Grocery": 0,
    "Category_Sports": 0
}

# City Encoding
if city != "Bengaluru":
    input_data[f"City_{city}"] = 1

# Category Encoding
if category != "Beauty":
    input_data[f"Category_{category}"] = 1

# =========================================================
# FEATURE ORDER
# =========================================================
feature_order = [

    "Customer_Age",
    "Gender",
    "Qty",
    "Unit Price",
    "Discount",
    "Shipping",
    "Delivery",
    "Rating",
    "Month",
    "Year",
    "Day_of_Week",
    "Weekend",

    "City_Chennai",
    "City_Delhi",
    "City_Hyderabad",
    "City_Jaipur",
    "City_Lucknow",
    "City_Mumbai",
    "City_Pune",

    "Category_Electronics",
    "Category_Fashion",
    "Category_Furniture",
    "Category_Grocery",
    "Category_Sports"
]

input_df = pd.DataFrame([input_data])

input_df = input_df[feature_order]

# Scale
input_scaled = scaler.transform(input_df)

# =========================================================
# ORDER SUMMARY
# =========================================================
estimated_value = qty * unit_price
discount_amount = estimated_value * (discount / 100)
estimated_after_discount = estimated_value - discount_amount

st.markdown(
    '<div class="section-title">📋 Order Summary</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("🛍️ Quantity", qty)

with m2:
    st.metric("💰 Order Value", f"₹{estimated_value:,.0f}")

with m3:
    st.metric("🏷️ Discount", f"₹{discount_amount:,.0f}")

with m4:
    st.metric(
        "💵 After Discount",
        f"₹{estimated_after_discount:,.0f}"
    )

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# PREDICTION
# =========================================================
if prediction := st.button("Predict Profit"):

    with st.spinner("🤖 Analyzing order with AI..."):

        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(input_scaled)[0]

        low_profit_probability = probability[0] * 100
        high_profit_probability = probability[1] * 100

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🎯 Prediction Result</div>',
        unsafe_allow_html=True
    )

    # =====================================================
    # HIGH PROFIT
    # =====================================================
    if prediction == 1:

        st.markdown(f"""
        <div class="prediction-card high-profit">

            

         

                
                    PREDICTION RESULT
                

               
                    High Profit
               

                    This order shows strong potential for profitability.
                

                {high_profit_probability:.1f}
              

      
        """, unsafe_allow_html=True)

        st.snow()

        confidence = high_profit_probability

    # =====================================================
    # LOW PROFIT
    # =====================================================
    else:

        st.markdown(f"""
        <div class="prediction-card low-profit">

          


               
                    PREDICTION RESULT
              

                
                    Low Profit
               
                    This order may generate a lower profit margin.
              

               {low_profit_probability:.1f}
               


        """, unsafe_allow_html=True)

        confidence = low_profit_probability

    # =====================================================
    # MODEL CONFIDENCE
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📊 Model Confidence")

    st.progress(
        int(confidence),
        text=f"Prediction Confidence: {confidence:.1f}%"
    )

    # =====================================================
    # PROBABILITY METRICS
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "🟢 High Profit Probability",
            f"{high_profit_probability:.1f}%"
        )

    with col2:

        st.metric(
            "🔴 Low Profit Probability",
            f"{low_profit_probability:.1f}%"
        )
# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    🧠 Powered by Gradient Boosting Machine Learning
    &nbsp; | &nbsp;
    🛍️ E-Commerce Profit Intelligence System
</div>
""", unsafe_allow_html=True)
