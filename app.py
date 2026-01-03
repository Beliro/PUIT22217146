import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import time

# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(
    page_title="NYC Real Estate Advisor",
    page_icon="g",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for that "SaaS" look (Teal & Clean)
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    /* Button Styling - Rounded & Shadow */
    div.stButton > button {
        background-color: #008080; /* Teal */
        color: white;
        border-radius: 12px;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #006666;
        box-shadow: 0px 6px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #008080;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD THE BRAIN (MODEL) ---
@st.cache_resource
def load_model():
    """Loads the model and the feature names to ensure inputs match perfectly."""
    try:
        # Load the dictionary we saved earlier
        data = joblib.load('nyc_real_estate_advisor.pkl')
        return data['model'], data['features']
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please run your training notebook first.")
        return None, None

model, feature_names = load_model()

# --- 3. SIDEBAR: USER INPUTS ---
st.sidebar.header("🏡 Property Details")
st.sidebar.markdown("Configure the listing below:")

# Categorical Inputs
neighbourhood = st.sidebar.selectbox(
    "Neighborhood Group",
    ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
)

room_type = st.sidebar.selectbox(
    "Room Type",
    ["Entire home/apt", "Private room", "Shared room"]
)

# Numerical Inputs (Sliders for interactive feel)
st.sidebar.subheader("📍 Location & Availability")
latitude = st.sidebar.slider("Latitude", 40.50, 40.90, 40.71, step=0.001)
longitude = st.sidebar.slider("Longitude", -74.25, -73.70, -73.98, step=0.001)
min_nights = st.sidebar.number_input("Minimum Nights", 1, 365, 2)
availability = st.sidebar.slider("Days Available (per year)", 0, 365, 150)

st.sidebar.subheader("📈 Activity Metrics")
reviews = st.sidebar.number_input("Total Reviews", 0, 1000, 10)
reviews_per_month = st.sidebar.slider("Reviews Per Month", 0.0, 20.0, 1.5)
host_listings = st.sidebar.number_input("Host's Total Listings", 1, 100, 1)

# The "Staleness" Feature we engineered
days_since_review = st.sidebar.slider("Days Since Last Review", 0, 3650, 14, help="How many days ago was the last review written?")

# --- 4. PREPROCESSING ENGINE ---
def preprocess_input(input_dict, model_features):
    """
    Converts user input into the exact One-Hot Encoded format the model expects.
    """
    # 1. Create a DataFrame from simple inputs
    df = pd.DataFrame([input_dict])
    
    # 2. One-Hot Encode (Get Dummies)
    df_processed = pd.get_dummies(df)
    
    # 3. Align with Model Features (The Critical Step)
    # We reindex to ensure we have ALL columns the model learned (filling missing ones with 0)
    df_final = df_processed.reindex(columns=model_features, fill_value=0)
    
    return df_final

# --- 5. MAIN PAGE LAYOUT ---
st.title("🏙️ NYC Real Estate AI Advisor")
st.markdown("### Intelligent Pricing Engine based on Data Science")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    This tool uses a machine learning model to estimate the fair market value of Airbnb listings in New York.
    Adjust the parameters in the sidebar to simulate different property scenarios.
    """)
    
    # PREDICT BUTTON
    if st.button("🚀 Predict Price"):
        if model:
            with st.spinner("Analyzing market patterns..."):
                time.sleep(0.5) # UI effect
                
                # Gather inputs
                input_data = {
                    'neighbourhood_group': neighbourhood,
                    'room_type': room_type,
                    'latitude': latitude,
                    'longitude': longitude,
                    'minimum_nights': min_nights,
                    'number_of_reviews': reviews,
                    'reviews_per_month': reviews_per_month,
                    'calculated_host_listings_count': host_listings,
                    'availability_365': availability,
                    'days_since_review': days_since_review
                }
                
                # Preprocess
                X_pred = preprocess_input(input_data, feature_names)
                
                # Predict (Result is Log Price, so we convert back)
                log_prediction = model.predict(X_pred)
                price_prediction = np.expm1(log_prediction)[0]
                
                # Display Results
                st.success("Analysis Complete!")
                
                # Result Container with Shadow
                st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
                    <h2 style="color: #333; margin-bottom: 0;">Recommended Price</h2>
                    <h1 style="color: #008080; font-size: 3.5rem; margin: 0;">${price_prediction:.2f} <span style="font-size: 1rem; color: gray;">/ night</span></h1>
                    <p style="color: #666;">Reliability Range: ${price_prediction - 37:.2f} — ${price_prediction + 37:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Feature Importance Chart (Visual Logic)
                st.markdown("### 🧠 What drove this decision?")
                if hasattr(model, 'feature_importances_'):
                    importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values(by='Importance', ascending=True).tail(7) # Top 7
                    
                    fig = px.bar(
                        importance, 
                        x='Importance', 
                        y='Feature', 
                        orientation='h',
                        title="Top Factors Influencing Price",
                        color='Importance',
                        color_continuous_scale='Teal'
                    )
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Sidebar Metrics Panel
    st.info("📊 Model Specs")
    st.metric("Model Type", "Random Forest")
    st.metric("Accuracy (R2)", "54%")
    st.metric("Avg Error Margin", "± $37.26")
    
    st.write("---")
    st.markdown("**Batch Upload**")
    st.markdown("Have a CSV file? Upload it here to predict multiple listings at once.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} rows.")
            if st.button("Process Batch"):
                # Simplified batch logic for demo
                st.warning("Batch processing requires matching column names. (Feature for future release)")
        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: grey;'>Built with Streamlit & Python • © 2026 NYC Real Estate Project</div>", unsafe_allow_html=True)