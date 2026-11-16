import streamlit as st
import pandas as pd
import joblib
import os

st.markdown(
    """
    <div style="text-align: center;">
        <h3>Project Developed By</h3>
        <p><b>Samrat Tripathi</b> | UID: 26BAI70082</p>
        <p><b>Anshu Yadav</b> | UID: 26BAI70062</p>
        <p><b>Chandan Sharma</b> | UID: 26BAI70060</p>
        <br>
        <p><b>Section:</b> 26AML-101</p>
        <p><b>Group:</b> B</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.divider()

# Page configuration
st.set_page_config(
    page_title="Bengaluru House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

# Check model file
if not os.path.exists("house_price_model.pkl"):
    st.error(
        "Trained model not found. "
        "Please run model.ipynb and save the model first."
    )
    st.stop()

model = load_model()

# Application heading
st.title("🏠 Bengaluru House Price Prediction")
st.write(
    "Enter the property details below to estimate "
    "the house price in Bengaluru."
)

st.divider()

# User inputs
st.subheader("Property Details")

area_type = st.selectbox(
    "Area Type",
    [
        "Super built-up  Area",
        "Built-up  Area",
        "Plot  Area",
        "Carpet  Area"
    ]
)

# Load available locations from the original dataset
@st.cache_data
def get_locations():
    data = pd.read_csv("Bengaluru_House_Data.csv")
    locations = (
        data["location"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    return sorted(locations)

locations = get_locations()

location = st.selectbox(
    "Select Location",
    locations,
    index=None,
    placeholder="Choose a Bengaluru location"
)

total_sqft = st.number_input(
    "Total Square Feet",
    min_value=100.0,
    max_value=50000.0,
    value=1200.0,
    step=50.0
)

bhk = st.number_input(
    "Number of BHK",
    min_value=1,
    max_value=20,
    value=2
)

bath = st.number_input(
    "Number of Bathrooms",
    min_value=1.0,
    max_value=20.0,
    value=2.0,
    step=1.0
)

balcony = st.number_input(
    "Number of Balconies",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=1.0
)

availability_group = st.selectbox(
    "Availability",
    ["ready_to_move", "date_or_year", "other"]
)

# Prediction button
if st.button("Predict House Price", type="primary"):

    if not location.strip():
        st.warning("Please enter a location.")
    else:

        # Prepare input in the same format as model training
        input_data = pd.DataFrame([{
            "area_type": area_type,
            "total_sqft": total_sqft,
            "bath": bath,
            "balcony": balcony,
            "location": location.strip(),
            "bhk": float(bhk),
            "sqft_per_bhk": total_sqft / bhk,
            "bath_per_bhk": bath / bhk,
            "availability_group": availability_group
        }])

        try:
            prediction = model.predict(input_data)[0]

            st.success("Prediction completed!")

            st.metric(
                label="Estimated House Price",
                value=f"₹ {prediction:.2f} Lakhs"
            )

            st.info(
                f"Estimated price: ₹ {prediction * 100000:,.0f}"
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

st.divider()

st.caption(
    "Bengaluru House Price Prediction | "
    "Machine Learning Project using Random Forest Regression"
)
