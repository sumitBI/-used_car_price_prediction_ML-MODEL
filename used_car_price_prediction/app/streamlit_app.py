"""Streamlit interface for historical used-car price estimates."""
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.predict import load_model, predict_price

st.set_page_config(page_title="Used Car Price Estimator", page_icon="🚗")
st.title("🚗 Used Car Price Estimator")
st.info("Trained on historical listings (1992–2020); not a live 2026 market quote.")

@st.cache_resource
def get_model():
    return load_model()

try:
    model = get_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

with st.form("car_form"):
    year = st.number_input("Manufacturing year", min_value=1992, max_value=2020, value=2015)
    km_driven = st.number_input("Kilometres driven", min_value=0, value=50000, step=1000)
    fuel = st.selectbox("Fuel", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    seller_type = st.selectbox("Seller type", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Ownership", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
    submitted = st.form_submit_button("Estimate selling price")

if submitted:
    price = predict_price(model, year=year, km_driven=km_driven, fuel=fuel,
                          seller_type=seller_type, transmission=transmission, owner=owner)
    st.metric("Estimated historical selling price", f"₹{price:,.0f}")
    st.caption("Indicative model estimate; actual prices depend on vehicle condition, location, and current market demand.")
