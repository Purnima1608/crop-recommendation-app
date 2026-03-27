import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ==============================
# Page Config (MUST BE FIRST)
# ==============================
st.set_page_config(page_title="Crop Recommendation System", layout="centered")

# ==============================
# Load Model Files
# ==============================
model = joblib.load("crop_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# ==============================
# Language Selection
# ==============================
if "lang" not in st.session_state:
    st.session_state.lang = "English"

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("English"):
        st.session_state.lang = "English"

with col2:
    if st.button("हिंदी"):
        st.session_state.lang = "Hindi"

with col3:
    if st.button("ગુજરાતી"):
        st.session_state.lang = "Gujarati"

lang = st.session_state.lang

# ==============================
# Language Dictionary
# ==============================

texts = {

    "English": {
        "title": "🌾 Crop Recommendation System",
        "subtitle": "🌱 Machine Learning Based Smart Farming Solution",
        "soil": " Soil Nutrients",
        "weather": "🌦️ Weather Information",
        "nitrogen": "Nitrogen (kg/ha)",
        "phosphorus": "Phosphorus (kg/ha)",
        "potassium": "Potassium (kg/ha)",
        "zinc": "Zinc (mg/kg)",
        "iron": "Iron (mg/kg)",
        "boron": "Boron (mg/kg)",
        "manganese": "Manganese (mg/kg)",
        "copper": "Copper (mg/kg)",
        "ph": "Soil pH",
        "temp": "Temperature (°C)",
        "rain": "Rainfall (mm)",
        "humidity": "Humidity (%)",
        "button": "🌱 Get Crop Recommendation",
        "result": "🌾 Top 3 Crop Recommendations",
        "first": " First Recommendation",
        "second": "Second Recommendation",
        "third": "Third Recommendation"
    },

    "Hindi": {
        "title": "🌾 फसल सिफारिश प्रणाली",
        "subtitle": "🌱 मशीन लर्निंग आधारित स्मार्ट कृषि समाधान",
        "soil": " मिट्टी के पोषक तत्व",
        "weather": "🌦️ मौसम की जानकारी",
        "nitrogen": "नाइट्रोजन (kg/ha)",
        "phosphorus": "फॉस्फोरस (kg/ha)",
        "potassium": "पोटैशियम (kg/ha)",
        "zinc": "जिंक (mg/kg)",
        "iron": "आयरन (mg/kg)",
        "boron": "बोरॉन (mg/kg)",
        "manganese": "मैंगनीज (mg/kg)",
        "copper": "कॉपर (mg/kg)",
        "ph": "मिट्टी का pH",
        "temp": "तापमान (°C)",
        "rain": "वर्षा (mm)",
        "humidity": "नमी (%)",
        "button": "🌱 फसल की सिफारिश देखें",
        "result": "🌾 शीर्ष 3 फसल सिफारिशें",
        "first": "🥇 पहली अनुशंसा",
        "second": "🥈 दूसरी अनुशंसा",
        "third": "🥉 तीसरी अनुशंसा"
    },

    "Gujarati": {
        "title": "🌾 પાક ભલામણ પ્રણાલી",
        "subtitle": "🌱 મશીન લર્નિંગ આધારિત સ્માર્ટ ખેતી ઉકેલ",
        "soil": " જમીનના પોષક તત્વો",
        "weather": "🌦️ હવામાન માહિતી",
        "nitrogen": "નાઈટ્રોજન (kg/ha)",
        "phosphorus": "ફોસ્ફરસ (kg/ha)",
        "potassium": "પોટેશિયમ (kg/ha)",
        "zinc": "ઝીંક (mg/kg)",
        "iron": "આયર્ન (mg/kg)",
        "boron": "બોરોન (mg/kg)",
        "manganese": "મેંગેનીઝ (mg/kg)",
        "copper": "કોપર (mg/kg)",
        "ph": "જમીન pH",
        "temp": "તાપમાન (°C)",
        "rain": "વરસાદ (mm)",
        "humidity": "ભેજ (%)",
        "button": "🌱 પાક ભલામણ જુઓ",
        "result": "🌾 ટોચની 3 પાક ભલામણ",
        "first": "🥇 પ્રથમ ભલામણ",
        "second": "🥈 બીજી ભલામણ",
        "third": "🥉 ત્રીજી ભલામણ"
    }
}

# ==============================
# Crop Name Translations
# ==============================

crop_translations = {

    "Cotton": {"Hindi": "कपास", "Gujarati": "કપાસ"},
    "Maize": {"Hindi": "मक्का", "Gujarati": "મકાઈ"},
    "Groundnut": {"Hindi": "मूंगफली", "Gujarati": "મગફળી"},
    "Wheat": {"Hindi": "गेहूं", "Gujarati": "ઘઉં"},
    "Rice": {"Hindi": "चावल", "Gujarati": "ચોખા"}

}

def translate_crop(crop_name, lang):
    if lang == "English":
        return crop_name
    return crop_translations.get(crop_name, {}).get(lang, crop_name)

# ==============================
# UI
# ==============================

st.title(texts[lang]["title"])
st.markdown(f"### {texts[lang]['subtitle']}")
st.divider()

st.subheader(texts[lang]["soil"])

N = st.slider(texts[lang]["nitrogen"], 0.0, 200.0, 50.0)
P = st.slider(texts[lang]["phosphorus"], 0.0, 200.0, 50.0)
K = st.slider(texts[lang]["potassium"], 0.0, 200.0, 50.0)

Zinc = st.slider(texts[lang]["zinc"], 1.0, 5.0, 3.0, step=0.01)
Iron = st.slider(texts[lang]["iron"], 10.0, 30.0, 20.0, step=0.01)
Boron = st.slider(texts[lang]["boron"], 0.2, 2.0, 1.1, step=0.01)
Manganese = st.slider(texts[lang]["manganese"], 5.0, 20.0, 12.5, step=0.01)
Copper = st.slider(texts[lang]["copper"], 0.1, 2.0, 1.0, step=0.01)

st.divider()

st.subheader(texts[lang]["weather"])

pH = st.slider(texts[lang]["ph"], 0.0, 14.0, 7.0, step=0.1)
Temperature = st.slider(texts[lang]["temp"], 0.0, 50.0, 25.0)
Rainfall = st.slider(texts[lang]["rain"], 0.0, 500.0, 100.0)
Humidity = st.slider(texts[lang]["humidity"], 0.0, 100.0, 50.0)

st.divider()

# ==============================
# Prediction
# ==============================

if st.button(texts[lang]["button"]):

    feature_names = scaler.feature_names_in_

    input_dict = {
        'Nitrogen (N) (kg/ha)': N,
        'Phosphorus (P) (kg/ha)': P,
        'Potassium (K) (kg/ha)': K,
        'Zinc (Zn) (mg/kg)': Zinc,
        'Iron (Fe) (mg/kg)': Iron,
        'Boron (B) (mg/kg)': Boron,
        'Manganese (Mn) (mg/kg)': Manganese,
        'Copper (Cu) (mg/kg)': Copper,
        'Soil pH': pH,
        'Temperature (°C)': Temperature,
        'Rainfall (mm)': Rainfall,
        'Humidity (%)': Humidity
    }

    input_data = pd.DataFrame(
        [[input_dict[col] for col in feature_names]],
        columns=feature_names
    )

    input_scaled = scaler.transform(input_data)

    probabilities = model.predict_proba(input_scaled)[0]
    top3_indices = np.argsort(probabilities)[-3:][::-1]
    top3_crops = le.inverse_transform(top3_indices)

    # Translate crops
    crop1 = translate_crop(top3_crops[0], lang)
    crop2 = translate_crop(top3_crops[1], lang)
    crop3 = translate_crop(top3_crops[2], lang)

    st.success(texts[lang]["result"])

    st.write(f"{texts[lang]['first']}: {crop1}")
    st.write(f"{texts[lang]['second']}: {crop2}")
    st.write(f"{texts[lang]['third']}: {crop3}")