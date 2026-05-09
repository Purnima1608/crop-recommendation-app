import streamlit as st
import numpy as np
import pandas as pd
import joblib
import datetime
import os
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Crop Recommendation", layout="centered")
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #f5f7fa;
}

/* Sidebar (if you use it later) */
[data-testid="stSidebar"] {
    background-color: #e6f0ff;
}

/* Titles */
h1, h2, h3 {
    color: #2c3e50;
}

/* Buttons */
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
}

/* Input boxes */
.stNumberInput input {
label {
    font-size: 18px !important;
    font-weight: 600;
}

/* Increase number input text */
.stNumberInput input {
    font-size: 16px !important;
}

/* Increase slider value text */
.stSlider span {
    font-size: 16px !important;
}    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)
model = joblib.load("crop_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

log_file = "user_data1.csv"

# ==============================
# SESSION
# ==============================
if "lang" not in st.session_state:
    st.session_state.lang = "English"

if "last_user" not in st.session_state:
    st.session_state.last_user = None

# ==============================
# LANGUAGE
# ==============================
texts = {
    "English": {
        "title": "🌾 Crop Recommendation System",
        "button": "🌱 Get Recommendation",
        "result": "🌾 Top 3 Recommendations",
        "correct": "✅ Correct",
        "incorrect": "❌ Wrong",
        "usage": "📊 System Analytics",

        "dashboard": "📊 Dashboard",
        "pred": "Predictions",
        "correct_count": "Correct",
        "wrong_count": "Wrong",

        "soil": "Soil",
        "weather": "Weather",

        "nitrogen": "Nitrogen",
        "phosphorus": "Phosphorus",
        "potassium": "Potassium",
        "zinc": "Zinc",
        "iron": "Iron",
        "boron": "Boron",
        "manganese": "Manganese",
        "copper": "Copper",

        "ph": "pH",
        "temp": "Temperature",
        "rain": "Rainfall",
        "humidity": "Humidity",

        "feedback": "👍 Feedback",
        "first": " First Choice",
        "second": " Second Choice",
        "third": "Third Choice",

        "Specify Reason": " Please tell reason",
        
        "reasons": [
            "Wrong Soil Data",
            "Due to Weather Change",
            "Due to Fertilizer",
            "Farmer Preference",
            "Quality of seed",
            "Due to Water Issue",
            "Due to Heavy Rainfall",
            "Other"
        ],
        "other_label": "Please specify:",
        "submit": "Submit Feedback"
    },

    "Hindi": {
        "title": "🌾 फसल सिफारिश प्रणाली",
        "button": "🌱 सिफारिश देखें",
        "result": "🌾 शीर्ष 3 सिफारिशें",
        "correct": "✅ सही",
        "incorrect": "❌ गलत",
        "usage": "📊 विश्लेषण",

        "dashboard": "📊 डैशबोर्ड",
        "pred": "कुल भविष्यवाणी",
        "correct_count": "सही",
        "wrong_count": "गलत",

        "soil": "मिट्टी",
        "weather": "मौसम",

        "nitrogen": "नाइट्रोजन",
        "phosphorus": "फॉस्फोरस",
        "potassium": "पोटैशियम",
        "zinc": "जिंक",
        "iron": "आयरन",
        "boron": "बोरॉन",
        "manganese": "मैंगनीज",
        "copper": "कॉपर",

        "ph": "pH",
        "temp": "तापमान",
        "rain": "वर्षा",
        "humidity": "नमी",

        "feedback": "👍 फीडबैक",
        "first": " पहला",
        "second": " दूसरा",
        "third": " तीसरा",

        "Specify Reason": "⚠️ कारण बताएं" ,
            "reasons": [
            "गलत मिट्टी डेटा",
            "मौसम परिवर्तन",
            "उर्वरक के कारण",
            "किसान की पसंद",
            "बीज की गुणवत्ता",
            "पानी की समस्या",
            "अत्यधिक वर्षा",
            "अन्य"
        ],
        "other_label": "कृपया बताएं:",
        "submit": "फीडबैक सबमिट करें"
    },

    "Gujarati": {
        "title": "🌾 પાક ભલામણ પ્રણાલી",
        "button": "🌱 ભલામણ જુઓ",
        "result": "🌾 ટોચની 3 ભલામણ",
        "correct": "✅ સાચું",
        "incorrect": "❌ ખોટું",
        "usage": "📊 વિશ્લેષણ",

        "dashboard": "📊 ડેશબોર્ડ",
        "pred": "કુલ આગાહી",
        "correct_count": "સાચું",
        "wrong_count": "ખોટું",

        "soil": "જમીન",
        "weather": "હવામાન",

        "nitrogen": "નાઈટ્રોજન",
        "phosphorus": "ફોસ્ફરસ",
        "potassium": "પોટેશિયમ",
        "zinc": "ઝીંક",
        "iron": "આયર્ન",
        "boron": "બોરોન",
        "manganese": "મેંગેનીઝ",
        "copper": "કોપર",

        "ph": "pH",
        "temp": "તાપમાન",
        "rain": "વરસાદ",
        "humidity": "ભેજ",

        "feedback": "👍 પ્રતિસાદ",
        "first": " પ્રથમ",
        "second": " બીજું",
        "third": " ત્રીજું",

        "Specify Reason": "કારણ આપો" , 
            "reasons": [
            "ખોટું જમીન ડેટા",
            "હવામાન બદલાવ",
            "ખાતર કારણે",
            "ખેડૂત પસંદગી",
            "બીજની ગુણવત્તા",
            "પાણીની સમસ્યા",
            "ભારે વરસાદ",
            "અન્ય"
        ],
        "other_label": "કૃપા કરીને લખો:",
        "submit": "પ્રતિસાદ મોકલો"
    }
}

# ==============================
# CROP TRANSLATION
# ==============================
crop_lang = {
    "Cotton": {"Hindi": "कपास", "Gujarati": "કપાસ"},
    "Maize": {"Hindi": "मक्का", "Gujarati": "મકાઈ"},
    "Groundnut": {"Hindi": "मूंगफली", "Gujarati": "મગફળી"},
    "Wheat": {"Hindi": "गेहूं", "Gujarati": "ઘઉં"},
    "Rice": {"Hindi": "चावल", "Gujarati": "ચોખા"}
}

def translate_crop(c):
    if st.session_state.lang == "English":
        return c
    return crop_lang.get(c, {}).get(st.session_state.lang, c)

# ==============================
# LANGUAGE BUTTONS
# ==============================
c1, c2, c3 = st.columns(3)

if c1.button("English"):
    st.session_state.lang = "English"
    st.rerun()

if c2.button("हिंदी"):
    st.session_state.lang = "Hindi"
    st.rerun()

if c3.button("ગુજરાતી"):
    st.session_state.lang = "Gujarati"
    st.rerun()

lang = st.session_state.lang

# ==============================
# TITLE
# ==============================
st.title(texts[lang]["title"])

# ==============================
# ANALYTICS SECTION
# ==============================

# ==============================
# SYSTEM ANALYTICS (SIDE BY SIDE)
# ==============================


# ==============================
# SYSTEM ANALYTICS
# ==============================

if os.path.exists(log_file):

    df = pd.read_csv(log_file)

    total = len(df)
    correct = (df["Feedback"] == "Correct").sum()
    wrong = (df["Feedback"] == "Wrong").sum()

    # Create 2 columns
    col1, col2 = st.columns([1,1])

    # LEFT SIDE → Metrics
    with col1:
        st.metric(texts[lang]["pred"], total)
        st.metric(texts[lang]["correct_count"], correct)
        st.metric(texts[lang]["wrong_count"], wrong)

    # RIGHT SIDE → Pie Chart
    with col2:

        fig, ax = plt.subplots(figsize=(3,3))

        # Prevent error when all values are zero
        if correct == 0 and wrong == 0:

            ax.text(
                0.5,
                0.5,
                "No Feedback Data",
                ha='center',
                va='center',
                fontsize=12
            )

            ax.axis('off')

        else:

            ax.pie(
                [correct, wrong],
                labels=["Correct", "Wrong"],
                autopct="%1.1f%%",
                colors=["green", "red"],
                wedgeprops={'edgecolor': 'white'}
            )

        st.pyplot(fig)

else:
    st.info("No data yet")

    

# ==============================
# INPUT FUNCTION
# ==============================
def dual_input(label, key, min_v, max_v, default):

    # Main session value
    if key not in st.session_state:
        st.session_state[key] = default

    # Slider session
    if f"{key}_slider" not in st.session_state:
        st.session_state[f"{key}_slider"] = default

    # Input session
    if f"{key}_input" not in st.session_state:
        st.session_state[f"{key}_input"] = default

    # When slider changes
    def sync_from_slider():
        st.session_state[key] = st.session_state[f"{key}_slider"]
        st.session_state[f"{key}_input"] = st.session_state[f"{key}_slider"]

    # When textbox changes
    def sync_from_input():
        st.session_state[key] = st.session_state[f"{key}_input"]
        st.session_state[f"{key}_slider"] = st.session_state[f"{key}_input"]

    col1, col2 = st.columns([2,1])

    # Slider
    col1.slider(
        label,
        min_v,
        max_v,
        key=f"{key}_slider",
        on_change=sync_from_slider
    )

    # Textbox
    col2.number_input(
        "",
        min_v,
        max_v,
        key=f"{key}_input",
        on_change=sync_from_input
    )

    return st.session_state[key]
# ==============================
# INPUTS
# ==============================
st.subheader(texts[lang]["soil"])

N = dual_input(texts[lang]["nitrogen"], "N", 0.0, 140.0, 20.0)
P = dual_input(texts[lang]["phosphorus"], "P", 0.0, 60.0, 20.0)
K = dual_input(texts[lang]["potassium"], "K", 0.0, 205.0, 5.0)

Zinc = dual_input(texts[lang]["zinc"], "Zn", 1.0, 5.0, 3.0)
Iron = dual_input(texts[lang]["iron"], "Fe", 1.0, 30.0, 20.0)
Boron = dual_input(texts[lang]["boron"], "B", 0.2, 2.0, 1.0)
Manganese = dual_input(texts[lang]["manganese"], "Mn", 2.0, 20.0, 12.0)
Copper = dual_input(texts[lang]["copper"], "Cu", 0.1, 2.0, 1.0)

st.subheader(texts[lang]["weather"])

pH = dual_input(texts[lang]["ph"], "pH", 0.0, 10.0, 4.0)
Temp = dual_input(texts[lang]["temp"], "Temp", 0.0, 50.0, 25.0)
Rain = dual_input(texts[lang]["rain"], "Rain", 0.0, 2500.0, 200.0)
Hum = dual_input(texts[lang]["humidity"], "Hum", 0.0, 100.0, 50.0)

# ==============================
# ==============================
# PREDICTION
# ==============================
if st.button(texts[lang]["button"]):

    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        user_id = f"User_{len(df)+1}"
    else:
        user_id = "User_1"

    data = pd.DataFrame([[N,P,K,Zinc,Iron,Boron,Manganese,Copper,pH,Temp,Rain,Hum]],
                        columns=scaler.feature_names_in_)

    scaled = scaler.transform(data)

    probs = model.predict_proba(scaled)[0]
    idx = np.argsort(probs)[-3:][::-1]
    crops = le.inverse_transform(idx)

    top_prob = probs[idx[0]]
    percentages = [(p/top_prob)*100 for p in probs[idx]]

    st.success(texts[lang]["result"])

    # ✅ CREATE VARIABLES (YOU MISSED THIS)
    crop_names = [translate_crop(c) for c in crops]
    perc_values = [round(p, 1) for p in percentages]

    # ✅ BAR CHART (MUST BE INSIDE BUTTON BLOCK)
    fig, ax = plt.subplots(figsize=(10,4))

    bars = ax.barh(crop_names, perc_values, 
                   color=["green", "blue", "orange"])

    ax.set_xlabel("Confidence (%)")
    ax.set_title("Top Crop Recommendations")

    # Show percentage on bars
    for i, v in enumerate(perc_values):
        ax.text(v + 2, i, f"{v}%", va='center')

    ax.invert_yaxis()

    st.pyplot(fig)



    new = pd.DataFrame([{
        "User_ID": user_id,
        "Nitrogen": N,
        "Phosphorus": P,
        "Potassium": K,
        "Zinc": Zinc,
        "Iron": Iron,
        "Boron": Boron,
        "Manganese": Manganese,
        "Copper": Copper,
        "pH": pH,
        "Temperature": Temp,
        "Rainfall": Rain,
        "Humidity": Hum,
        "Prediction1": crops[0],
        "Prediction2": crops[1],
        "Prediction3": crops[2],
        "Feedback": "Pending",
        "Time": datetime.datetime.now()
    }])

    if os.path.exists(log_file):
        old = pd.read_csv(log_file)
        new = pd.concat([old, new], ignore_index=True)

    new.to_csv(log_file, index=False)

    st.session_state.last_user = user_id

# ==============================
# FEEDBACK
# ==============================
# ==============================
# FEEDBACK SECTION
# ==============================
# ==============================
# FEEDBACK SECTION
# ==============================

st.divider()
st.markdown(f"## {texts[lang]['feedback']}")

# ✅ MUST be at top (only once in full app)
if "show_feedback_form" not in st.session_state:
    st.session_state.show_feedback_form = False

if st.session_state.last_user:

    col1, col2 = st.columns(2)

    # ✅ CORRECT BUTTON
    if col1.button(texts[lang]["correct"], key="correct_btn"):
        df = pd.read_csv(log_file)
        df.loc[df["User_ID"] == st.session_state.last_user, "Feedback"] = "Correct"
        df.to_csv(log_file, index=False)

        st.success("Feedback Stored ✅")
        st.session_state.show_feedback_form = False
        st.rerun()

    # ✅ INCORRECT BUTTON
    if col2.button(texts[lang]["incorrect"], key="wrong_btn"):
        st.session_state.show_feedback_form = True

# ✅ SHOW FORM (OUTSIDE button block)
if st.session_state.show_feedback_form:

    st.warning(texts[lang]["Specify Reason"])

    reasons = texts[lang]["reasons"]

    selected = []

    for i, reason in enumerate(reasons):
        if st.checkbox(reason, key=f"reason_{i}"):
            selected.append(reason)

    # ✅ FIX: MOVE THIS INSIDE
    other_text = ""
    if reasons[-1] in selected:
        other_text = st.text_input(texts[lang]["other_label"], key="other_text")
        if other_text:
            selected.append(other_text)

    reason_list = selected

    # ✅ SUBMIT BUTTON
    if st.button(texts[lang]["submit"], key="submit_btn"):

        if len(reason_list) == 0:
            st.error("Please select at least one reason")
        else:
            df = pd.read_csv(log_file)

            df.loc[df["User_ID"] == st.session_state.last_user, "Feedback"] = "Wrong"
            df.loc[df["User_ID"] == st.session_state.last_user, "Reason"] = ", ".join(reason_list)

            df.to_csv(log_file, index=False)

            st.success("Feedback Recorded ✅")

            st.session_state.show_feedback_form = False
            st.rerun()

else:
    st.info("Make prediction first")
