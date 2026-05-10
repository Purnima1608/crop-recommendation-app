# ==============================
# IMPORTS
# ==============================
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import datetime
import matplotlib.pyplot as plt
import matplotlib
import requests

# ==============================
# GOOGLE SHEET WEB APP URL
# ==============================
WEB_APP_URL = "https://script.google.com/macros/s/AKfycbw2fwfs3VYVy4z-JjPFeQQUDDkZ6r3f2KFi_rRpYuD6Vhfu5-R2KHjjMP2wgraCDcQcdA/exec"

# ==============================
# FONT
# ==============================
matplotlib.rcParams['font.family'] = 'Mangal'

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

/* Sidebar */
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

/* Labels */
label {
    font-size: 18px !important;
    font-weight: 600;
}

/* Number Input */
.stNumberInput input {
    font-size: 16px !important;
    border-radius: 8px;
}

/* Slider */
.stSlider span {
    font-size: 16px !important;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("crop_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# ==============================
# SESSION
# ==============================
if "lang" not in st.session_state:
    st.session_state.lang = "English"

if "last_user" not in st.session_state:
    st.session_state.last_user = None

if "show_feedback_form" not in st.session_state:
    st.session_state.show_feedback_form = False

# ==============================
# LANGUAGE TEXTS
# ==============================
texts = {

    "English": {
        "title": "🌾 Crop Recommendation System",
        "button": "🌱 Get Recommendation",
        "result": "🌾 Top 3 Recommendations",
        "correct": "✅ Correct",
        "incorrect": "❌ Wrong",
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
        "Specify Reason": "Please tell reason",
        "other_label": "Please specify:",
        "submit": "Submit Feedback",

        "reasons": [
            "Wrong Soil Data",
            "Due to Weather Change",
            "Due to Fertilizer",
            "Farmer Preference",
            "Quality of seed",
            "Due to Water Issue",
            "Due to Heavy Rainfall",
            "Other"
        ]
    },

    "Hindi": {
        "title": "🌾 फसल सिफारिश प्रणाली",
        "button": "🌱 सिफारिश देखें",
        "result": "🌾 शीर्ष 3 सिफारिशें",
        "correct": "✅ सही",
        "incorrect": "❌ गलत",
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
        "Specify Reason": "⚠️ कारण बताएं",
        "other_label": "कृपया बताएं:",
        "submit": "फीडबैक सबमिट करें",

        "reasons": [
            "गलत मिट्टी डेटा",
            "मौसम परिवर्तन",
            "उर्वरक के कारण",
            "किसान की पसंद",
            "बीज की गुणवत्ता",
            "पानी की समस्या",
            "अत्यधिक वर्षा",
            "अन्य"
        ]
    },

    "Gujarati": {
        "title": "🌾 પાક ભલામણ પ્રણાલી",
        "button": "🌱 ભલામણ જુઓ",
        "result": "🌾 ટોચની 3 ભલામણ",
        "correct": "✅ સાચું",
        "incorrect": "❌ ખોટું",
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
        "Specify Reason": "કારણ આપો",
        "other_label": "કૃપા કરીને લખો:",
        "submit": "પ્રતિસાદ મોકલો",

        "reasons": [
            "ખોટું જમીન ડેટા",
            "હવામાન બદલાવ",
            "ખાતર કારણે",
            "ખેડૂત પસંદગી",
            "બીજની ગુણવત્તા",
            "પાણીની સમસ્યા",
            "ભારે વરસાદ",
            "અન્ય"
        ]
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
# ANALYTICS
# ==============================
try:

    response = requests.get(WEB_APP_URL)
    records = response.json()

    df = pd.DataFrame(records)

    total = len(df)

    correct = 0
    wrong = 0

    if total > 0:
        correct = (df["Feedback"] == "Correct").sum()
        wrong = (df["Feedback"] == "Wrong").sum()

    col1, col2 = st.columns([1,1])

    with col1:
        st.metric(texts[lang]["pred"], total)
        st.metric(texts[lang]["correct_count"], correct)
        st.metric(texts[lang]["wrong_count"], wrong)

    with col2:

        fig, ax = plt.subplots(figsize=(3,3))

        if correct == 0 and wrong == 0:

            ax.text(
                0.5,
                0.5,
                "No Feedback",
                ha='center',
                va='center'
            )

            ax.axis("off")

        else:

            ax.pie(
                [correct, wrong],
                labels=[
                    texts[lang]["correct"],
                    texts[lang]["incorrect"]
                ],
                autopct="%1.1f%%",
                colors=["green", "red"]
            )

        st.pyplot(fig)

except Exception as e:
    st.error(f"Analytics Error: {e}")

# ==============================
# DUAL INPUT
# ==============================
def dual_input(label, key, min_v, max_v, default):

    if key not in st.session_state:
        st.session_state[key] = default

    if f"{key}_slider" not in st.session_state:
        st.session_state[f"{key}_slider"] = default

    if f"{key}_input" not in st.session_state:
        st.session_state[f"{key}_input"] = default

    def sync_from_slider():
        st.session_state[f"{key}_input"] = st.session_state[f"{key}_slider"]

    def sync_from_input():
        st.session_state[f"{key}_slider"] = st.session_state[f"{key}_input"]

    col1, col2 = st.columns([2,1])

    col1.slider(
        label,
        min_v,
        max_v,
        key=f"{key}_slider",
        on_change=sync_from_slider
    )

    col2.number_input(
        "",
        min_v,
        max_v,
        key=f"{key}_input",
        on_change=sync_from_input
    )

    return st.session_state[f"{key}_slider"]

# ==============================
# INPUTS
# ==============================
st.subheader(texts[lang]["soil"])

N = dual_input(texts[lang]["nitrogen"], "N", 0.0, 140.0, 50.0)
P = dual_input(texts[lang]["phosphorus"], "P", 0.0, 145.0, 50.0)
K = dual_input(texts[lang]["potassium"], "K", 0.0, 205.0, 50.0)

Zinc = dual_input(texts[lang]["zinc"], "Zn", 0.5, 5.0, 2.0)
Iron = dual_input(texts[lang]["iron"], "Fe", 2.0, 25.0, 10.0)
Boron = dual_input(texts[lang]["boron"], "B", 0.1, 2.0, 1.0)
Manganese = dual_input(texts[lang]["manganese"], "Mn", 1.0, 15.0, 5.0)
Copper = dual_input(texts[lang]["copper"], "Cu", 0.1, 2.0, 1.0)

st.subheader(texts[lang]["weather"])

pH = dual_input(texts[lang]["ph"], "pH", 4.0, 9.0, 7.0)
Temp = dual_input(texts[lang]["temp"], "Temp", 10.0, 45.0, 25.0)
Rain = dual_input(texts[lang]["rain"], "Rain", 20.0, 300.0, 100.0)
Hum = dual_input(texts[lang]["humidity"], "Hum", 20.0, 100.0, 50.0)

# ==============================
# PREDICTION
# ==============================
if st.button(texts[lang]["button"]):

    try:
        response = requests.get(WEB_APP_URL)
        records = response.json()
        user_id = f"User_{len(records) + 1}"
    except:
        user_id = "User_67"

    data_input = pd.DataFrame(
        [[N,P,K,Zinc,Iron,Boron,Manganese,Copper,pH,Temp,Rain,Hum]],
        columns=scaler.feature_names_in_
    )

    scaled = scaler.transform(data_input)

    probs = model.predict_proba(scaled)[0]

    idx = np.argsort(probs)[-3:][::-1]

    crops = le.inverse_transform(idx)

    top_prob = probs[idx[0]]

    percentages = [(p/top_prob)*100 for p in probs[idx]]

    st.success(texts[lang]["result"])

    crop_names = [translate_crop(c) for c in crops]

    perc_values = [round(p,1) for p in percentages]

   # ==============================
# GRAPH
# ==============================
fig, ax = plt.subplots(figsize=(9, 4))

bars = ax.barh(
    crop_names,
    perc_values,
    color=["green", "blue", "orange"]
)

# Title
ax.set_title("Top Crop Recommendations", fontsize=16)

# X Label
ax.set_xlabel("Confidence (%)", fontsize=12)

# Limit graph width
ax.set_xlim(0, 110)

# Add percentage labels INSIDE graph
for i, v in enumerate(perc_values):

    ax.text(
        min(v + 1, 102),   # prevent overflow
        i,
        f"{v:.1f}%",
        va='center',
        fontsize=11,
        fontweight='bold'
    )

# Highest recommendation on top
ax.invert_yaxis()

# Improve Hindi/Gujarati visibility
plt.tight_layout()

# Extra left spacing for regional language text
plt.subplots_adjust(left=0.28)

st.pyplot(fig)

    # ==============================
    # SAVE TO GOOGLE SHEET
    # ==============================
    save_data = {

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
        "Reason": "",
        "Time": str(datetime.datetime.now())
    }

    requests.post(WEB_APP_URL, json=save_data)

    st.session_state.last_user = user_id

# ==============================
# FEEDBACK
# ==============================
# FEEDBACK
# ==============================
st.divider()

st.markdown(f"## {texts[lang]['feedback']}")

if st.session_state.last_user:

    col1, col2 = st.columns(2)

    # ==============================
    # CORRECT BUTTON
    # ==============================
    if col1.button(texts[lang]["correct"]):

        feedback_data = {

            "User_ID": st.session_state.last_user,
            "Feedback": "Correct",
            "Reason": ""

        }

        requests.post(WEB_APP_URL, json=feedback_data)

        st.success("Feedback Stored ✅")

        st.session_state.show_feedback_form = False

        st.rerun()

    # ==============================
    # WRONG BUTTON
    # ==============================
    if col2.button(texts[lang]["incorrect"]):

        st.session_state.show_feedback_form = True

# ==============================
# FEEDBACK FORM
# ==============================
if st.session_state.show_feedback_form:

    st.warning(texts[lang]["Specify Reason"])

    reasons = texts[lang]["reasons"]

    selected = []

    for i, reason in enumerate(reasons):

        if st.checkbox(reason, key=f"reason_{i}"):

            selected.append(reason)

    # Other textbox
    if reasons[-1] in selected:

        other_text = st.text_input(texts[lang]["other_label"])

        if other_text:
            selected.append(other_text)

    # ==============================
    # SUBMIT BUTTON
    # ==============================
    if st.button(texts[lang]["submit"]):

        feedback_data = {

            "User_ID": st.session_state.last_user,
            "Feedback": "Wrong",
            "Reason": ", ".join(selected)

        }

        requests.post(WEB_APP_URL, json=feedback_data)

        st.success("Feedback Recorded ✅")

        st.session_state.show_feedback_form = False

        st.rerun()

else:

    st.info("Make prediction first")
