import streamlit as st
import pickle
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Exam Score Predictor",
    page_icon="📘",
    layout="centered"
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    with open("tuned_xgb_regressor.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ================= MANUAL ENCODING =================
sleep_quality_map = {"Low": 0, "Medium": 1, "High": 2}
study_method_map = {"Self Study": 0, "Group Study": 1, "Online": 2}
facility_rating_map = {"Low": 0, "Medium": 1, "High": 2}

# ================= CSS =================
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: white;
}
.main {
    padding: 2rem;
}
.app-card {
    background: linear-gradient(145deg, #020617, #020617);
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0 0 30px rgba(99,102,241,0.15);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.app-card:hover {
    transform: scale(1.015);
    box-shadow: 0 0 45px rgba(99,102,241,0.35);
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: 800;
    background: linear-gradient(90deg, #6366f1, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #6366f1, #22d3ee);
    color: black;
    font-size: 18px;
    font-weight: 700;
    border-radius: 12px;
    padding: 14px;
    transition: all 0.25s ease;
}
.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 25px rgba(34,211,238,0.6);
}
.result-box {
    margin-top: 25px;
    padding: 18px;
    border-radius: 14px;
    text-align: center;
    font-size: 26px;
    font-weight: 800;
    background: linear-gradient(90deg, #22c55e, #4ade80);
    color: black;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ================= JS =================
st.markdown("""
<script>
document.addEventListener("DOMContentLoaded", function () {
    const sliders = document.querySelectorAll("input[type='range']");
    sliders.forEach(slider => {
        slider.addEventListener("input", () => {
            slider.style.filter = "drop-shadow(0 0 6px #22d3ee)";
        });
    });
});
</script>
""", unsafe_allow_html=True)

# ================= UI =================
st.markdown("<div class='title'>📘 Exam Score Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Boosting Algorithm • XGBoost Regressor</div>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)

    study_hours = st.slider("📖 Study Hours (per day)", 0.0, 12.0, 4.0)
    class_attendance = st.slider("🏫 Class Attendance (%)", 0, 100, 75)
    sleep_hours = st.slider("😴 Sleep Hours", 0.0, 10.0, 7.0)

    sleep_quality = st.selectbox("🌙 Sleep Quality", ["Low", "Medium", "High"])
    study_method = st.selectbox("📚 Study Method", ["Self Study", "Group Study", "Online"])
    facility_rating = st.selectbox("🏫 Facility Rating", ["Low", "Medium", "High"])

    st.markdown("</div>", unsafe_allow_html=True)

# ================= PREDICTION =================
if st.button("🎯 Predict Exam Score"):
    try:
        final_input = np.array([[
            study_hours,
            class_attendance,
            sleep_hours,
            sleep_quality_map[sleep_quality],
            study_method_map[study_method],
            facility_rating_map[facility_rating]
        ]])

        prediction = model.predict(final_input)[0]

        st.markdown(
            f"<div class='result-box'>📊 Predicted Exam Score: {prediction:.2f}</div>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(e)

# ================= FOOTER =================
st.markdown("""
<hr>
<center style="color:#64748b;">
🚀 Machine Learning Project | Streamlit + XGBoost
</center>
""", unsafe_allow_html=True)
