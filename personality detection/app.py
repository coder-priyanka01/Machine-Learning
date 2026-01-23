import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Personality Predictor",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Load model and scaler
# -----------------------------
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# -----------------------------
# Feature columns
# -----------------------------
features = [
    'social_energy',
    'alone_time_preference',
    'talkativeness',
    'deep_reflection',
    'group_comfort',
    'party_liking',
    'listening_skill',
    'empathy',
    'organization',
    'leadership',
    'risk_taking',
    'public_speaking_comfort',
    'curiosity',
    'routine_preference',
    'excitement_seeking',
    'friendliness',
    'planning',
    'spontaneity',
    'adventurousness',
    'reading_habit',
    'sports_interest',
    'online_social_usage',
    'travel_desire',
    'gadget_usage',
    'work_style_collaborative',
    'decision_speed'
]

# -----------------------------
# Label mapping (numeric → category)
# -----------------------------
label_mapping = {
    0: "Introvert",
    1: "Extrovert",
    2: "Ambivert"
}

# -----------------------------
# Header
# -----------------------------
st.title("🧠 Personality Type Prediction")
st.caption("Rate the following traits to predict your personality category")

st.markdown("---")

# -----------------------------
# Input sliders (2 columns)
# -----------------------------
input_data = {}
col1, col2 = st.columns(2)

for i, feature in enumerate(features):
    with col1 if i % 2 == 0 else col2:
        input_data[feature] = st.slider(
            feature.replace("_", " ").title(),
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=1.0
        )

st.markdown("---")

# -----------------------------
# Prediction button centered
# -----------------------------
center_col1, center_col2, center_col3 = st.columns([1,2,1])
with center_col2:
    predict_btn = st.button("🔮 Predict Personality", use_container_width=True)

# -----------------------------
# Prediction output
# -----------------------------
if predict_btn:
    input_df = pd.DataFrame([input_data])
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)  # numeric label
    probability = model.predict_proba(scaled_input)

    predicted_label = prediction[0]
    predicted_text = label_mapping.get(predicted_label, "Unknown")

    # Show category
    st.subheader("✅ Prediction Result")
    st.success(f"Predicted Personality Category: **{predicted_text}**")

    # Show probability table
    st.subheader("📊 Prediction Probability")
    proba_df = pd.DataFrame(probability, columns=[label_mapping[i] for i in range(len(probability[0]))])
    st.dataframe(proba_df, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: #9FA6B2;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        z-index: 100;
    }
    </style>

    <div class="footer">
        🧠 Personality Prediction App | Built with Streamlit & Logistic Regression
    </div>
    """,
    unsafe_allow_html=True
)
