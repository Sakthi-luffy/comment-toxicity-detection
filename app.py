import streamlit as st
import sys
sys.path.append("src")

from tensorflow.keras.models import load_model
from preprocessing import clean_text, tokenize_and_pad

# Load model
model = load_model("saved_models/toxicity_model.keras")

st.title("💬 Comment Toxicity Detector")

user_input = st.text_area("Enter a comment")

if st.button("Predict"):
    cleaned = [clean_text(user_input)]
    padded, _ = tokenize_and_pad(cleaned)

    prediction = model.predict(padded)[0]

    labels = [
        "Toxic",
        "Severe Toxic",
        "Obscene",
        "Threat",
        "Insult",
        "Identity Hate"
    ]

    st.subheader("Prediction Results")

    for label, score in zip(labels, prediction):
        st.write(f"{label}: {score:.2f}")
