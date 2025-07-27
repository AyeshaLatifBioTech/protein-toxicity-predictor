import streamlit as st
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

# 🔧 Define the custom transformer used during vectorization
def kmer_analyzer(sequence, k=3):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# 🌟 Load model and vectorizer with custom function
model = joblib.load("toxicity_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# 🌐 App configuration
st.set_page_config(page_title="Protein Toxicity Predictor", layout="centered")

# 🎨 Styling and Title
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #800000;'>🧬 Protein Toxicity Prediction Tool</h1>
        <p style='font-size: 17px;'>Paste your protein sequence to check if it is toxic or non-toxic.</p>
    </div>
""", unsafe_allow_html=True)

# 📷 Logo (if available)
st.image("logo.png", width=120)

# ✍️ Input box
sequence = st.text_area("Enter Protein Sequence:", height=150)

# 🔮 Prediction logic
if st.button("🔍 Predict Toxicity"):
    if not sequence.strip():
        st.warning("⚠️ Please enter a protein sequence.")
    else:
        try:
            X_input = vectorizer.transform([sequence])
            prediction = model.predict(X_input)[0]
            prob = model.predict_proba(X_input)[0][1]

            label = "🧪 Toxic" if prediction == 1 else "✅ Non-Toxic"
            st.markdown(f"### 🧾 Prediction: **{label}**")
            st.info(f"🔬 Probability of being toxic: **{prob:.2%}**")

            # Optional: Add a progress bar
            st.progress(int(prob * 100))
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# 🧾 Footer / About
st.markdown("""
    <hr>
    <div style='text-align: center; font-size: 14px;'>
        <p><strong>About:</strong> This ML-powered tool analyzes protein sequences using k-mer features to predict toxicity. Developed by <em>Ayesha Latif</em>.</p>
    </div>
""", unsafe_allow_html=True)
