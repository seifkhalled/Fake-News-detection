# app.py
import streamlit as st
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import os
import gdown

# ----------------------------
# Download model from Google Drive if not exists
# ----------------------------
model_path = "fake_news_lstm.h5"
file_id = "1_d0vt4JLwCwGgV0BZOeWLRmJIUhMk6Iz"  # Google Drive file ID
if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# Load LSTM model and compile
model = load_model(model_path)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load tokenizer (keep in repo)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Define stopwords
stop_words = set(ENGLISH_STOP_WORDS)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()  # simple tokenization
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# Streamlit UI
st.title("Fake News Detection (Real or Fake)")

title_input = st.text_input("Enter Title")
text_input = st.text_area("Enter Text")

if st.button("Predict"):
    if title_input.strip() == "" or text_input.strip() == "":
        st.warning("Please enter both title and text!")
    else:
        # Combine title + text
        combined_text = title_input + " " + text_input
        
        # Preprocess
        processed_text = preprocess_text(combined_text)
        
        # Tokenize + pad
        seq = tokenizer.texts_to_sequences([processed_text])
        padded_seq = pad_sequences(seq, maxlen=1000)  # same maxlen as training
        
        # Predict
        prediction = model.predict(padded_seq)
        pred_class = int(np.round(prediction[0][0]))
        result = "Fake" if pred_class == 1 else "Real"
        
        st.success(f"Prediction: {result}")
        st.write(f"Raw model output (probability of Fake): {prediction[0][0]:.4f}")
