
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import time

# Load model dan vectorizer
model = pickle.load(open('voting_clf.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Judul aplikasi
st.title('Sentiment Analisis Pengguna Aplikasi Tunaiku')

# Input review
coms = st.text_input('Masukkan Review Anda Tentang Aplikasi Kami')

# Tombol submit
submit = st.button('Prediksi')

if submit:
    if coms.strip() == "":
        st.warning("Mohon masukkan review terlebih dahulu.")
    else:
        start = time.time()
        transformed_text = vectorizer.transform([coms]).toarray()
        transformed_text = transformed_text.reshape(1, -1)
        prediction = model.predict(transformed_text)
        end = time.time()

        st.write('Waktu prediksi: ', round(end - start, 2), 'detik')
        if prediction[0] == 1:
            st.success("Sentimen review Anda **positif**.")
        else:
            st.error("Sentimen review Anda **negatif**.")
