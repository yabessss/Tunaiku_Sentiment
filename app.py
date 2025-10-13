
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

# Misal kamu punya data uji disimpan di 'test_data.csv'
data = pd.read_csv('data_tunaiku.csv')
X_test = vectorizer.transform(data['final_text'])
y_test = data['label']

accuracy = model.score(X_test, y_test)

# Judul aplikasi
st.title('Sentiment Analisis Pengguna Aplikasi Tunaiku')

tab1,tab2 = st.tabs(['Prediksi Sentimen Analisis Pengguna','Informasi Model Analisis Sentimen'])

with tab1:
    st.subheader("📊 Informasi Akurasi Model (Model SVM + XGBoost)")
    st.metric(label="Akurasi Model", value=f"{accuracy*100:.2f}%")
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
with tab2:
    st.image('acc_graph.png')
    st.write('')
    st.write('Pada saat eksperimen membuat model prediksi sentimen pengguna, kami bereksperimen dengan menggunakan 5 jenis model yaitu: Naive Bayes, SVM, Logistic Regression, XGBoost, dan SVM + XGBoost. Seperti yang dapat dilihat dari gambar diatas hasil akurasi model berhasil didapatkan oleh model yang dibuat dari gabungan antara XGBoost dan SVM, dan hasil tersebut membuat kami memilih model tersebut untuk dideploy sebagai model prediksi sentimen pengguna aplikasi Tunaiku.')
