import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. IMPORT LIB PENDUKUNG (PENTING AGAR MODEL TERBACA) ---
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# --- 2. LOAD MODEL ---
st.title("Aplikasi Prediksi Churn Pelanggan")
st.write("UAS Bengkel Koding - Data Science")

try:
    model = joblib.load('model_churn_terbaik.joblib')
except Exception as e:
    st.error(f"Gagal memuat model. Error: {e}")
    st.stop()

# --- 3. INPUT USER (SIDEBAR) ---
st.sidebar.header("Masukkan Data Pelanggan")

def user_input_features():
    # Input Numerik
    tenure = st.sidebar.number_input('Lama Berlangganan (Bulan)', min_value=0, value=12)
    MonthlyCharges = st.sidebar.number_input('Biaya Bulanan', min_value=0.0, value=50.0)
    TotalCharges = st.sidebar.number_input('Total Biaya', min_value=0.0, value=500.0)

    # Input Kategorikal (Sesuaikan pilihan dengan data asli Anda)
    Contract = st.sidebar.selectbox('Jenis Kontrak', ('Month-to-month', 'One year', 'Two year'))
    InternetService = st.sidebar.selectbox('Layanan Internet', ('DSL', 'Fiber optic', 'No'))
    PaymentMethod = st.sidebar.selectbox('Metode Pembayaran', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    
    # ... Tambahkan input lain sesuai fitur yang Anda pakai saat training ...
    
    # Simpan dalam DataFrame
    data = {
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'Contract': Contract,
        'InternetService': InternetService,
        'PaymentMethod': PaymentMethod
        # ... Masukkan kolom lain di sini ...
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Tampilkan Input
st.subheader('Data yang dimasukkan:')
st.write(input_df)

# --- 4. PREDIKSI ---
if st.button('Prediksi Sekarang'):
    try:
        prediction = model.predict(input_df)
        
        if prediction[0] == 1:
            st.error("Hasil: Pelanggan Berpotensi CHURN (Berhenti)")
        else:
            st.success("Hasil: Pelanggan SETIA (Tidak Churn)")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
        st.warning("Pastikan kolom input di app.py SAMA PERSIS dengan kolom saat training model.")
