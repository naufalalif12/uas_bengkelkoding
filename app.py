import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Import library Scikit-Learn agar joblib mengenali struktur model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Judul Aplikasi
st.title("Telco Customer Churn Prediction")
st.write("Aplikasi Prediksi Berhenti Langganan (UAS Data Science)")

# Load Model dengan Error Handling
try:
    model = joblib.load('model_churn_terbaik.joblib')
except Exception as e:
    st.error(f"Error memuat model: {e}")
    st.stop()

st.sidebar.header("Input Data Pelanggan")

# Fungsi Input User
def user_input_features():
    # Input Numerik
    tenure = st.sidebar.number_input('Lama Berlangganan (Bulan)', min_value=0, value=12)
    MonthlyCharges = st.sidebar.number_input('Biaya Bulanan', min_value=0.0, value=70.0)
    TotalCharges = st.sidebar.number_input('Total Biaya', min_value=0.0, value=1000.0)

    # Input Kategorikal (Contoh 5 fitur utama, sesuaikan jika ingin lengkap)
    Contract = st.sidebar.selectbox('Jenis Kontrak', ('Month-to-month', 'One year', 'Two year'))
    InternetService = st.sidebar.selectbox('Layanan Internet', ('DSL', 'Fiber optic', 'No'))
    PaymentMethod = st.sidebar.selectbox('Metode Pembayaran', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    OnlineSecurity = st.sidebar.selectbox('Keamanan Online', ('No', 'Yes', 'No internet service'))
    TechSupport = st.sidebar.selectbox('Support Teknis', ('No', 'Yes', 'No internet service'))

    # Kita harus membuat DataFrame dengan KOLOM LENGKAP seperti saat training
    # Trik: Kita buat data dummy untuk kolom yang tidak diinput user agar model tidak error
    data = {
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'Contract': Contract,
        'InternetService': InternetService,
        'PaymentMethod': PaymentMethod,
        'OnlineSecurity': OnlineSecurity,
        'TechSupport': TechSupport,
        # Default value untuk kolom lain yang tidak ada di input sidebar (agar shape sama)
        'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
        'PhoneService': 'Yes', 'MultipleLines': 'No', 'OnlineBackup': 'No',
        'DeviceProtection': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
        'PaperlessBilling': 'Yes'
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Tampilkan Data Input
st.subheader('Data Pelanggan:')
st.write(input_df)

# Tombol Prediksi [cite: 95]
if st.button('Prediksi'):
    try:
        # Lakukan prediksi
        prediction = model.predict(input_df)
        probabilitas = model.predict_proba(input_df)
        
        # Tampilkan Hasil [cite: 96]
        if prediction[0] == 1:
            st.error(f"HASIL: CHURN (Berpotensi Berhenti). Probabilitas: {probabilitas[0][1]:.2f}")
        else:
            st.success(f"HASIL: TIDAK CHURN (Pelanggan Setia). Probabilitas: {probabilitas[0][0]:.2f}")
            
    except Exception as e:
        st.error("Terjadi kesalahan saat prediksi.")
        st.warning(f"Detail Error: {e}")
