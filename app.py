import streamlit as st
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load Model
model = joblib.load('model_churn_terbaik.joblib')

st.title("Aplikasi Prediksi Churn Pelanggan Telco")
st.write("Dibuat untuk UAS Bengkel Koding Data Science")

# Sidebar untuk Input User
st.sidebar.header("Masukkan Data Pelanggan")

def user_input_features():
    # Fitur Numerik
    tenure = st.sidebar.number_input('Lama Berlangganan (bulan)', min_value=0, max_value=100, value=12)
    MonthlyCharges = st.sidebar.number_input('Biaya Bulanan', min_value=0.0, value=50.0)
    TotalCharges = st.sidebar.number_input('Total Biaya', min_value=0.0, value=500.0)

    # Fitur Kategorikal (Contoh sebagian, lengkapi sesuai dataset)
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    Partner = st.sidebar.selectbox('Punya Pasangan?', ('Yes', 'No'))
    Dependents = st.sidebar.selectbox('Punya Tanggungan?', ('Yes', 'No'))
    PhoneService = st.sidebar.selectbox('Layanan Telepon', ('Yes', 'No'))
    MultipleLines = st.sidebar.selectbox('Multiple Lines', ('No phone service', 'No', 'Yes'))
    InternetService = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    OnlineSecurity = st.sidebar.selectbox('Online Security', ('No', 'Yes', 'No internet service'))
    OnlineBackup = st.sidebar.selectbox('Online Backup', ('No', 'Yes', 'No internet service'))
    DeviceProtection = st.sidebar.selectbox('Device Protection', ('No', 'Yes', 'No internet service'))
    TechSupport = st.sidebar.selectbox('Tech Support', ('No', 'Yes', 'No internet service'))
    StreamingTV = st.sidebar.selectbox('Streaming TV', ('No', 'Yes', 'No internet service'))
    StreamingMovies = st.sidebar.selectbox('Streaming Movies', ('No', 'Yes', 'No internet service'))
    Contract = st.sidebar.selectbox('Kontrak', ('Month-to-month', 'One year', 'Two year'))
    PaperlessBilling = st.sidebar.selectbox('Paperless Billing', ('Yes', 'No'))
    PaymentMethod = st.sidebar.selectbox('Metode Pembayaran',
                                         ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    SeniorCitizen = st.sidebar.selectbox('Senior Citizen (0=No, 1=Yes)', (0, 1))

    data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Tampilkan input user
st.subheader('Data Pelanggan')
st.write(input_df)

# Tombol Prediksi
if st.button('Prediksi Churn'):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.error(f"Prediksi: CHURN (Berhenti Berlangganan). Probabilitas: {probability[0][1]:.2f}")
    else:

        st.success(f"Prediksi: TIDAK CHURN (Tetap Berlangganan). Probabilitas: {probability[0][0]:.2f}")

