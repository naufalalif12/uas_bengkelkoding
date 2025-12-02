import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- IMPORT PENTING AGAR MODEL TERBACA (JANGAN DIHAPUS) ---
# Mengimpor komponen Scikit-Learn yang kemungkinan tersimpan di dalam file .joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📡",
    layout="wide"
)

# --- 1. PEMUATAN MODEL [cite: 93] ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model_churn_terbaik.joblib')
        return model
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file .joblib ada. Error: {e}")
        return None

model = load_model()

# --- JUDUL & DESKRIPSI [cite: 97] ---
st.title("📡 Aplikasi Prediksi Churn Pelanggan")
st.markdown("""
Aplikasi ini dirancang untuk memprediksi apakah seorang pelanggan telekomunikasi akan **berhenti berlangganan (Churn)** atau **tetap berlangganan** berdasarkan data profil dan pola penggunaan mereka.
""")
st.markdown("---")

# --- 2. FORM INPUT FITUR (SIDEBAR) [cite: 94] ---
st.sidebar.header("📝 Masukkan Data Pelanggan")

def user_input_features():
    # Group 1: Profil Demografis
    st.sidebar.subheader("Profil Pelanggan")
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    SeniorCitizen = st.sidebar.selectbox('Senior Citizen (Lansia?)', (0, 1), format_func=lambda x: "Ya" if x == 1 else "Tidak")
    Partner = st.sidebar.selectbox('Memiliki Pasangan?', ('Yes', 'No'))
    Dependents = st.sidebar.selectbox('Memiliki Tanggungan?', ('Yes', 'No'))

    # Group 2: Layanan Berlangganan
    st.sidebar.subheader("Layanan")
    tenure = st.sidebar.slider('Lama Berlangganan (Bulan)', 0, 72, 12)
    PhoneService = st.sidebar.selectbox('Layanan Telepon', ('Yes', 'No'))
    MultipleLines = st.sidebar.selectbox('Multiple Lines', ('No phone service', 'No', 'Yes'))
    InternetService = st.sidebar.selectbox('Jenis Internet', ('DSL', 'Fiber optic', 'No'))
    
    # Layanan Tambahan (Hanya muncul jika punya internet, tapi kita tampilkan semua untuk kemudahan)
    OnlineSecurity = st.sidebar.selectbox('Keamanan Online', ('No', 'Yes', 'No internet service'))
    OnlineBackup = st.sidebar.selectbox('Backup Online', ('No', 'Yes', 'No internet service'))
    DeviceProtection = st.sidebar.selectbox('Proteksi Perangkat', ('No', 'Yes', 'No internet service'))
    TechSupport = st.sidebar.selectbox('Support Teknis', ('No', 'Yes', 'No internet service'))
    StreamingTV = st.sidebar.selectbox('Streaming TV', ('No', 'Yes', 'No internet service'))
    StreamingMovies = st.sidebar.selectbox('Streaming Movies', ('No', 'Yes', 'No internet service'))

    # Group 3: Akun & Pembayaran
    st.sidebar.subheader("Informasi Akun")
    Contract = st.sidebar.selectbox('Kontrak', ('Month-to-month', 'One year', 'Two year'))
    PaperlessBilling = st.sidebar.selectbox('Tagihan Paperless?', ('Yes', 'No'))
    PaymentMethod = st.sidebar.selectbox('Metode Pembayaran', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    MonthlyCharges = st.sidebar.number_input('Biaya Bulanan ($)', min_value=0.0, value=50.0)
    TotalCharges = st.sidebar.number_input('Total Biaya ($)', min_value=0.0, value=tenure * 50.0)

    # Menggabungkan data menjadi DataFrame
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

# Menampung input user
input_df = user_input_features()

# --- TAMPILAN INPUT USER ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Review Data Pelanggan")
    st.dataframe(input_df)

# --- 3. PROSES PREDIKSI & 4. TAMPILAN HASIL [cite: 95, 96] ---
with col2:
    st.subheader("⚡ Prediksi")
    predict_btn = st.button("Analisis Churn", type="primary")

if predict_btn:
    if model:
        try:
            # Prediksi Kelas (0 atau 1)
            prediction = model.predict(input_df)
            # Prediksi Probabilitas (Persentase)
            probability = model.predict_proba(input_df)
            
            # Mengambil probabilitas churn (kelas 1)
            churn_prob = probability[0][1]

            st.markdown("### Hasil Analisis:")
            
            # Logika Tampilan Hasil
            if prediction[0] == 1:
                st.error(f"🚨 **BERPOTENSI CHURN (Berhenti)**")
                st.write(f"Tingkat Risiko: **{churn_prob*100:.2f}%**")
                st.progress(int(churn_prob * 100))
                st.warning("Rekomendasi: Tawarkan diskon atau perpanjangan kontrak segera.")
            else:
                st.success(f"✅ **PELANGGAN SETIA (Tidak Churn)**")
                st.write(f"Tingkat Risiko: **{churn_prob*100:.2f}%**")
                st.progress(int(churn_prob * 100))
                st.info("Pelanggan ini cenderung aman. Pertahankan layanan.")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
            st.warning("Tips: Pastikan nama kolom input sama persis dengan dataset training.")
    else:
        st.error("Model belum dimuat.")

# --- 5. ELEMEN PENDUKUNG (VISUALISASI/INFO) [cite: 97] ---
st.markdown("---")
with st.expander("ℹ️ Penjelasan Fitur Penting"):
    st.markdown("""
    * **Tenure**: Berapa lama pelanggan sudah berlangganan (bulan). Semakin lama, biasanya semakin setia.
    * **Contract**: Jenis kontrak sangat mempengaruhi. Kontrak 'Month-to-month' biasanya lebih rentan Churn.
    * **Internet Service**: Pengguna Fiber Optic seringkali memiliki tagihan lebih tinggi dan pola churn berbeda.
    * **Monthly Charges**: Biaya bulanan yang terlalu tinggi bisa memicu pelanggan pindah ke kompetitor.
    """)
