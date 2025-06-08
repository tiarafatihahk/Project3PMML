import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# ======================================================================================
# KONFIGURASI PUSAT
# ======================================================================================
MODEL_FILENAME = 'best_random_forest_model (2) try.pkl'
SCALER_FILENAME = 'scaler (1).pkl'

FEATURE_ORDER = [
    'network_packet_size', 'login_attempts', 'ip_reputation_score', 'failed_logins',
    'unusual_time_access', 'browser_type_Chrome', 'browser_type_Edge', 'browser_type_Firefox',
    'browser_type_Safari', 'browser_type_Unknown', 'protocol_type_ICMP', 'protocol_type_TCP',
    'protocol_type_UDP', 'encryption_used_AES', 'encryption_used_DES', 'encryption_used_unencrypted'
]
# Kolom numerik yang akan di-scaling (SEKARANG SUDAH BENAR - HANYA 4)
NUMERIC_FEATURES = ['network_packet_size', 'login_attempts', 'ip_reputation_score', 'failed_logins']

INPUT_COL_LOGIN_ATTEMPTS = 'login_attempts'
INPUT_COL_FAILED_LOGINS = 'failed_logins'
INPUT_COL_IP_REPUTATION = 'ip_reputation_score'
INPUT_COL_PACKET_SIZE = 'network_packet_size'
INPUT_COL_UNUSUAL_TIME = 'unusual_time_access'
INPUT_COL_BROWSER = 'browser_type'
INPUT_COL_PROTOCOL = 'protocol_type'
INPUT_COL_ENCRYPTION = 'encryption_used'

BROWSER_OPTIONS = ["Chrome", "Firefox", "Edge", "Safari", "Unknown"]
PROTOCOL_OPTIONS = ["TCP", "UDP", "ICMP"]
ENCRYPTION_OPTIONS = ["AES", "DES", "unencrypted"]

# ======================================================================================
# FUNGSI-FUNGSI PEMUAT
# ======================================================================================
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path): return None
    return joblib.load(model_path)

@st.cache_resource
def load_scaler(scaler_path):
    if not os.path.exists(scaler_path):
        st.error(f"File scaler '{scaler_path}' tidak ditemukan!")
        return None
    return joblib.load(scaler_path)

# ======================================================================================
# FUNGSI PRA-PEMROSESAN (VERSI FINAL YANG BENAR)
# ======================================================================================
def preprocess_dataframe(df_input, scaler):
    df = df_input.copy()
    
    # Pastikan semua kolom input ada, jika tidak ada isi dengan nilai default
    required_cols = NUMERIC_FEATURES + [INPUT_COL_UNUSUAL_TIME, INPUT_COL_BROWSER, INPUT_COL_PROTOCOL, INPUT_COL_ENCRYPTION]
    for col in required_cols:
        if col not in df.columns:
            if col in NUMERIC_FEATURES:
                df[col] = 0
            else:
                df[col] = "Unknown" 

    # 1. Pisahkan dan Scale HANYA 4 data numerik
    numeric_data = df[NUMERIC_FEATURES]
    scaled_numeric_data = scaler.transform(numeric_data)
    df_scaled = pd.DataFrame(scaled_numeric_data, columns=NUMERIC_FEATURES, index=df.index)

    # 2. Proses fitur kategorikal secara terpisah
    categorical_features = {}
    
    categorical_features['unusual_time_access'] = df[INPUT_COL_UNUSUAL_TIME].apply(lambda x: 1 if x == "Ya" else 0)
    
    for option in BROWSER_OPTIONS:
        categorical_features[f'browser_type_{option}'] = df[INPUT_COL_BROWSER].apply(lambda x: 1 if x == option else 0)
    
    for option in PROTOCOL_OPTIONS:
        categorical_features[f'protocol_type_{option}'] = df[INPUT_COL_PROTOCOL].apply(lambda x: 1 if x == option else 0)

    encryption_mapped = df[INPUT_COL_ENCRYPTION].apply(lambda x: "unencrypted" if x == "None" else x)
    for option in ENCRYPTION_OPTIONS:
        categorical_features[f"encryption_used_{option}"] = encryption_mapped.apply(lambda x: 1 if x == option else 0)

    df_categorical_processed = pd.DataFrame(categorical_features, index=df.index)

    # 3. Gabungkan kembali data numerik yang sudah di-scale dengan data kategorikal
    df_combined = pd.concat([df_scaled, df_categorical_processed], axis=1)

    # 4. Pastikan urutan final benar
    df_processed = df_combined.reindex(columns=FEATURE_ORDER, fill_value=0)
    
    return df_processed

# ======================================================================================
# FUNGSI PREDIKSI
# ======================================================================================
def model_prediksi_ancaman_dataset(model, df_processed):
    if df_processed.empty: return pd.DataFrame()
    predictions = model.predict(df_processed)
    prediction_probas = model.predict_proba(df_processed)
    df_hasil = pd.DataFrame({
        'Status Deteksi': ['Terancam' if p == 1 else 'Aman' for p in predictions],
        'Probabilitas Ancaman (%)': (prediction_probas[:, 1] * 100).round(2)
    })
    return df_hasil

# ======================================================================================
# UI
# ======================================================================================
st.set_page_config(page_title="IDS Dashboard", layout="wide", page_icon="🛡️")
st.title("🛡️ Intrusion Detection System (IDS) Dashboard")

model = load_model(MODEL_FILENAME)
scaler = load_scaler(SCALER_FILENAME)

if model and scaler:
    st.sidebar.header("Unggah Dataset Aktivitas Jaringan")
    uploaded_file = st.sidebar.file_uploader("Pilih file CSV (pemisah ';') atau Excel", type=["csv", "xlsx", "xls"])
    
    if uploaded_file:
        try:
            df_input_original = pd.read_csv(uploaded_file, sep=';') if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.subheader("📄 Data Awal yang Diunggah")
            st.dataframe(df_input_original.head(), use_container_width=True)

            if st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True):
                with st.spinner("Menganalisis data... ⏳"):
                    df_processed_for_model = preprocess_dataframe(df_input_original, scaler)
                    df_prediksi_hasil = model_prediksi_ancaman_dataset(model, df_processed_for_model)

                    if not df_prediksi_hasil.empty:
                        df_tampilan_akhir = pd.concat([df_input_original.reset_index(drop=True), df_prediksi_hasil.reset_index(drop=True)], axis=1)
                        st.subheader("📊 Hasil Analisis Lengkap")
                        st.dataframe(df_tampilan_akhir, use_container_width=True)
                        
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
            st.exception(e)
else:
    st.warning("Gagal memuat model dan/atau scaler. Pastikan file .pkl ada di folder yang benar.")
