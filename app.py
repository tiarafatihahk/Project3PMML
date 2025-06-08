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
SCALER_FILENAME = 'scaler.pkl'

FEATURE_ORDER = [
    'network_packet_size', 'login_attempts', 'ip_reputation_score', 'failed_logins',
    'unusual_time_access', 'browser_type_Chrome', 'browser_type_Edge', 'browser_type_Firefox',
    'browser_type_Safari', 'browser_type_Unknown', 'protocol_type_ICMP', 'protocol_type_TCP',
    'protocol_type_UDP', 'encryption_used_AES', 'encryption_used_DES', 'encryption_used_unencrypted'
]
# PERBAIKAN: Tambahkan 'unusual_time_access' ke daftar fitur yang akan di-scale
NUMERIC_FEATURES_FOR_SCALING = ['network_packet_size', 'login_attempts', 'ip_reputation_score', 'failed_logins', 'unusual_time_access']

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
# FUNGSI-FUNGSI PEMUAT (LOADER)
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
# FUNGSI PRA-PEMROSESAN DATA (PERBAIKAN UNTUK SCALER)
# ======================================================================================
def preprocess_dataframe(df_input, scaler):
    df = df_input.copy()

    # 1. Siapkan SEMUA fitur yang dibutuhkan oleh scaler
    # PERBAIKAN: Buat kolom biner 'unusual_time_access' TERLEBIH DAHULU
    df['unusual_time_access'] = df[INPUT_COL_UNUSUAL_TIME].apply(lambda x: 1 if x == "Ya" else 0)

    # Pastikan semua kolom untuk scaling ada
    for col in NUMERIC_FEATURES_FOR_SCALING:
        if col not in df.columns:
            df[col] = 0
    
    data_for_scaling = df[NUMERIC_FEATURES_FOR_SCALING]
    
    # 2. Lakukan scaling
    scaled_data = scaler.transform(data_for_scaling)
    df_scaled = pd.DataFrame(scaled_data, columns=NUMERIC_FEATURES_FOR_SCALING, index=df.index)

    # 3. Proses fitur kategorikal lainnya (yang tidak di-scale)
    df_categorical_processed = pd.DataFrame(index=df.index)
    
    for option in BROWSER_OPTIONS:
        df_categorical_processed[f'browser_type_{option}'] = df[INPUT_COL_BROWSER].apply(lambda x: 1 if x == option else 0)
    
    for option in PROTOCOL_OPTIONS:
        df_categorical_processed[f'protocol_type_{option}'] = df[INPUT_COL_PROTOCOL].apply(lambda x: 1 if x == option else 0)

    user_choice_encryption = df[INPUT_COL_ENCRYPTION].apply(lambda x: "unencrypted" if x == "None" else x)
    for option in ENCRYPTION_OPTIONS:
        df_categorical_processed[f"encryption_used_{option}"] = user_choice_encryption.apply(lambda x: 1 if x == option else 0)

    # 4. Gabungkan kembali semua fitur
    df_combined = pd.concat([df_scaled, df_categorical_processed], axis=1)

    # 5. Pastikan semua kolom ada dan dalam urutan yang benar
    df_processed = df_combined.reindex(columns=FEATURE_ORDER, fill_value=0)
    
    return df_processed

# ======================================================================================
# SISA KODE (TIDAK ADA PERUBAHAN)
# ======================================================================================
def model_prediksi_ancaman_dataset(model, df_processed):
    # ... (Sama seperti sebelumnya)
    if df_processed.empty: return pd.DataFrame()
    predictions = model.predict(df_processed)
    prediction_probas = model.predict_proba(df_processed)
    df_hasil = pd.DataFrame({
        'Status Deteksi': ['Terancam' if p == 1 else 'Aman' for p in predictions],
        'Probabilitas Ancaman (%)': (prediction_probas[:, 1] * 100).round(2)
    })
    return df_hasil

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
    st.warning("Gagal memuat model dan/atau scaler.")
