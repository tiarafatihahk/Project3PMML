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

# Urutan fitur harus sama persis dengan saat pelatihan model
FEATURE_ORDER = [
    'network_packet_size', 'login_attempts', 'ip_reputation_score', 'failed_logins',
    'unusual_time_access', 'browser_type_Chrome', 'browser_type_Edge', 'browser_type_Firefox',
    'browser_type_Safari', 'browser_type_Unknown', 'protocol_type_ICMP', 'protocol_type_TCP',
    'protocol_type_UDP', 'encryption_used_AES', 'encryption_used_DES', 'encryption_used_unencrypted'
]
# Kolom numerik yang akan di-scaling
NUMERIC_FEATURES = ['network_packet_size', 'login_attempts', 'ip_reputation_score', 'failed_logins']

# Nama kolom input dari file upload
INPUT_COL_LOGIN_ATTEMPTS = 'login_attempts'
INPUT_COL_FAILED_LOGINS = 'failed_logins'
INPUT_COL_IP_REPUTATION = 'ip_reputation_score'
INPUT_COL_PACKET_SIZE = 'network_packet_size'
INPUT_COL_UNUSUAL_TIME = 'unusual_time_access'
INPUT_COL_BROWSER = 'browser_type'
INPUT_COL_PROTOCOL = 'protocol_type'
INPUT_COL_ENCRYPTION = 'encryption_used'

# Opsi untuk fitur kategorikal
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
        st.error(f"File scaler '{scaler_path}' tidak ditemukan! Harap simpan dari skrip pelatihan Anda.")
        return None
    return joblib.load(scaler_path)

# ======================================================================================
# FUNGSI PRA-PEMROSESAN DATA (DIREVISI AGAR LEBIH ROBUST)
# ======================================================================================
def preprocess_dataframe(df_input, scaler):
    df = df_input.copy()
    processed_rows = []

    # Scaling dilakukan sekali di awal untuk efisiensi
    # Pastikan semua kolom numerik ada sebelum scaling
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0 # Jika kolom numerik hilang, isi dengan 0
    
    numeric_data = df[NUMERIC_FEATURES]
    scaled_numeric_data = scaler.transform(numeric_data)
    df_scaled = pd.DataFrame(scaled_numeric_data, columns=NUMERIC_FEATURES, index=df.index)

    # Iterasi per baris untuk memproses fitur dengan aman
    for i, row in df.iterrows():
        final_model_inputs = {}
        
        # 1. Ambil nilai numerik yang sudah di-scale untuk baris ini
        scaled_row = df_scaled.loc[i]
        final_model_inputs.update(scaled_row.to_dict())

        # 2. Proses fitur kategorikal satu per satu menggunakan .get() untuk keamanan
        #    Jika kolom tidak ada di file upload, .get() akan menggunakan nilai default.
        
        # FIX UNTUK 'unusual_time_access'
        unusual_time_val = row.get(INPUT_COL_UNUSUAL_TIME, "Tidak") # Default "Tidak" jika kolom hilang
        final_model_inputs['unusual_time_access'] = 1 if unusual_time_val == "Ya" else 0

        browser_val = row.get(INPUT_COL_BROWSER, "Unknown")
        for option in BROWSER_OPTIONS:
            final_model_inputs[f'browser_type_{option}'] = 1 if browser_val == option else 0

        protocol_val = row.get(INPUT_COL_PROTOCOL, "TCP")
        for option in PROTOCOL_OPTIONS:
            final_model_inputs[f'protocol_type_{option}'] = 1 if protocol_val == option else 0
        
        encryption_val = row.get(INPUT_COL_ENCRYPTION, "None")
        encryption_mapped = "unencrypted" if encryption_val == "None" else encryption_val
        for option in ENCRYPTION_OPTIONS:
            final_model_inputs[f"encryption_used_{option}"] = 1 if encryption_mapped == option else 0

        processed_rows.append(final_model_inputs)

    # Buat DataFrame final, pastikan semua kolom ada dan dalam urutan yang benar
    df_processed = pd.DataFrame(processed_rows)
    df_processed = df_processed.reindex(columns=FEATURE_ORDER, fill_value=0)
    
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
# ANTARMUKA PENGGUNA (UI)
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
                        
                        st.subheader("📈 Ringkasan dan Statistik Hasil Analisis")
                        jumlah_total_data = len(df_tampilan_akhir)
                        jumlah_terancam = len(df_tampilan_akhir[df_tampilan_akhir['Status Deteksi'] == 'Terancam'])
                        jumlah_aman = jumlah_total_data - jumlah_terancam
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric(label="Total Data Dianalisis", value=f"{jumlah_total_data}")
                        col2.metric(label="🚨 Data Terdeteksi Ancaman", value=f"{jumlah_terancam}")
                        col3.metric(label="✅ Data Terdeteksi Aman", value=f"{jumlah_aman}")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
            st.exception(e)
else:
    st.warning("Gagal memuat model dan/atau scaler. Pastikan file `best_random_forest_model (1).pkl` dan `scaler.pkl` ada di folder yang benar.")
