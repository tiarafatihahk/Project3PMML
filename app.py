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
MODEL_FILENAME = 'model_final.pkl'
SCALER_FILENAME = 'scaler_final.pkl'

FEATURE_ORDER = [
    'network_packet_size', 'login_attempts', 'ip_reputation_score', 'failed_logins',
    'unusual_time_access', 'browser_type_Chrome', 'browser_type_Edge', 'browser_type_Firefox',
    'browser_type_Safari', 'browser_type_Unknown', 'protocol_type_ICMP', 'protocol_type_TCP',
    'protocol_type_UDP', 'encryption_used_AES', 'encryption_used_DES', 'encryption_used_unencrypted'
]
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
ENCRYPTION_OPTIONS_DROPDOWN = ["AES", "DES", "None"] # Untuk dropdown
ENCRYPTION_OPTIONS_MODEL = ["AES", "DES", "unencrypted"] # Untuk pemrosesan

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
# FUNGSI PRA-PEMROSESAN DATA
# ======================================================================================
def preprocess_dataframe(df_input, scaler):
    df = df_input.copy()
    
    required_cols = NUMERIC_FEATURES + [INPUT_COL_UNUSUAL_TIME, INPUT_COL_BROWSER, INPUT_COL_PROTOCOL, INPUT_COL_ENCRYPTION]
    for col in required_cols:
        if col not in df.columns:
            if col in NUMERIC_FEATURES:
                df[col] = 0
            else:
                df[col] = "Unknown" 

    numeric_data = df[NUMERIC_FEATURES]
    scaled_numeric_data = scaler.transform(numeric_data)
    df_scaled = pd.DataFrame(scaled_numeric_data, columns=NUMERIC_FEATURES, index=df.index)

    categorical_features = {}
    categorical_features['unusual_time_access'] = df[INPUT_COL_UNUSUAL_TIME].apply(lambda x: 1 if x == "Ya" else 0)
    
    for option in BROWSER_OPTIONS:
        categorical_features[f'browser_type_{option}'] = df[INPUT_COL_BROWSER].apply(lambda x: 1 if x == option else 0)
    
    for option in PROTOCOL_OPTIONS:
        categorical_features[f'protocol_type_{option}'] = df[INPUT_COL_PROTOCOL].apply(lambda x: 1 if x == option else 0)

    encryption_mapped = df[INPUT_COL_ENCRYPTION].apply(lambda x: "unencrypted" if x == "None" else x)
    for option in ENCRYPTION_OPTIONS_MODEL:
        categorical_features[f"encryption_used_{option}"] = encryption_mapped.apply(lambda x: 1 if x == option else 0)

    df_categorical_processed = pd.DataFrame(categorical_features, index=df.index)

    df_combined = pd.concat([df_scaled, df_categorical_processed], axis=1)
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
# ANTARMUKA PENGGUNA (UI)
# ======================================================================================
st.set_page_config(page_title="IDS Dashboard", layout="wide", page_icon="🛡️")
st.title("🛡️ Dashboard Intrusion Detection System (IDS)")

# --- Muat Model dan Scaler ---
model = load_model(MODEL_FILENAME)
scaler = load_scaler(SCALER_FILENAME)

# --- Sidebar ---
st.sidebar.header("⚙️ Opsi Analisis")
analysis_mode = st.sidebar.radio("Pilih Mode Analisis:", ('Unggah File', 'Input Manual'))
st.sidebar.markdown("---")

# --- UI untuk Input Manual ---
if analysis_mode == 'Input Manual':
    st.sidebar.subheader("Form Prediksi Manual")
    with st.sidebar.form(key='manual_input_form'):
        # Input Numerik
        login_attempts = st.number_input('Jumlah Percobaan Login', min_value=0, max_value=20, value=3, step=1)
        failed_logins = st.number_input('Jumlah Login Gagal', min_value=0, max_value=10, value=1, step=1)
        ip_reputation_score = st.slider('Skor Reputasi IP', 0, 100, 50)
        network_packet_size = st.number_input('Ukuran Paket Jaringan (byte)', min_value=0, value=512)
        
        # Input Kategorikal
        unusual_time_access = st.selectbox('Akses di Waktu Tidak Wajar?', ('Tidak', 'Ya'))
        browser_type = st.selectbox('Tipe Browser', BROWSER_OPTIONS)
        protocol_type = st.selectbox('Tipe Protokol', PROTOCOL_OPTIONS)
        encryption_used = st.selectbox('Enkripsi yang Digunakan', ENCRYPTION_OPTIONS_DROPDOWN)

        submit_button = st.form_submit_button(label='🔍 Prediksi Sekarang')

# --- Tampilan Utama ---
if not model or not scaler:
    st.error("Gagal memuat file `model_terbaik.pkl` dan/atau `scaler.pkl`. Pastikan file-file tersebut ada di folder yang sama dengan `app.py`.")
else:
    if analysis_mode == 'Unggah File':
        uploaded_file = st.file_uploader("Unggah Dataset (CSV atau Excel)", type=["csv", "xlsx"])
        if uploaded_file is None:
            st.info("Silakan unggah file dataset untuk memulai analisis, atau pilih 'Input Manual' di sidebar.")
        else:
            try:
                df_input_original = pd.read_csv(uploaded_file, sep=';') if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.subheader("📄 Pratinjau Data yang Diunggah")
                st.dataframe(df_input_original.head(), use_container_width=True)

                if st.button("🚀 Jalankan Analisis File", type="primary", use_container_width=True):
                    with st.spinner("Menganalisis data..."):
                        df_processed = preprocess_dataframe(df_input_original, scaler)
                        df_prediction = model_prediksi_ancaman_dataset(model, df_processed)
                        df_final = pd.concat([df_input_original.reset_index(drop=True), df_prediction.reset_index(drop=True)], axis=1)
                        
                        st.markdown("---")
                        st.subheader("📊 Hasil Analisis Lengkap")
                        st.dataframe(df_final, use_container_width=True)
                        # Anda bisa menambahkan visualisasi di sini jika diinginkan
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file: {e}")

    elif analysis_mode == 'Input Manual':
        st.info("Silakan isi formulir di sidebar kiri dan klik 'Prediksi Sekarang' untuk melihat hasilnya di sini.")
        if submit_button:
            # Kumpulkan input menjadi dictionary
            manual_data = {
                INPUT_COL_LOGIN_ATTEMPTS: [login_attempts],
                INPUT_COL_FAILED_LOGINS: [failed_logins],
                INPUT_COL_IP_REPUTATION: [ip_reputation_score / 100.0], # Normalisasi skor ke 0-1
                INPUT_COL_PACKET_SIZE: [network_packet_size],
                INPUT_COL_UNUSUAL_TIME: [unusual_time_access],
                INPUT_COL_BROWSER: [browser_type],
                INPUT_COL_PROTOCOL: [protocol_type],
                INPUT_COL_ENCRYPTION: [encryption_used]
            }
            # Buat menjadi DataFrame
            df_manual = pd.DataFrame.from_dict(manual_data)

            with st.spinner("Melakukan prediksi..."):
                # Proses dan prediksi
                df_processed = preprocess_dataframe(df_manual, scaler)
                df_prediction = model_prediksi_ancaman_dataset(model, df_processed)

                status = df_prediction['Status Deteksi'].iloc[0]
                probabilitas = df_prediction['Probabilitas Ancaman (%)'].iloc[0]

                st.markdown("---")
                st.subheader("🔍 Hasil Prediksi Manual")

                if status == 'Terancam':
                    st.error(f"**Status: Terdeteksi Ancaman**")
                    st.metric(label="Tingkat Kepercayaan Ancaman", value=f"{probabilitas}%")
                    
                    st.subheader("Analisis Potensi Penyebab:")
                    if ip_reputation_score < 40:
                        st.warning("🚨 **Skor Reputasi IP Rendah:** IP yang digunakan memiliki reputasi yang buruk, ini adalah indikator kuat adanya aktivitas berbahaya.")
                    if failed_logins > 2:
                        st.warning(f"🚨 **Banyak Login Gagal ({failed_logins} kali):** Jumlah percobaan login yang gagal sangat tinggi, mengindikasikan kemungkinan serangan *brute-force*.")
                    if unusual_time_access == 'Ya':
                        st.warning("🕒 **Akses di Waktu Tidak Wajar:** Aktivitas terjadi di luar jam operasional normal, meningkatkan kecurigaan.")
                else:
                    st.success(f"**Status: Aman**")
                    st.metric(label="Tingkat Kepercayaan Ancaman", value=f"{probabilitas}%")
                    st.info("Aktivitas jaringan tidak menunjukkan indikasi ancaman yang signifikan berdasarkan analisis model.")

                with st.expander("Lihat Detail Input Anda"):
                    st.dataframe(df_manual)
